"""Read the current request's existing evidence without rerunning either explanation."""
from contextvars import ContextVar
import numpy as np

_pair_sink = ContextVar("retina_ui_pair_sink", default=None)


class RelationView:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def explain_relations(self, *args, **kwargs):
        evidence, record = self.module.explain_relations(*args, **kwargs)
        sink = _pair_sink.get()
        if sink is not None:
            sink.update({k: record[k] for k in ("prediction", "target_class", "top_positive", "top_negative")})
            # Keep only the arrays needed to draw this request. No hooks, gradients or image copies.
            keys = ("labels", "pairs", "pair_signed", "geometry")
            if all(k in evidence for k in keys):
                sink["_drawing"] = {k: evidence[k] for k in keys}
        return evidence, record


def attach(engine):
    # The engine is private to this process. The frozen module, weights and functions stay intact.
    engine.relations = RelationView(engine.relations)
    return engine


def run(engine, prepared):
    record = {}
    token = _pair_sink.set(record)
    try:
        result = engine.analyse(prepared)
    finally:
        _pair_sink.reset(token)
    if record and not result["prediction"] == record["prediction"] == record["target_class"]:
        raise ValueError("Relationship target does not match the displayed prediction.")
    drawing = record.pop("_drawing", None)
    result["slic_views"] = None
    if drawing is not None:
        try:
            # Render after the original timing and state checks. Serialise Agg as in the baseline.
            with engine.lock:
                result["slic_views"] = relation_boundary_views(prepared.rgb, drawing, engine.relations, result)
        except Exception as exc:
            # An optional drawing must not invalidate or silently replace the original result.
            print("SLIC display unavailable:", type(exc).__name__, flush=True)
    return result, record


def boundary_photo(rgb, labels, count):
    """Draw one thin contour per horizontal/vertical label transition on a display copy."""
    rgb, labels = np.asarray(rgb), np.asarray(labels)
    if (rgb.shape != (384, 384, 3) or rgb.dtype != np.uint8 or labels.shape != rgb.shape[:2]
            or labels.dtype.kind not in "iu" or not 1 <= count <= labels.size
            or not np.array_equal(np.unique(labels), np.arange(count))):
        raise ValueError("The model's region labels and input grid do not agree.")
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    out = rgb.copy()
    pixels = rgb[edge].astype(np.int32)
    light = (pixels @ np.array([54, 183, 19])) >= 160 * 256
    ink = np.where(light[:, None], [40, 54, 54], [226, 234, 232])
    out[edge] = ((3 * pixels + 2 * ink + 2) // 5).astype(np.uint8)
    return out


def relation_boundary_views(rgb, drawing, module, original):
    """Put the unchanged signed links, dots and ranks on top of the true SLIC outlines."""
    from _baseline import prototype_core_v1 as core
    count = int(original["metrics"]["regions"])
    outlined = boundary_photo(rgb, drawing["labels"], count)
    prepared = dict(drawing, rgb_uint8=outlined)
    output = {}
    for key, sign in (("positive", 1), ("negative", -1)):
        if original.get(key) is None:
            raise ValueError("The original relationship overlay is unavailable.")
        image = core.render_relations(prepared, module, sign)
        if image.shape != original[key].shape:
            raise ValueError("The outlined and original relationship views do not align.")
        output[key] = image
    return output


def field_copy(rgb, overlay):
    """The same v1 RGB footprint used by the accepted design. No input or ranking changes."""
    overlay = np.asarray(overlay)
    if np.asarray(rgb).shape != (384, 384, 3) or overlay.shape != (384, 384, 3) or overlay.dtype != np.uint8:
        return None, "The overlay and model input do not share the required pixel grid."
    try:
        from experiments.aptos_fov_v1.fov_regions import fundus_fov
        mask, quality = fundus_fov(rgb)
        if mask.dtype != np.bool_ or mask.shape != overlay.shape[:2] or quality["flags"] or not mask.any() or mask.all():
            return None, "The field boundary could not be estimated reliably for this image."
    except Exception as error:
        print("Display footprint unavailable:", type(error).__name__, flush=True)
        return None, "A reliable retinal field estimate is unavailable for this image."
    view = overlay.copy()
    view[~mask] = 0
    return view, ""
