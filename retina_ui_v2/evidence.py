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
    return result, record


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
