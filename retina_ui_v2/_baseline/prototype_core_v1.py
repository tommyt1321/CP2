"""Frozen CP2 A3 adapter. No training, optimizer, calibration or cohort edits."""
from __future__ import annotations
import ast
import base64
from dataclasses import dataclass
import hashlib
import inspect
import io
from pathlib import Path
import threading
import time
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

VERSION = 'prototype_v1_20261113'
GRADIO_VERSION = '6.31.0'
CHECKPOINT_SHA = '921e0368a183287939d0142b4d2a1c2b0c3da5db5a3b97791ee7adff4a947219'
PREPROCESS_AST_SHA = '4142bb504f6925201d2e85e31c04584ee00fd93f59790950dac7daeaa945d06d'
XAI_SHA = {'attention_rollout_v1.py': 'abf98a5f174d8e7730f52873f630d80a548549e99c4e3701f9d9e5aa882ae98e',
           'gat_relations_v1.py': 'eb1a3383ad9b18d6022fcdfcab39a0ed4111bfb2e126fd97615d3c23f793bc07'}
MAX_BYTES = 20 * 1024**2
MAX_PIXELS = 25_000_000
CLASSES = ('No apparent DR', 'Mild non-proliferative DR', 'Moderate non-proliferative DR',
           'Severe non-proliferative DR', 'Proliferative DR')


class InputError(ValueError):
    """User-facing input errors; messages contain no paths or filenames."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_preprocessor(module):
    node = ast.parse(inspect.getsource(module.preprocess)).body[0]
    digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    if digest != PREPROCESS_AST_SHA or module.SIZE != 384 or module.BLACK_THRESHOLD != 7:
        raise ValueError('Accepted preview_preprocessing_v1.preprocess or constants changed.')


def read_upload(path):
    path = Path(path)
    if path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
        raise InputError('Choose a PNG or JPEG fundus image.')
    if not path.is_file() or path.stat().st_size == 0:
        raise InputError('The upload is empty or unavailable. Choose the image again.')
    if path.stat().st_size > MAX_BYTES:
        raise InputError('The image exceeds the 20 MiB upload limit.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as im:
                if im.format not in ('PNG', 'JPEG'):
                    raise InputError('The file content is not a PNG or JPEG image.')
                if im.width * im.height > MAX_PIXELS or min(im.size) < 32:
                    raise InputError('Use an image with at least 32 pixels per side and at most 25 megapixels.')
                if getattr(im, 'n_frames', 1) != 1:
                    raise InputError('Animated images are not supported.')
                if im.getexif().get(274, 1) != 1:
                    raise InputError('This image uses EXIF rotation. Export an upright copy with rotation applied to its pixels.')
                if im.mode not in ('RGB', 'RGBA', 'L', 'P'):
                    raise InputError('Export the image as an 8-bit RGB PNG or JPEG.')
                im.load()
                if 'transparency' in im.info or im.mode == 'RGBA':
                    if im.convert('RGBA').getchannel('A').getextrema() != (255, 255):
                        raise InputError('Transparent images are not supported. Use the original opaque fundus image.')
                return im.convert('RGB').copy()
    except InputError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning, ValueError) as exc:
        raise InputError('The image cannot be decoded safely. Choose a valid PNG or JPEG.') from exc


def image_html(rgb, alt):
    """Inline display avoids creating persistent or Gradio-cached output images."""
    stream = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(stream, format='PNG')
    data = base64.b64encode(stream.getvalue()).decode('ascii')
    return (f'<div class="image-frame"><img alt="{alt}" src="data:image/png;base64,{data}" '
            'style="width:100%;height:100%;max-height:520px;object-fit:contain"/></div>')


@dataclass
class Prepared:
    rgb: np.ndarray
    preview: np.ndarray
    original_size: tuple
    crop_box: tuple
    warning: str
    preparation_seconds: float


def prepare(path, preprocess):
    start = time.perf_counter()
    original = read_upload(path)
    processed, box, warning = preprocess(original)
    if processed.mode != 'RGB' or processed.size != (384, 384):
        raise ValueError('Preprocessor did not return RGB 384x384.')
    preview = ImageOps.contain(original, (640, 640), method=Image.Resampling.LANCZOS)
    return Prepared(np.asarray(processed).copy(), np.asarray(preview).copy(),
                    original.size, tuple(box), warning, time.perf_counter() - start)


def render_relations(evidence, relation_module, sign):
    """Same pair selection, anchors, region/border tint and strength as accepted preview."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import patheffects
    labels, pairs, scores = (evidence[k] for k in ('labels', 'pairs', 'pair_signed'))
    rgb = evidence['rgb_uint8'].astype(float) / 255
    selected = relation_module.ranked_indices(scores, sign)[:relation_module.DISPLAY_PAIRS]
    anchors = relation_module.region_anchors(labels, evidence['geometry'])
    colour = np.array([1., .60, .10]) if sign == 1 else np.array([.1, .80, 1.])
    union = np.zeros(labels.shape, bool); borders = union.copy()
    for i in selected:
        u, v = pairs[i]
        union |= (labels == u) | (labels == v)
        borders |= relation_module.shared_border_pixels(labels, u, v)
    rgb[union] = .78 * rgb[union] + .22 * colour
    rgb[borders] = colour
    fig = Figure(figsize=(5, 5), dpi=120, facecolor='black')
    canvas = FigureCanvasAgg(fig); ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation='nearest'); ax.set_axis_off()
    maximum = float(np.max(sign * scores[selected])) if len(selected) else 0.
    placed = []
    for rank, i in enumerate(selected, 1):
        points = anchors[pairs[i]]; strength = sign * scores[i] / maximum
        line, = ax.plot(points[:, 0], points[:, 1], color=colour,
                        linewidth=1.1 + 2*strength, marker='o', markersize=3)
        line.set_path_effects([patheffects.Stroke(linewidth=2.5 + 2*strength, foreground='black'),
                              patheffects.Normal()])
        position = points.mean(0); candidates = [position]
        for radius in (18, 28, 40, 54):
            candidates += [position + radius*np.array([np.cos(a), np.sin(a)])
                           for a in np.linspace(-np.pi/2, 3*np.pi/2, 8, endpoint=False)]
        candidates = [np.clip(p, 9, np.array(labels.shape[::-1])-10) for p in candidates]
        loc = next((p for p in candidates if all(np.linalg.norm(p-q) >= 18 for q in placed)), candidates[-1])
        placed.append(loc)
        leader = dict(arrowstyle='-', color='white', linewidth=.6) if np.linalg.norm(loc-position)>2 else None
        ax.annotate(str(rank), xy=position, xytext=loc, color='white', ha='center', va='center',
                    fontsize=8, arrowprops=leader,
                    bbox=dict(boxstyle='round,pad=.12', facecolor='black', edgecolor='none', alpha=.8))
    if not len(selected):
        ax.text(.5, .5, 'No pairs with this sign', transform=ax.transAxes,
                color='white', ha='center', bbox=dict(facecolor='black', alpha=.7))
    canvas.draw(); arr = np.asarray(canvas.buffer_rgba())[..., :3].copy(); fig.clear()
    return arr


class Engine:
    """One frozen model; one lock protects hooks, CUDA state and rendering."""
    def __init__(self, model, source_state, ar, relations):
        self.model, self.source_state, self.ar, self.relations = model, source_state, ar, relations
        self.lock = threading.Lock()
        d, torch = ar.d, ar.torch
        self.to_rgb = d.v2.Compose([d.v2.ToImage(), d.v2.ToDtype(torch.float32, scale=True), d.v2.ToPureTensor()])
        self.normalise = d.v2.Normalize(d.MEAN, d.STD, inplace=False)

    def analyse(self, prepared):
        ar, gr, torch = self.ar, self.relations, self.ar.torch
        with self.lock:
            device = next(self.model.parameters()).device
            def sync():
                if device.type == 'cuda': torch.cuda.synchronize(device)
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            sync(); start = time.perf_counter()
            rgb = self.to_rgb(Image.fromarray(prepared.rgb)).contiguous()
            cnn = self.normalise(rgb).contiguous()
            sample = dict(rgb=rgb, cnn=cnn, label=-1, id_code='uploaded_image', split='prototype')
            try:
                rollout, record = ar.explain_sample(self.model, sample)
                sync(); after_rollout = time.perf_counter()
                rel, rel_record = gr.explain_relations(self.model, cnn.unsqueeze(0).to(device), rollout['labels'])
                sync(); after_relations = time.perf_counter()
                np.testing.assert_allclose(rel['logits'], rollout['logits'], atol=5e-4, rtol=5e-4)
                if rel_record['prediction'] != record['prediction']:
                    raise ValueError('Prediction changed between explanation paths.')
                from matplotlib import colormaps
                heat = colormaps['magma'](rollout['pixel_display'])[..., :3]
                overlay = np.rint(255*(.55*(prepared.rgb.astype(float)/255)+.45*heat)).clip(0,255).astype(np.uint8)
                rel['rgb_uint8'] = prepared.rgb
                positive = render_relations(rel, gr, 1)
                negative = render_relations(rel, gr, -1)
                after_render = time.perf_counter()
                probabilities = rollout['probabilities'].tolist()
                if not np.isclose(sum(probabilities), 1., atol=1e-10):
                    raise ValueError('Invalid probabilities.')
                metrics = dict(preparation_seconds=prepared.preparation_seconds,
                               rollout_and_prediction_seconds=after_rollout-start,
                               relations_seconds=after_relations-after_rollout,
                               render_seconds=after_render-after_relations,
                               regions=record['regions'], physical_pairs=rel_record['physical_pairs'],
                               model_forward_calls=4, optimizer_steps=0,
                               peak_cuda_allocated_mib=(torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None))
                result = dict(prediction=record['prediction'], probabilities=probabilities,
                              logits=rollout['logits'].tolist(), overlay=overlay,
                              positive=positive, negative=negative, metrics=metrics)
            finally:
                ar.tr.assert_tree_equal(self.model.state_dict(), self.source_state)
                if any(p.grad is not None or p.requires_grad for p in self.model.parameters()) or any(m.training for m in self.model.modules()):
                    raise ValueError('Frozen model state changed.')
            metrics['backend_compute_seconds'] = time.perf_counter()-start+prepared.preparation_seconds
            metrics['timing_scope'] = 'Backend preparation + 4 forwards/explanations + rendering + state check; excludes browser upload, queue and display.'
            return result


def load_engine(root, device='cuda'):
    import torch
    import attention_rollout_v1 as ar
    import gat_relations_v1 as gr
    import preview_preprocessing_v1 as pp
    root = Path(root).resolve()
    verify_preprocessor(pp)
    for module in (ar, gr):
        name = Path(module.__file__).name
        if sha(module.__file__) != XAI_SHA[name]:
            raise ValueError(f'Frozen explanation code changed: {name}')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Use the original CP2 .venv with its CUDA-enabled PyTorch.')
    class Quiet:
        stage = ''
    started = time.perf_counter()
    project, info, files, codes = ar.locked_context(root, 'A3', Quiet())
    if info['best_sha256'] != CHECKPOINT_SHA or info['summary']['best_epoch'] != 21:
        raise ValueError('This prototype requires the accepted A3 epoch-21 checkpoint.')
    ar.tr.seed_runtime(42)
    model, state = ar.ev.load_best_model(info, torch.device(device))
    engine = Engine(model, state, ar, gr)
    engine.project, engine.preprocess, engine.files, engine.codes = project, pp.preprocess, files, codes
    engine.startup = dict(version=VERSION, checkpoint_sha256=CHECKPOINT_SHA, best_epoch=21,
                          device=device, versions=ar.ev.runtime_versions(),
                          load_and_check_seconds=time.perf_counter()-started,
                          parameters=sum(p.numel() for p in model.parameters()),
                          image_files_decoded=0, model_forwards=0, optimizer_steps=0)
    return engine


def verify_one_training_image(engine):
    """Local acceptance: one fixed TRAIN source plus its canonical processed PNG."""
    ar = engine.ar; row = ar.choose_train_rows(engine.project)[0]
    image_id = row['id_code']; root = engine.project['root']
    raw_path = root/'train_images'/(image_id+'.png')
    processed_path = root/'processed_images_384_v1'/(image_id+'.png')
    before = {p: sha(p) for p in (raw_path, processed_path)}
    if before[processed_path] != engine.project['processed'][image_id]['output_sha256']:
        raise ValueError('Canonical processed training image hash differs.')
    prepared = prepare(raw_path, engine.preprocess)
    with Image.open(processed_path) as im:
        canonical = np.asarray(im).copy()
    np.testing.assert_array_equal(prepared.rgb, canonical)
    result = engine.analyse(prepared)
    rgb = engine.to_rgb(Image.fromarray(canonical)).contiguous()
    cnn = engine.normalise(rgb).unsqueeze(0).to(next(engine.model.parameters()).device)
    with ar.torch.no_grad():
        details = ar.tr.bm.boundary_map(rgb, return_details=True)
        labels = ar.tr.sp.segment_superpixels(details['lab'], details['boundary'][0], boundary_weight=2.)['labels']
        direct = engine.model(cnn, [labels], segmentation_method='boundary_slic', return_attention=False)['logits'][0]
    np.testing.assert_allclose(result['logits'], ar.as_numpy(direct), atol=5e-4, rtol=5e-4)
    np.testing.assert_allclose(result['probabilities'], ar.as_numpy(direct.double().softmax(-1)),
                               atol=5e-4, rtol=5e-4)
    if result['prediction'] != int(direct.argmax()):
        raise ValueError('Prototype/reference argmax differs.')
    ar.tr.assert_tree_equal(engine.model.state_dict(), engine.source_state)
    for path, digest in {**engine.files, **engine.codes, **before}.items():
        if sha(path) != digest: raise ValueError('An upstream input changed during verification.')
    return dict(status='PROTOTYPE_LOCAL_VERIFY_PASS', version=VERSION, training_image_id=image_id,
                checkpoint_sha256=CHECKPOINT_SHA, raw_to_canonical_RGB='exact_pixels_equal',
                direct_prediction='allclose_5e-4_and_same_argmax', parameters_and_BN_unchanged=True,
                direct_probabilities='float64_softmax_allclose_5e-4',
                upstream_files_unchanged=True, validation_test_IDRiD_images_decoded=0,
                unique_training_images=1, image_files_decoded=2, model_forward_calls=5,
                optimizer_steps=0, prediction=result['prediction'], probabilities=result['probabilities'],
                timing=result['metrics'],
                scope='One-image technical acceptance, not new study metrics or full usability/security acceptance.')
