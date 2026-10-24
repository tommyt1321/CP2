r"""CP2 proposal 3.9.1: trained-model Attention Rollout on aligned superpixels.

    python attention_rollout_v1.py
    python attention_rollout_v1.py --resume
    python attention_rollout_v1.py --variant A0

Default: A3's accepted validation-best checkpoint; five fixed TRAIN images,
one lexicographically first ID per grade. No augmentation, training or test
image access. This is a mapping/visualisation preview, not a faithfulness score.
The pure attention_rollout and region_relevance functions are reusable later.

Abnar & Zuidema (ACL 2020): mean heads, add residual identity, row-normalise,
then R_l = A_l @ R_(l-1). Take the final CLS row and match token REGION IDs.
Scores trace ViT input positions, NOT a class-specific gradient or a lesion mask.
For A2/A3 those positions already contain GAT-refined neighbour information.

All trained files remain read-only. A fixed output folder and atomic per-image
commits allow --resume; stopping does not create numbered interrupt files.
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import threading
import time

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading attention-rollout libraries...', flush=True)
import numpy as np
import torch
import evaluate_v1 as ev

tr, d = ev.tr, ev.d
require = tr.require
VERSION = 'attention_rollout_v1'
EVALUATION_SIGNATURE = '4798db9cc4dda8c843eb15104b2fdc88a70a91715af446979b83712500f5aad0'
METHOD = {
    'heads': 'arithmetic mean of all four heads in each of the four ViT blocks',
    'residual': 'add identity on valid tokens, then normalise each valid row',
    'layer_order': 'forward order: joint = residual_attention[layer] @ joint; start at identity',
    'readout': 'final CLS row; retain CLS self mass separately; exclude CLS and padding from region scores',
    'region_score': 'raw CLS rollout mass at that region token; conditional score divides by total region mass',
    'pixel_mapping': 'piecewise constant region score at every pixel with the matching region ID',
    'display': 'raw region score divided by that image maximum; magma; overlay alpha=0.45',
    'display_limits': 'no minimum subtraction, smoothing, threshold, FOV mask or attention discard',
    'area': 'region scores are not divided by area; a replicated pixel heatmap is not a pixel probability density',
    'precision': 'model float32; detached rollout arithmetic on CPU in float64',
    'interpretation': 'class-agnostic ViT token relevance, not predicted-class attribution or causal evidence',
    'scope': 'does not propagate relevance through GAT, CNN, value projections, MLPs or layer normalisation',
    'selection': 'first train ID in lexicographic order for each grade 0..4; independent of predictions',
}
SOURCES = {
    'proposal': 'CP1 3.9.1, printed page 46 (PDF page 52); faithfulness is separate in 3.9.3.',
    'paper': 'Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. '
             'ACL, pp. 4190-4197. https://doi.org/10.18653/v1/2020.acl-main.385',
    'paper_url': 'https://aclanthology.org/2020.acl-main.385/',
    'author_implementation': 'https://github.com/samiraabnar/attention_flow/blob/master/attention_graph_util.py',
    'implementation_scope': 'Residual addition, row normalisation and matrix order follow the paper. '
                            'Padding exclusion and irregular-superpixel mapping adapt it to this model. '
                            'Preview selection, colour map and numerical tolerances are project choices.',
}


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def attention_rollout(attentions, key_padding_mask):
    """Return float64 joint[B,N,N] and mean_attention[L,B,N,N].

    attentions is a forward-ordered sequence of [B,H,N,N] probabilities.
    True mask entries are padding. Padding rows/columns are excluded BEFORE
    residual addition; valid attention rows must already sum to one.
    This is deliberately detached post-hoc analysis, not a gradient operator.
    """
    mask = as_numpy(key_padding_mask)
    require(mask.ndim == 2 and mask.dtype == np.bool_, 'Expected bool padding mask [B,N].')
    batch, length = mask.shape
    require(batch > 0 and length >= 2 and not mask[:, 0].any(), 'CLS must be valid at position zero.')
    require(np.all((~mask).sum(1) >= 2), 'Each image needs CLS and at least one region.')
    require(len(attentions) > 0, 'No attention layers supplied.')
    valid = ~mask
    pairs = valid[:, :, None] & valid[:, None, :]
    identity = np.broadcast_to(np.eye(length, dtype=np.float64), (batch, length, length)) * pairs
    joint = identity.copy()
    means = []
    for layer, alpha in enumerate(attentions):
        a = as_numpy(alpha)
        require(a.ndim == 4 and a.shape[0] == batch and a.shape[1] > 0 and
                a.shape[2:] == (length, length) and np.issubdtype(a.dtype, np.floating),
                f'Layer {layer}: expected floating attention [B,H,N,N].')
        # Also discards NaNs in padded query rows, without altering input arrays.
        clean = np.where(pairs[:, None], a, 0).astype(np.float64)
        require(np.isfinite(clean).all() and np.all(clean >= 0), 'Invalid valid-token attention.')
        require(np.allclose(clean.sum(-1), valid[:, None], atol=5e-5, rtol=0),
                'Attention rows must sum to one on valid keys; use eval mode and correct masks.')
        mean = clean.mean(axis=1)
        augmented = mean + identity
        denominator = augmented.sum(-1, keepdims=True)
        augmented = np.divide(augmented, denominator, out=np.zeros_like(augmented), where=denominator > 0)
        joint = augmented @ joint
        means.append(mean)
    require(np.isfinite(joint).all() and np.all(joint >= 0) and
            np.allclose(joint.sum(-1), valid, atol=1e-10, rtol=0), 'Rollout mass is not conserved.')
    require(np.all(joint[~pairs] == 0), 'Padding entered the rollout.')
    return {'joint': joint, 'mean_attention': np.stack(means), 'key_padding_mask': mask.copy()}


def region_relevance(cls_row, token_region_ids, label_map):
    """Match scores by region ID, allowing any order of valid token IDs.

    CLS=-1, padding=-2. Label-map IDs must be contiguous 0..R-1, as in the
    frozen SLIC implementation. No forced 256-token reshape or interpolation.
    """
    scores = as_numpy(cls_row).astype(np.float64, copy=False)
    ids = as_numpy(token_region_ids)
    labels = as_numpy(label_map)
    require(scores.ndim == 1 and ids.shape == scores.shape and
            np.issubdtype(ids.dtype, np.integer), 'Expected aligned score/ID vectors.')
    require(ids[0] == -1 and np.count_nonzero(ids == -1) == 1 and np.all(ids >= -2),
            'Expected exactly one CLS at position zero and padding ID -2.')
    require(labels.ndim == 2 and labels.size > 0 and np.issubdtype(labels.dtype, np.integer),
            'Expected integer superpixel label map.')
    region_ids = np.unique(labels)
    require(np.array_equal(region_ids, np.arange(len(region_ids))), 'Label IDs must be contiguous from zero.')
    valid_ids = ids[ids >= 0]
    require(np.array_equal(np.sort(valid_ids), region_ids), 'Token/region IDs are missing, repeated or misaligned.')
    require(np.isfinite(scores).all() and np.all(scores >= 0) and
            np.isclose(scores.sum(), 1, atol=1e-10, rtol=0) and np.all(scores[ids == -2] == 0),
            'Invalid CLS rollout mass or nonzero padding contribution.')
    raw = np.zeros(len(region_ids), dtype=np.float64)
    raw[valid_ids] = scores[ids >= 0]
    mass = float(raw.sum())
    conditional = raw / mass if mass > 0 else np.zeros_like(raw)
    peak = float(raw.max())
    display = raw / peak if peak > 0 else np.zeros_like(raw)
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=len(raw)).astype(np.int64)
    yy, xx = np.indices(labels.shape)
    geometry = np.column_stack(((np.bincount(flat, weights=yy.ravel()) / areas + .5) / labels.shape[0],
                                (np.bincount(flat, weights=xx.ravel()) / areas + .5) / labels.shape[1],
                                areas / labels.size))
    require(int(areas.sum()) == labels.size and np.all(areas > 0), 'Region coverage failed.')
    return {'region_ids': region_ids, 'region_raw': raw, 'region_conditional': conditional,
            'region_display': display, 'areas': areas, 'geometry': geometry,
            'pixel_raw': raw[labels], 'pixel_display': display[labels],
            'cls_self_mass': float(scores[0]), 'region_mass': mass, 'zero_region_flow': mass == 0}


def technical_checks():
    """Tiny independent references; no patient images or model checkpoints."""
    first = np.array([[[.6, .3, .1], [.1, .8, .1], [.2, .3, .5]],
                      [[.2, .2, .6], [.3, .3, .4], [.5, .2, .3]]], dtype=np.float64)
    second = np.array([[[.1, .7, .2], [.5, .1, .4], [.3, .6, .1]],
                       [[.5, .1, .4], [.2, .5, .3], [.1, .2, .7]]], dtype=np.float64)
    layers = [first[None], second[None]]
    original = [x.copy() for x in layers]
    mask = np.zeros((1, 3), dtype=bool)
    # Independent enumeration of intermediate token paths, not matrix multiplication.
    a = (first[0] + first[1]) / 4 + np.eye(3) / 2
    b = (second[0] + second[1]) / 4 + np.eye(3) / 2
    reference = np.array([[sum(b[i, k] * a[k, j] for k in range(3))
                           for j in range(3)] for i in range(3)])
    result = attention_rollout(layers, mask)
    np.testing.assert_allclose(result['joint'][0], reference, atol=1e-14, rtol=0)
    require(not np.allclose(reference, a @ b), 'Reference cannot detect reversed layer order.')
    one = attention_rollout(layers[:1], mask)
    np.testing.assert_allclose(one['joint'][0], a, atol=1e-14, rtol=0)
    padded = [np.pad(x, ((0, 0), (0, 0), (0, 1), (0, 1)), constant_values=np.nan) for x in layers]
    padded_result = attention_rollout(padded, np.array([[False, False, False, True]]))
    np.testing.assert_allclose(padded_result['joint'][0, :3, :3], reference, atol=1e-14, rtol=0)
    labels = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32)
    mapped = region_relevance(reference[0], np.array([-1, 1, 0]), labels)
    np.testing.assert_allclose(mapped['pixel_raw'],
                               np.where(labels == 0, reference[0, 2], reference[0, 1]), atol=0, rtol=0)
    require(np.isclose(mapped['region_conditional'].sum(), 1), 'Region normalisation failed.')
    # Uniform attention is uniform; identity-only attention has no regional flow.
    uniform = attention_rollout([np.full((1, 2, 3, 3), 1 / 3)], mask)
    np.testing.assert_allclose(uniform['joint'][0, 0], [2/3, 1/6, 1/6], atol=1e-14, rtol=0)
    identity = attention_rollout([np.eye(3)[None, None]], mask)
    empty = region_relevance(identity['joint'][0, 0], np.array([-1, 0, 1]), labels)
    require(empty['zero_region_flow'] and not empty['pixel_display'].any(), 'Zero flow fabricated a heatmap.')
    # A second image with a shorter valid sequence checks batching and CLS offsets.
    batched = []
    for x in layers:
        y = np.zeros((2, 2, 4, 4))
        y[0, :, :3, :3] = x[0]
        y[1, :, :2, :2] = np.array([[.2, .8], [.7, .3]])
        batched.append(y)
    mixed = attention_rollout(batched, np.array([[False, False, False, True], [False, False, True, True]]))
    short = attention_rollout([x[1:2, :, :2, :2] for x in batched], np.zeros((1, 2), dtype=bool))
    np.testing.assert_allclose(mixed['joint'][0, :3, :3], reference, atol=1e-14, rtol=0)
    np.testing.assert_allclose(mixed['joint'][1, :2, :2], short['joint'][0], atol=1e-14, rtol=0)
    for invalid_ids in (np.array([-1, 0, 0]), np.array([-1, 0, 2])):
        try:
            region_relevance(reference[0], invalid_ids, labels)
        except ValueError:
            pass
        else:
            raise ValueError('Invalid token IDs were accepted.')
    for old, new in zip(original, layers):
        np.testing.assert_array_equal(old, new)
    return {'independent_path_reference': True, 'head_mean_and_residual': True,
            'noncommuting_layer_order': True, 'padding_exclusion': True,
            'variable_length_batching': True, 'region_ID_mapping_and_rejection': True,
            'mass_conservation': True, 'uniform_and_zero_flow': True, 'inputs_unchanged': True}


class Progress:
    def __init__(self):
        self.stage = 'checking inputs'; self.done = self.total = self.initial = 0
        self.phase_started = time.perf_counter()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.refresh, daemon=True)

    def reset(self, stage, total=0, done=0):
        self.stage, self.total, self.done, self.initial = stage, total, done, done
        self.phase_started = time.perf_counter()
        self.show()

    def show(self):
        processed = self.done - self.initial
        eta = ev.clock((time.perf_counter() - self.phase_started) / processed * (self.total - self.done)) if processed else '--'
        if self.total and self.done == self.total:
            eta = '00:00:00'
        print(f'Attention Rollout | {self.stage} | Saved {self.done}/{self.total} | '
              f'Elapsed {ev.clock(time.perf_counter()-STARTED)} | Phase ETA {eta}', flush=True)

    def refresh(self):
        while not self.stop.wait(2):
            self.show()

    def __enter__(self):
        self.show(); self.thread.start(); return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(); self.show()


class OutputLock:
    """OS releases this lock even after forceful termination."""
    def __init__(self, output):
        self.path = output / 'attention_rollout.lock'

    def __enter__(self):
        self.stream = self.path.open('a+b')
        self.stream.seek(0, 2)
        if not self.stream.tell():
            self.stream.write(b'0'); self.stream.flush()
        self.stream.seek(0)
        try:
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise ValueError('Another process is using this attention-rollout folder.') from exc
        return self

    def __exit__(self, *args):
        self.stream.close()


def locked_context(root, variant, progress):
    """Anchor to the accepted evaluation identity; inspect only selected weights.

    Reading split/evaluation metadata does not load any validation/test image.
    Original absolute Windows paths are resolved via their recorded basenames
    and known training run, allowing the project directory itself to move.
    """
    progress.stage = f'verifying frozen {variant} checkpoint and data'
    evaluation = root / 'environment/data_audit/evaluation_v1/test_A0_A3_v1'
    paths = [evaluation / f'evaluation_{part}_v1.json' for part in ('config', 'status')]
    config, status = (ev.read_json(p) for p in paths)
    identity = config['identity']
    require(config['signature'] == tr.fingerprint(identity) == EVALUATION_SIGNATURE,
            'Evaluation identity differs from the accepted A0-A3 experiment.')
    require(status['status'] == 'COMPLETE' and status['signature'] == config['signature'],
            'The accepted classification evaluation must be complete.')
    project = d.load_project(root)
    require(identity['data_sha256'] == project['source_sha256'], 'Split/preprocessing metadata changed.')
    require(ev.runtime_versions() == identity['versions'],
            'Use the original Python/PyTorch/torchvision/PyG/NumPy environment from evaluation.')
    expected_codes = {p.replace('\\', '/').rsplit('/', 1)[-1]: h for p, h in identity['code_sha256'].items()}
    codes = {}
    for module in (*tr.MODULES, tr, ev):
        path = Path(module.__file__).resolve(); digest = d.sha256(path)
        require(expected_codes.get(path.name) == digest, f'Frozen code changed: {path.name}')
        codes[str(path)] = digest
    codes[str(Path(__file__).resolve())] = d.sha256(Path(__file__))
    run = root / 'environment/data_audit/training_v1' / ev.RUNS[variant]
    checkpoint = identity['checkpoints'][variant]
    best_epoch = checkpoint['best_epoch']
    require(best_epoch == ev.BEST_EPOCHS[variant], 'Unexpected selected epoch.')
    paths += [run / f'training_{part}_v1.json' for part in ('config', 'history', 'summary')]
    training_config, history, summary = (ev.read_json(p) for p in paths[-3:])
    for path in paths[-3:]:
        suffix = f'/{run.name}/{path.name}'
        hashes = [h for p, h in identity['input_file_sha256'].items() if p.replace('\\', '/').endswith(suffix)]
        require(len(hashes) == 1 and d.sha256(path) == hashes[0], f'Frozen training record changed: {path.name}')
    require(training_config['variant'] == summary['variant'] == variant and
            summary['status'] == 'TRAINING_COMPLETE' and summary['best_epoch'] == best_epoch,
            'Wrong/incomplete training run.')
    ev.validate_history(training_config, history, summary)
    best_path = run / f'best_epoch_{best_epoch:04d}.pt'
    require(best_path.name == summary['best_checkpoint'] and d.sha256(best_path) == checkpoint['sha256'],
            'Selected trained weights differ from the accepted evaluation checkpoint.')
    paths.append(best_path)
    files = {str(p): d.sha256(p) for p in paths}
    info = {'variant': variant, 'run': run, 'config': training_config, 'history': history,
            'summary': summary, 'best_path': best_path, 'best_sha256': checkpoint['sha256']}
    return project, info, files, codes


def choose_train_rows(project):
    rows = project['rows']['train']
    chosen = [min((r for r in rows if int(r['diagnosis']) == grade), key=lambda r: r['id_code'])
              for grade in range(5)]
    excluded = {r['id_code'] for split in ('validation', 'test') for r in project['rows'][split]}
    require(all(r['split'] == 'train' and r['id_code'] not in excluded and
                re.fullmatch(r'[0-9a-fA-F]{12}', r['id_code']) for r in chosen), 'Invalid training-only preview selection.')
    require(len({r['id_code'] for r in chosen}) == 5, 'Repeated preview ID.')
    return chosen


def explain_sample(model, sample):
    """One live deterministic image through the exact frozen neural pipeline."""
    require(not any(m.training for m in model.modules()), 'Explanation requires eval mode.')
    require(not any(p.requires_grad for p in model.parameters()), 'Preview expects a frozen model.')
    device = next(model.parameters()).device
    with torch.no_grad():
        details = tr.bm.boundary_map(sample['rgb'], return_details=True)
        segmentation = tr.sp.segment_superpixels(details['lab'], details['boundary'][0],
                         boundary_weight=2. if model.superpixel_method == 'boundary_slic' else 0.)
        labels = segmentation['labels']
        cnn = sample['cnn'].unsqueeze(0).to(device)
        output = model(cnn, [labels], segmentation_method=model.superpixel_method, return_attention=True)
        plain = model(cnn, [labels], segmentation_method=model.superpixel_method, return_attention=False)
        torch.testing.assert_close(output['logits'], plain['logits'], atol=5e-4, rtol=5e-4)
        require(torch.equal(output['logits'].argmax(-1), plain['logits'].argmax(-1)),
                'Enabling attention return changed the predicted class.')
        error = float((output['logits'] - plain['logits']).abs().max())
        attentions = output['vit_attentions']
        require(len(attentions) == 4 and all(a.shape[1] == 4 for a in attentions), 'Expected four layers/four heads.')
        rolled = attention_rollout(attentions, output['key_padding_mask'])
        mapped = region_relevance(rolled['joint'][0, 0], output['token_region_ids'][0], labels)
        require(int(output['region_counts'][0]) == len(mapped['region_raw']), 'Region counts disagree.')
        np.testing.assert_allclose(as_numpy(output['geometry'])[0, :len(mapped['areas'])],
                                   mapped['geometry'], atol=2e-6, rtol=0)
        logits = as_numpy(output['logits'][0]).copy()
        probabilities = as_numpy(output['logits'][0].double().softmax(-1)).copy()
        rgb = as_numpy(sample['rgb']).transpose(1, 2, 0)
        rgb_uint8 = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)
        evidence = {'rgb_uint8': rgb_uint8, 'labels': labels,
                    'layer_mean_attention': rolled['mean_attention'][:, 0],
                    'rollout_joint': rolled['joint'][0],
                    'token_region_ids': as_numpy(output['token_region_ids'][0]).copy(),
                    'key_padding_mask': as_numpy(output['key_padding_mask'][0]).copy(),
                    'logits': logits, 'probabilities': probabilities,
                    'label': np.asarray(sample['label'], dtype=np.int64), 'id_code': np.asarray(sample['id_code']),
                    'variant': np.asarray(model.variant), **{k: np.asarray(v) for k, v in mapped.items()}}
    record = {'id_code': sample['id_code'], 'label': int(sample['label']), 'prediction': int(np.argmax(logits)),
              'probabilities': probabilities.tolist(), 'regions': len(mapped['areas']),
              'cls_self_mass': mapped['cls_self_mass'], 'region_mass': mapped['region_mass'],
              'raw_score_min': float(mapped['region_raw'].min()), 'raw_score_max': float(mapped['region_raw'].max()),
              'zero_region_flow': mapped['zero_region_flow'], 'attention_return_max_logit_difference': error,
              'region_alignment': True, 'prediction_preserved': True}
    return evidence, record


def save_evidence(output, evidence, record):
    image_id = record['id_code']
    npz = output / f'{image_id}_rollout_v1.npz'
    temporary = npz.with_name(npz.name + '.tmp')
    try:
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **evidence)
        temporary.replace(npz)
    finally:
        temporary.unlink(missing_ok=True)
    table = output / f'{image_id}_regions_v1.csv'
    temporary = table.with_name(table.name + '.tmp')
    order = np.lexsort((evidence['region_ids'], -evidence['region_raw']))
    ranks = np.empty(len(order), dtype=int); ranks[order] = np.arange(1, len(order) + 1)
    try:
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id_code', 'region_id', 'rank_descending_raw_score', 'area_pixels', 'centroid_y',
                             'centroid_x', 'relative_area', 'rollout_raw', 'rollout_conditional', 'display_0_1'])
            for r in evidence['region_ids']:
                writer.writerow([image_id, int(r), int(ranks[r]), int(evidence['areas'][r]),
                                 *evidence['geometry'][r].tolist(), float(evidence['region_raw'][r]),
                                 float(evidence['region_conditional'][r]), float(evidence['region_display'][r])])
        temporary.replace(table)
    finally:
        temporary.unlink(missing_ok=True)
    # Never commit a cache entry whose NPZ differs from what was computed.
    with np.load(npz, allow_pickle=False) as saved:
        require(set(saved.files) == set(evidence), 'Evidence file fields changed on disk.')
        for key, value in evidence.items():
            np.testing.assert_array_equal(saved[key], value)
    return {**record, 'files': {p.name: d.sha256(p) for p in (npz, table)}, 'evidence_file': npz.name}


def load_saved_evidence(output, entry):
    for name, digest in entry['files'].items():
        require(Path(name).name == name and d.sha256(output / name) == digest, 'Saved explanation is missing or changed.')
    require(entry['evidence_file'] == entry['id_code'] + '_rollout_v1.npz' and
            entry['evidence_file'] in entry['files'], 'Unexpected evidence filename.')
    with np.load(output / entry['evidence_file'], allow_pickle=False) as values:
        evidence = {key: values[key].copy() for key in values.files}
    require(str(evidence['id_code']) == entry['id_code'] and int(evidence['label']) == entry['label'],
            'Cached evidence belongs to a different image.')
    return evidence


def save_preview(output, variant, entries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig, axes = plt.subplots(len(entries), 4, figsize=(12, 3.2 * len(entries)), squeeze=False)
    cmap = matplotlib.colormaps['magma']
    for row, entry in enumerate(entries):
        evidence = load_saved_evidence(output, entry)
        rgb = evidence['rgb_uint8'].astype(np.float64) / 255
        labels = evidence['labels']; heat = evidence['pixel_display']
        edges = np.zeros_like(labels, dtype=bool)
        edges[:-1] |= labels[:-1] != labels[1:]
        edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        outlined = rgb.copy(); outlined[edges] = [0, 1, .6]
        overlay = .55 * rgb + .45 * cmap(heat)[..., :3]
        axes[row, 0].imshow(rgb, interpolation='nearest')
        axes[row, 1].imshow(outlined, interpolation='nearest')
        axes[row, 2].imshow(heat, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        axes[row, 3].imshow(overlay, interpolation='nearest')
        axes[row, 0].set_title(f"{entry['id_code']}\nTrue grade {entry['label']} | predicted {entry['prediction']}", fontsize=9)
        axes[row, 1].set_title(f"{entry['regions']} superpixels", fontsize=10)
        axes[row, 2].set_title('ViT rollout / image maximum', fontsize=10)
        axes[row, 3].set_title('Overlay (alpha 0.45)', fontsize=10)
        for ax in axes[row]:
            ax.set_axis_off()
    fig.suptitle(f'{variant} | trained checkpoint | fixed training preview', fontsize=15, y=.985)
    fig.subplots_adjust(left=.02, right=.98, bottom=.085, top=.95, hspace=.25, wspace=.04)
    cax = fig.add_axes([.3, .052, .4, .009])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax, orientation='horizontal')
    colorbar.set_label('Relative within each image; colours are not comparable as absolute scores', fontsize=9)
    fig.text(.5, .012, 'Class-agnostic token relevance. Includes background. Not a lesion mask or a faithfulness result.',
             ha='center', fontsize=9)
    path = output / 'attention_rollout_preview_v1.jpg'
    temporary = path.with_name(path.stem + '.tmp.jpg')
    try:
        fig.savefig(temporary, dpi=140, facecolor='white')
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    return path


def run_preview(args, progress):
    root = args.root.resolve(); variant = args.variant
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA is unavailable.')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    project, info, source_files, codes = locked_context(root, variant, progress)
    chosen = choose_train_rows(project)
    selected = [{'id_code': r['id_code'], 'diagnosis': int(r['diagnosis']),
                 'image_sha256': project['processed'][r['id_code']]['output_sha256']} for r in chosen]
    checks = technical_checks()
    # Fail early if plots are unavailable; no new dependencies beyond evaluation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot
    identity = {'version': VERSION, 'variant': variant, 'checkpoint_sha256': info['best_sha256'],
                'best_epoch': info['summary']['best_epoch'], 'evaluation_signature': EVALUATION_SIGNATURE,
                'split': 'train', 'selected': selected, 'augmentation': False, 'method': METHOD,
                'data_sha256': project['source_sha256'], 'input_file_sha256': source_files,
                'code_sha256': codes, 'versions': ev.runtime_versions(), 'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    signature = tr.fingerprint(identity)
    output = root / 'environment/data_audit/attention_rollout_v1' / f'train_preview_{variant}_v1'
    require(args.resume or not output.exists(),
            f'Output exists. Continue: python attention_rollout_v1.py --variant {variant} --resume')
    require(not args.resume or (output / 'attention_rollout_config_v1.json').is_file(), 'No matching preview to resume.')
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'attention_rollout_state_v1.json'
    with OutputLock(output):
        if args.resume:
            existing = ev.read_json(output / 'attention_rollout_config_v1.json')
            require(existing['identity'] == identity and existing['signature'] == signature,
                    'Resume code/data/checkpoint/method/environment differs; existing output was not overwritten.')
            state = ev.read_json(state_path) if state_path.is_file() else {'signature': signature, 'entries': []}
        else:
            ev.write_json(output / 'attention_rollout_config_v1.json', {'identity': identity, 'signature': signature,
                          'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': SOURCES})
            state = {'signature': signature, 'entries': []}
        entries = state['entries']
        require(state['signature'] == signature and len(entries) <= 5 and
                [e['id_code'] for e in entries] == [r['id_code'] for r in chosen[:len(entries)]],
                'Saved images do not match the fixed selection order.')
        for entry, row in zip(entries, chosen):
            require(entry['label'] == int(row['diagnosis']), 'Saved label changed.')
            require(str(load_saved_evidence(output, entry)['variant']) == variant, 'Cached variant mismatch.')
        initial = len(entries); read_count = 0
        state.update(status='RUNNING', error=None)
        ev.write_json(state_path, state)
        try:
            tr.seed_runtime(42)
            model = before = None
            if initial < 5:
                progress.stage = f'loading trained {variant}, best epoch {info["summary"]["best_epoch"]}'
                model, before = ev.load_best_model(info, device)
                dataset = d.AptosDataset(project, 'train', verify_file_hash=True)
                dataset.rows = list(chosen)
                dataset.augment = None  # only this read-only explanation instance; no source file change
            progress.reset(f'{variant} training-image explanations', total=5, done=initial)
            for index in range(initial, 5):
                row = chosen[index]
                progress.stage = f'{variant} | {row["id_code"]} | grade {row["diagnosis"]}'
                sample = dataset[index]; read_count += 1
                require(sample['id_code'] == row['id_code'] and sample['split'] == 'train', 'Wrong image retrieved.')
                evidence, record = explain_sample(model, sample)
                entry = save_evidence(output, evidence, record)
                entries.append(entry)
                ev.write_json(state_path, state)  # commit AFTER both evidence files are complete
                progress.done = len(entries)
                print(f"{entry['id_code']} | {variant} | regions={entry['regions']} | true={entry['label']} | "
                      f"pred={entry['prediction']} | region alignment / prediction preserved: PASS", flush=True)
            if model is not None:
                tr.assert_tree_equal(before, tr.cpu_tree(model.state_dict()))
                require(all(p.grad is None for p in model.parameters()), 'Unexpected model gradients.')
            progress.stage = 'verifying saved evidence and building preview'
            ev.verify_sources(project, source_files, codes)
            for row in selected:
                require(d.sha256(root / 'processed_images_384_v1' / (row['id_code'] + '.png')) == row['image_sha256'],
                        'Selected training PNG changed during explanation.')
            preview = save_preview(output, variant, entries)
            summary = {'status': 'ATTENTION_ROLLOUT_PREVIEW_COMPLETE', 'version': VERSION, 'signature': signature,
                       'variant': variant, 'best_epoch': info['summary']['best_epoch'],
                       'checkpoint_sha256': info['best_sha256'], 'device': str(device),
                       'train_images': 5, 'train_PNGs_decoded_this_process': read_count,
                       'resumed_images': initial, 'validation_images_read': 0, 'test_images_read': 0,
                       'optimizer_steps': 0, 'parameters_and_BN_buffers_unchanged': True,
                       'upstream_files_unchanged': True, 'technical_checks': checks, 'results': entries,
                       'method': METHOD, 'sources': SOURCES, 'visual_review': 'AWAITING_USER_REVIEW',
                       'preview_file': preview.name, 'preview_sha256': d.sha256(preview),
                       'elapsed_seconds_this_process': time.perf_counter() - STARTED,
                       'limits': 'Five fixed training examples only; no explanation-quality or clinical-validity claim. '
                                 'GAT-refined tokens mix neighbouring evidence. Relational explanation, regional/edge '
                                 'faithfulness and lesion localisation remain separate. No test-driven method tuning.'}
            ev.write_json(output / 'attention_rollout_summary_v1.json', summary)
            state.update(status='COMPLETE', error=None)
            ev.write_json(state_path, state)
        except (Exception, KeyboardInterrupt) as exc:
            state.update(status='STOPPED', error=f'{type(exc).__name__}: {exc}')
            ev.write_json(state_path, state)
            raise
    print(f'\nATTENTION ROLLOUT CHECK: PASS | {variant} | trained best epoch {info["summary"]["best_epoch"]}')
    print('Head average / residual / layer order / padding / region mapping: PASS')
    print('Train images: 5 | Validation/test images read: 0 | Optimizer steps: 0')
    print('Prediction preservation / parameters / BN buffers / upstream files: PASS')
    print('ATTENTION PREVIEW: AWAITING VISUAL REVIEW; faithfulness not evaluated')
    print(f'Total elapsed: {ev.clock(time.perf_counter()-STARTED)}\nSaved: {output}')
    print('Return attention_rollout_summary_v1.json and attention_rollout_preview_v1.jpg.')
    print('After review passes: remember to commit attention_rollout_v1.py.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--variant', choices=ev.VARIANTS, default='A3')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--self-test', action='store_true', help='Only small mathematical references; no project/image access.')
    args = parser.parse_args()
    try:
        with Progress() as progress:
            if args.self_test:
                technical_checks()
                print('ROLLOUT NUMERICAL CHECK: PASS | Image files read: 0 | Checkpoints read: 0')
            else:
                run_preview(args, progress)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'\nATTENTION ROLLOUT STOPPED | {type(exc).__name__}: {exc}', flush=True)
        print(f'If a preview was started, continue with: python attention_rollout_v1.py --variant {args.variant} --resume', flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
