r"""CP2: fixed A0-A3 checkpoints on the held-out APTOS test partition.

    python evaluate_v1.py --mode test
    python evaluate_v1.py --mode test --resume

Default --mode check validates metadata and checkpoints without reading images.
Test mode includes those checks, then evaluates all 524 test IDs per variant.
No training, tuning, new model selection, augmentation or image export occurs.
Completed batches are atomically saved; --resume uses the SAME fixed output
folder and skips committed predictions. Existing training files stay read-only.
Requires the existing CP2 libraries plus matplotlib for confusion-matrix plots.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import threading
import time

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading evaluation libraries...', flush=True)
import numpy as np
import torch
import torchvision
import torch_geometric
import train_v1 as tr

d = tr.d
classifier = tr.classifier
require = tr.require
VERSION = 'evaluation_v1'
VARIANTS = ('A0', 'A1', 'A2', 'A3')
RUNS = {
    'A0': 'run_20261014_072633_8b1a24',
    'A1': 'run_20261016_082703_acb28d',
    'A2': 'run_20261017_042401_6258a0',
    'A3': 'run_20261018_055602_7e9de2',
}
BEST_EPOCHS = {'A0': 23, 'A1': 21, 'A2': 22, 'A3': 21}
METRICS = ('qwk', 'macro_f1', 'accuracy', 'macro_precision', 'macro_recall')
SOURCES = {
    'proposal': 'CP1 section 3.10.1, printed page 48; computational reporting scope in 3.10.4.',
    'checkpoint_eval': 'https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html',
    'qwk': 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html',
    'precision_recall_f1': 'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html',
    'scope': 'Definitions implemented directly; sklearn is not a runtime dependency. '
             'Fixed labels 0..4, quadratic kappa, unweighted macro averages, zero_division=0.',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def clock(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


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
        eta = (clock((time.perf_counter() - self.phase_started) / processed * (self.total - self.done))
               if processed > 0 else ('00:00:00' if self.total and self.done == self.total else '--'))
        print(f'Evaluation | {self.stage} | Saved {self.done}/{self.total} | '
              f'Elapsed {clock(time.perf_counter() - STARTED)} | Phase ETA {eta}', flush=True)

    def refresh(self):
        while not self.stop.wait(5):
            self.show()

    def __enter__(self):
        self.show(); self.thread.start(); return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(); self.show()


class RunLock:
    """OS-owned lock: automatically released even after process termination."""
    def __init__(self, output):
        self.path = output / 'evaluation.lock'
        self.record_errors = False

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
            raise ValueError('Another evaluation is using this output folder.') from exc
        return self

    def __exit__(self, *args):
        try:
            if self.record_errors and args[1] is not None:
                write_json(self.path.parent / 'evaluation_status_v1.json', {
                    'status': 'STOPPED', 'error': f'{type(args[1]).__name__}: {args[1]}',
                    'note': 'Committed batches remain saved. Only an unfinished batch may need repeating.'})
        finally:
            self.stream.close()


def runtime_versions():
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'pyg': str(torch_geometric.__version__),
            'numpy': np.__version__}


def metrics_from_confusion(cm):
    cm = np.asarray(cm)
    require(cm.shape == (5, 5) and np.isfinite(cm).all() and (cm >= 0).all()
            and np.equal(cm, np.floor(cm)).all() and cm.sum() > 0, 'Invalid five-class counts.')
    result = tr.metrics_from_confusion(cm)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(np.diag(cm), predicted, out=np.zeros(5), where=predicted > 0)
    recall = np.divide(np.diag(cm), support, out=np.zeros(5), where=support > 0)
    return {**result, 'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
            'per_class_precision': precision.tolist(), 'per_class_recall': recall.tolist(),
            'support': support.astype(int).tolist()}


def validate_history(config, history, summary):
    cfg = config['settings']; best = None; best_epoch = None; anchor = None; bad = 0
    require(len(history) == summary['epochs_completed'] and 0 < len(history) <= cfg['epochs'],
            'Training history length differs from summary.')
    steps = len(tr.index_groups(d.COUNTS['train'], 4, cfg['seed'], 0))
    for epoch, row in enumerate(history, 1):
        require(row['epoch'] == epoch and row['optimizer_steps'] == steps * epoch,
                'Missing/repeated epoch or inconsistent optimizer step count.')
        val = row['validation']; recomputed = metrics_from_confusion(val['confusion_matrix'])
        require(recomputed['support'] == d.CLASS_COUNTS['validation'] and val['samples'] == 527,
                'Validation sample counts changed.')
        for key in ('qwk', 'macro_f1', 'accuracy', 'per_class_f1'):
            require(np.allclose(recomputed[key], val[key], rtol=0, atol=1e-12),
                    f'Validation {key} differs from confusion counts.')
        require(math.isfinite(row['train_weighted_loss']) and math.isfinite(val['weighted_loss']),
                'Nonfinite training/validation loss.')
        if tr.is_better(val, best, cfg):
            best, best_epoch = val, epoch
        if anchor is None or val['qwk'] > anchor + cfg['min_qwk_delta']:
            anchor, bad = val['qwk'], 0
        else:
            bad += 1
        require(row['bad_epochs'] == bad and (bad < cfg['patience'] or epoch == len(history)),
                'Early-stopping history is inconsistent.')
    require((len(history) == cfg['epochs'] or bad >= cfg['patience']) and
            summary['stopped_early'] == (bad >= cfg['patience']), 'Training is not complete.')
    require(summary['optimizer_steps'] == history[-1]['optimizer_steps'] and
            summary['best_epoch'] == best_epoch and summary['best_validation_metrics'] == best,
            'Best validation checkpoint does not follow the frozen selection rule.')


def verify_payload(payload, info, epoch):
    signature = tr.fingerprint({'core': info['config']['core_fingerprint'], 'variant': info['variant']})
    require(payload.get('kind') == 'train' and payload.get('version') == tr.VERSION and
            payload.get('variant') == info['variant'] and payload.get('signature') == signature and
            payload.get('epoch_completed') == epoch and
            payload.get('global_step') == info['history'][epoch - 1]['optimizer_steps'],
            'Checkpoint is not the matching formal training checkpoint.')
    state = payload['selection_state']; summary = info['summary']
    require(state['history'] == info['history'][:epoch] and state['best_epoch'] == summary['best_epoch'] and
            state['best'] == summary['best_validation_metrics'] and
            state['best_checkpoint_file'] == summary['best_checkpoint'],
            'Checkpoint selection state differs from the recorded training run.')


def load_best_model(info, device):
    require(d.sha256(info['best_path']) == info['best_sha256'], 'Best checkpoint changed.')
    payload = torch.load(info['best_path'], map_location='cpu', weights_only=True)
    verify_payload(payload, info, info['summary']['best_epoch'])
    state = payload['model']
    model = classifier.RegionalDRModel(info['variant'], pretrained=False,
                                       seed=info['config']['settings']['seed'])
    require(all(not value.is_floating_point() or bool(torch.isfinite(value).all())
                for value in state.values()), 'Nonfinite checkpoint tensor.')
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, state


def preflight(root, runs, progress):
    """All four runs and weights checked BEFORE any test image is opened."""
    project = d.load_project(root)
    weights, counts = classifier.training_class_weights(project['rows']['train'])
    codes = {Path(m.__file__).name: d.sha256(Path(m.__file__)) for m in (*tr.MODULES, tr)}
    versions = runtime_versions(); infos = {}; file_hashes = {}
    for variant in VARIANTS:
        progress.reset(f'checking {variant} metadata and checkpoint')
        run = runs[variant].resolve()
        require(run.parent == (root / 'environment/data_audit/training_v1').resolve(),
                'Training run must belong to this project training_v1 folder.')
        files = {name: run / f'training_{name}_v1.json' for name in ('config', 'history', 'summary')}
        config, history, summary = (read_json(files[name]) for name in ('config', 'history', 'summary'))
        require(config.get('mode') == summary.get('mode') == 'train' and
                config.get('variant') == summary.get('variant') == variant and
                config.get('version') == summary.get('version') == tr.VERSION and
                summary.get('status') == 'TRAINING_COMPLETE', f'{variant}: incomplete/wrong training run.')
        require(summary['train_images'] == 2453 and summary['validation_images'] == 527 and
                summary['test_images_read'] == 0, 'Unexpected training split usage.')
        require(config['settings'] == config['core']['settings'] == tr.SETTINGS,
                'Training settings differ from the locked v1 experiment.')
        require(tr.fingerprint(config['core']) == config['core_fingerprint'], 'Training fingerprint mismatch.')
        require(config['core']['code_sha256'] == codes and config['core']['data_sha256'] == project['source_sha256'],
                'Model code or split/preprocessing metadata changed since training.')
        require(config['core']['versions'] == versions,
                'Python/PyTorch/torchvision/PyG/NumPy versions changed; use the original training environment.')
        require(config['train_class_counts'] == counts.tolist() and
                np.allclose(config['train_class_weights'], weights.tolist(), atol=0, rtol=0),
                'Training class weights differ.')
        validate_history(config, history, summary)
        require(summary['best_epoch'] == BEST_EPOCHS[variant], 'Selected epoch differs from the accepted experiment.')
        best_name = f'best_epoch_{summary["best_epoch"]:04d}.pt'
        require(summary['best_checkpoint'] == best_name and summary['last_checkpoint'] == 'last.pt',
                'Unexpected checkpoint filename.')
        info = {'variant': variant, 'run': run, 'config': config, 'history': history, 'summary': summary,
                'best_path': run / best_name, 'best_sha256': d.sha256(run / best_name)}
        last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
        verify_payload(last, info, summary['epochs_completed'])
        require(last['selection_state']['best_checkpoint_sha256'] == info['best_sha256'],
                'Best checkpoint hash differs from last.pt selection record.')
        del last
        model, state = load_best_model(info, torch.device('cpu'))
        info['parameter_count'] = sum(p.numel() for p in model.parameters())
        info['parameter_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
        info['state_bytes'] = sum(t.numel() * t.element_size() for t in state.values())
        del model, state
        for path in [*files.values(), run / 'last.pt', info['best_path']]:
            file_hashes[str(path)] = d.sha256(path)
        infos[variant] = info
        print(f'{variant} | best epoch={summary["best_epoch"]} | strict model load/hash/selection=PASS', flush=True)
    require(len({info['config']['core_fingerprint'] for info in infos.values()}) == 1,
            'A0-A3 settings/data/code/training environments differ.')
    return project, weights.numpy().astype(np.float64), infos, file_hashes


def score_rows(rows, weights):
    require(bool(rows), 'No predictions to score.')
    logits = np.asarray([row['logits'] for row in rows], dtype=np.float64)
    labels = np.asarray([row['label'] for row in rows], dtype=np.int64)
    require(logits.shape == (len(rows), 5) and np.isfinite(logits).all(), 'Invalid saved logits.')
    require(((labels >= 0) & (labels < 5)).all(), 'Invalid saved labels.')
    shifted = logits - logits.max(1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    probabilities = np.exp(logp); predictions = logits.argmax(1)
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    metrics = metrics_from_confusion(cm)
    nll = -logp[np.arange(len(rows)), labels]
    metrics['weighted_loss'] = float(np.dot(weights[labels], nll) / weights[labels].sum())
    return metrics, probabilities, predictions


def save_cache(path, cache):
    body = {k: v for k, v in cache.items() if k != 'payload_sha256'}
    write_json(path, {**body, 'payload_sha256': tr.fingerprint(body)})


def load_cache(path, variant, signature, project, batch_size):
    if not path.exists():
        return {'version': VERSION, 'variant': variant, 'signature': signature, 'rows': [],
                'preparation_seconds': 0., 'forward_seconds': 0.,
                'peak_cuda_allocated_mib': 0., 'peak_cuda_reserved_mib': 0.}
    cache = read_json(path); digest = cache.pop('payload_sha256')
    require(digest == tr.fingerprint(cache) and cache['version'] == VERSION and
            cache['variant'] == variant and cache['signature'] == signature, 'Saved prediction cache changed/mismatched.')
    rows = cache['rows']; expected = project['rows']['test']
    require(len(rows) <= len(expected) and (len(rows) == len(expected) or len(rows) % batch_size == 0),
            'Cache does not end at a complete evaluation batch.')
    for row, ref in zip(rows, expected):
        require(row['id_code'] == ref['id_code'] and row['label'] == int(ref['diagnosis']) and
                row['split_group_id'] == ref['split_group_id'] and
                row['output_sha256'] == project['processed'][ref['id_code']]['output_sha256'],
                'Saved prediction ID/label/hash differs from the fixed test manifest.')
        values = np.asarray(row['logits'])
        require(values.shape == (5,) and np.isfinite(values).all(), 'Invalid cached logits.')
    for key in ('preparation_seconds', 'forward_seconds', 'peak_cuda_allocated_mib', 'peak_cuda_reserved_mib'):
        require(math.isfinite(cache[key]) and cache[key] >= 0, 'Invalid cached timing/memory.')
    return cache


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def evaluate_variant(info, project, weights, device, batch_size, cache, cache_path, progress):
    dataset = d.AptosDataset(project, 'test', verify_file_hash=True)
    require(dataset.augment is None and dataset.rows == project['rows']['test'], 'Test dataset changed/augmented.')
    total = len(dataset); initial = len(cache['rows'])
    progress.reset(f'TEST {info["variant"]}', total, initial)
    if initial < total:
        tr.seed_runtime(info['config']['settings']['seed'])
        model, original_state = load_best_model(info, device)
        require(not any(module.training for module in model.modules()), 'Model is not fully in eval mode.')
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for start in range(initial, total, batch_size):
                indices = list(range(start, min(start + batch_size, total)))
                sync(device); begun = time.perf_counter()
                batch = tr.prepare_microbatch(dataset, indices, model.superpixel_method, device)
                sync(device); prepared = time.perf_counter()
                reference = dataset.rows[start:start + len(indices)]
                require(batch['ids'] == [r['id_code'] for r in reference] and
                        batch['labels'].tolist() == [int(r['diagnosis']) for r in reference],
                        'Test batch ID/label order differs from manifest.')
                result = model(batch['cnn'], batch['maps'], segmentation_method=model.superpixel_method)
                sync(device); finished = time.perf_counter()
                logits = result['logits'].detach().cpu().numpy()
                require(logits.shape == (len(indices), 5) and np.isfinite(logits).all(), 'Nonfinite/invalid test logits.')
                rows = [{'id_code': r['id_code'], 'label': int(r['diagnosis']),
                         'split_group_id': r['split_group_id'],
                         'output_sha256': project['processed'][r['id_code']]['output_sha256'],
                         'logits': values.tolist()} for r, values in zip(reference, logits)]
                # Save only predictions produced with unchanged weights/BN buffers.
                # This also covers termination before the end of a variant.
                tr.assert_tree_equal(model.state_dict(), original_state)
                cache['rows'].extend(rows)
                cache['preparation_seconds'] += prepared - begun
                cache['forward_seconds'] += finished - prepared
                if device.type == 'cuda':
                    cache['peak_cuda_allocated_mib'] = max(cache['peak_cuda_allocated_mib'],
                        torch.cuda.max_memory_allocated(device) / 2**20)
                    cache['peak_cuda_reserved_mib'] = max(cache['peak_cuda_reserved_mib'],
                        torch.cuda.max_memory_reserved(device) / 2**20)
                save_cache(cache_path, cache)  # Commit before reporting progress.
                progress.done = len(cache['rows'])
                del result, batch, logits
        tr.assert_tree_equal(model.state_dict(), original_state)
        require(all(p.grad is None for p in model.parameters()), 'Unexpected evaluation gradients.')
        del model, original_state
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    metrics, probabilities, predictions = score_rows(cache['rows'], weights)
    require(metrics['samples'] == total == 524 and metrics['support'] == d.CLASS_COUNTS['test'],
            'Incomplete test coverage/class counts.')
    require(metrics['qwk'] is not None, 'Undefined test kappa.')
    timing = {key: cache[key] for key in ('preparation_seconds', 'forward_seconds',
                                         'peak_cuda_allocated_mib', 'peak_cuda_reserved_mib')}
    timing.update(preparation_seconds_per_image=cache['preparation_seconds'] / total,
                  forward_seconds_per_image=cache['forward_seconds'] / total)
    return {'variant': info['variant'], 'best_epoch': info['summary']['best_epoch'],
            'checkpoint': str(info['best_path']), 'checkpoint_sha256': info['best_sha256'],
            'test_metrics': metrics, 'parameter_count': info['parameter_count'],
            'parameter_bytes': info['parameter_bytes'], 'model_state_bytes': info['state_bytes'],
            'checkpoint_bytes_including_optimizer': info['best_path'].stat().st_size,
            'timing': timing, 'resumed_predictions': initial,
            'new_predictions_this_process': total - initial}, probabilities, predictions


def write_csv(path, fields, rows):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def save_variant_tables(output, variant, cache, result, probabilities, predictions):
    rows = []
    for row, probability, prediction in zip(cache['rows'], probabilities, predictions):
        rows.append({'id_code': row['id_code'], 'split_group_id': row['split_group_id'],
                     'true_label': row['label'], 'predicted_label': int(prediction),
                     'output_sha256': row['output_sha256'],
                     **{f'probability_{c}': float(probability[c]) for c in range(5)},
                     **{f'logit_{c}': row['logits'][c] for c in range(5)}})
    write_csv(output / f'{variant}_predictions_v1.csv', list(rows[0]), rows)
    cm = result['test_metrics']['confusion_matrix']
    rows = [{'true_grade': c, **{f'predicted_{j}': cm[c][j] for j in range(5)}} for c in range(5)]
    write_csv(output / f'{variant}_confusion_v1.csv', list(rows[0]), rows)
    metrics = result['test_metrics']
    rows = [{'grade': c, 'support': metrics['support'][c], 'precision': metrics['per_class_precision'][c],
             'recall': metrics['per_class_recall'][c], 'f1': metrics['per_class_f1'][c]} for c in range(5)]
    write_csv(output / f'{variant}_per_class_v1.csv', list(rows[0]), rows)


def save_comparison(output, results):
    rows = [{'variant': v, 'best_epoch': results[v]['best_epoch'],
             **{k: results[v]['test_metrics'][k] for k in (*METRICS, 'weighted_loss')},
             'test_samples': results[v]['test_metrics']['samples']} for v in VARIANTS]
    write_csv(output / 'test_comparison_v1.csv', list(rows[0]), rows)
    contrasts = [('A1', 'A0', 'Boundary effect without GAT'), ('A2', 'A0', 'GAT effect with SLIC'),
                 ('A3', 'A2', 'Boundary effect with GAT'), ('A3', 'A1', 'GAT effect with boundary SLIC')]
    differences = [{'contrast': f'{a}-{b}', 'interpretation': meaning,
                    **{f'delta_{k}': results[a]['test_metrics'][k] - results[b]['test_metrics'][k]
                       for k in METRICS}} for a, b, meaning in contrasts]
    write_csv(output / 'ablation_differences_v1.csv', list(differences[0]), differences)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for normalized in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout='constrained')
        for ax, variant in zip(axes.flat, VARIANTS):
            cm = np.asarray(results[variant]['test_metrics']['confusion_matrix'])
            values = cm / cm.sum(1, keepdims=True) if normalized else cm
            image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else max(d.CLASS_COUNTS['test']))
            for i in range(5):
                for j in range(5):
                    label = f'{values[i,j]:.1%}' if normalized else str(cm[i,j])
                    ax.text(j, i, label, ha='center', va='center',
                            color='white' if values[i,j] > image.norm.vmax * .5 else '#18202A')
            ax.set(xticks=range(5), yticks=range(5), xlabel='Predicted grade', ylabel='True grade',
                   title=f'{variant} | validation-selected epoch {results[variant]["best_epoch"]}')
        fig.colorbar(image, ax=axes, shrink=.8, label='Fraction within true grade' if normalized else 'Images')
        fig.suptitle('APTOS held-out test | 524 images per model | ' + ('Row normalized' if normalized else 'Counts'))
        stem = 'test_confusion_normalized_v1' if normalized else 'test_confusion_counts_v1'
        fig.savefig(output / f'{stem}.png', dpi=180)
        plt.close(fig)


def verify_sources(project, file_hashes, codes):
    for name, digest in project['source_sha256'].items():
        require(d.sha256(project['root'] / name) == digest, 'Split/preprocessing metadata changed during evaluation.')
    for path, digest in file_hashes.items():
        require(d.sha256(Path(path)) == digest, 'Training evidence/checkpoint changed during evaluation.')
    for path, digest in codes.items():
        require(d.sha256(Path(path)) == digest, 'Code changed during evaluation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--mode', choices=('check', 'test'), default='check')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    for v in VARIANTS:
        parser.add_argument(f'--{v.lower()}-run', type=Path)
    args = parser.parse_args(); output = None; active = False
    try:
        require(not args.resume or args.mode == 'test', '--resume is for test evaluation only.')
        require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA is unavailable.')
        device = torch.device('cuda' if args.device == 'cuda' or
                              (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        root = args.root.resolve(); runs = {}
        for v in VARIANTS:
            selected = getattr(args, f'{v.lower()}_run')
            runs[v] = (selected if selected.is_absolute() else root / selected) if selected is not None else (
                root / 'environment/data_audit/training_v1' / RUNS[v])
        with Progress() as progress:
            project, weights, infos, file_hashes = preflight(root, runs, progress)
            if args.mode == 'check':
                print('\nEVALUATION INPUT CHECK: PASS | Image files read: 0')
                print('Ready: python evaluate_v1.py --mode test')
                return 0
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot  # Fail before opening test images if plotting is unavailable.
            except ImportError as exc:
                raise ValueError('Install plots first: python -m pip install matplotlib') from exc
            batch_size = 2
            code_hashes = {str(Path(m.__file__).resolve()): d.sha256(Path(m.__file__))
                           for m in (*tr.MODULES, tr)}
            code_hashes[str(Path(__file__).resolve())] = d.sha256(Path(__file__))
            identity = {'version': VERSION, 'split': 'test', 'class_order': list(range(5)),
                'batch_size': batch_size, 'seed': 42, 'precision': 'float32 neural inference; float64 saved-logit scoring',
                'prediction': 'argmax of five logits; no thresholds/TTA/calibration',
                'test_ids': [r['id_code'] for r in project['rows']['test']],
                'data_sha256': project['source_sha256'], 'input_file_sha256': file_hashes,
                'code_sha256': code_hashes, 'versions': runtime_versions(),
                'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'checkpoints': {v: {'path': str(infos[v]['best_path']), 'sha256': infos[v]['best_sha256'],
                                    'best_epoch': infos[v]['summary']['best_epoch']} for v in VARIANTS}}
            signature = tr.fingerprint(identity)
            output = root / 'environment/data_audit/evaluation_v1/test_A0_A3_v1'
            if args.resume:
                require((output / 'evaluation_config_v1.json').is_file(), 'No evaluation to resume.')
            else:
                require(not output.exists(), f'Output already exists. Continue with: python evaluate_v1.py --mode test --resume')
            output.mkdir(parents=True, exist_ok=True)
            with RunLock(output) as run_lock:
                if args.resume:
                    existing = read_json(output / 'evaluation_config_v1.json')
                    require(existing['identity'] == identity and existing['signature'] == signature,
                            'Resume data/checkpoints/code/environment/device differs from the original evaluation.')
                else:
                    write_json(output / 'evaluation_config_v1.json', {'identity': identity, 'signature': signature,
                        'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': SOURCES,
                        'selection': 'All four checkpoints selected using validation only before any test image read.',
                        'resume': 'Atomic saved-logit cache after every completed batch; same fixed output directory.'})
                active = True
                run_lock.record_errors = True
                caches = {v: load_cache(output / f'{v}_cache_v1.json', v, signature, project, batch_size)
                          for v in VARIANTS}
                write_json(output / 'evaluation_status_v1.json', {'status': 'RUNNING', 'signature': signature})
                results = {}
                for v in VARIANTS:
                    results[v], probabilities, predictions = evaluate_variant(infos[v], project, weights, device,
                        batch_size, caches[v], output / f'{v}_cache_v1.json', progress)
                    save_variant_tables(output, v, caches[v], results[v], probabilities, predictions)
                    m = results[v]['test_metrics']
                    print(f'{v} | 524/524 | test QWK={m["qwk"]:.5f} | macro-F1={m["macro_f1"]:.5f} | '
                          f'accuracy={m["accuracy"]:.5f}', flush=True)
                progress.reset('saving comparison and confusion matrices')
                verify_sources(project, file_hashes, code_hashes)
                save_comparison(output, results)
                summary = {'status': 'TEST_EVALUATION_COMPLETE', 'version': VERSION, 'split': 'test',
                    'signature': signature, 'class_order': list(range(5)), 'results': results,
                    'unique_test_images': 524, 'predictions_total': 2096,
                    'new_predictions_this_process': sum(r['new_predictions_this_process'] for r in results.values()),
                    'train_images_read': 0, 'validation_images_read': 0, 'optimizer_steps': 0,
                    'parameters_and_BN_buffers_unchanged': True, 'upstream_files_unchanged': True,
                    'sources': SOURCES, 'elapsed_seconds_this_process': time.perf_counter() - STARTED,
                    'timing_scope': 'Committed batches only; preparation includes PNG hash/read, transforms, CPU '
                        'boundary/SLIC and transfer. Forward includes CNN/pooling/graph/ViT/head and CUDA synchronization. '
                        'Per-image figures are batch throughput averages, not isolated latency; first batch included. '
                        'Interrupted uncommitted work and plotting/checkpoint checks are excluded from batch timings. '
                        'CUDA peaks are maxima of sampled processes, include model tensors, and are not total GPU usage.',
                    'limitations': 'Single seed and fixed internal split; descriptive component comparisons, no '
                        'statistical significance claim. Test is not used to change checkpoints/hyperparameters. '
                        'External IDRiD, explanation, localisation and controlled timing benchmarks remain separate. '
                        'CUDA graph scatter may preclude bitwise reproducibility.'}
                write_json(output / 'evaluation_summary_v1.json', summary)
                write_json(output / 'evaluation_status_v1.json', {'status': 'COMPLETE', 'signature': signature})
                active = False
        print('\nTEST EVALUATION COMPLETE | A0-A3 | 524 unique test images | 2096 predictions')
        print('Train/validation images read: 0 | Optimizer steps: 0 | Training checkpoints unchanged: PASS')
        print(f'Total elapsed this process: {clock(time.perf_counter() - STARTED)}\nSaved: {output}')
        print('Return evaluation_summary_v1.json and test_comparison_v1.csv.')
        print('After review passes: remember to commit evaluate_v1.py.')
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        message = f'{type(exc).__name__}: {exc}'
        print(f'\nEVALUATION STOPPED | {message}', flush=True)
        if active and output is not None:
            print('Continue: python evaluate_v1.py --mode test --resume', flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
