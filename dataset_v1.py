r"""CP2 input pipeline and small-batch preview. Run: python dataset_v1.py

One standalone file; requires torch, torchvision, Pillow and NumPy.
Reads the EXISTING split CSVs, RGB batch logs and 384x384 PNGs.
No FOV masks, model download, training, or new full-dataset image export.

Import for later training:
    project = load_project(Path(r'C:\CP2'))
    train = AptosDataset(project, 'train')
    for epoch in range(number_of_epochs):
        loader = make_loader(train, epoch=epoch, batch_size=4)
        for batch in loader:
            rgb = batch['rgb']       # float32 [0,1], for future boundary/SLIC
            cnn = batch['cnn']       # SAME geometry; ImageNet-normalised
            label = batch['label']   # int64 grades 0..4

Rebuild the loader with the current epoch as above. Per-image augmentation is
seeded by (seed, epoch, id), independent of model RNG consumption. This first
version deliberately uses num_workers=0; do not use concurrent threads to call
__getitem__, or change workers without revisiting epoch/RNG handling.
Reproducibility is within the recorded software/backend, not across versions.
Augmentation strengths are project choices awaiting training-image review.
"""
import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

BOOT_STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading input-pipeline libraries...', flush=True)

import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2

VERSION = 'input_pipeline_v1'
SIZE = 384
SEED = 42
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
COUNTS = {'train': 2453, 'validation': 527, 'test': 524}
CLASS_COUNTS = {'train': [1257, 236, 646, 124, 190],
                'validation': [270, 51, 138, 27, 41],
                'test': [269, 51, 138, 26, 40]}
SETTINGS = {
    'image_size': [SIZE, SIZE], 'seed': SEED,
    'training_only': {
        'order': ['square_random_resized_crop', 'rotation', 'horizontal_flip',
                  'vertical_flip', 'colour_jitter'],
        'crop_retained_area_range': [0.95, 1.0],
        'crop_aspect_ratio': [1.0, 1.0],
        'rotation_degrees': [-10.0, 10.0], 'rotation_expand': False,
        'rotation_fill_rgb': [0, 0, 0],
        'horizontal_flip_probability': 0.5, 'vertical_flip_probability': 0.5,
        'brightness_factor_range': [0.9, 1.1],
        'contrast_factor_range': [0.9, 1.1],
        'saturation_factor_range': [0.95, 1.05], 'hue_shift': 0.0,
        'interpolation': 'bilinear', 'resize_antialias': True,
        'status': 'project settings; require visual review; not proven optimal',
    },
    'rgb': 'float32 CHW in [0,1]; shared augmented image before normalisation',
    'cnn': {'mean': list(MEAN), 'std': list(STD),
            'reference_weights': 'DenseNet121_Weights.IMAGENET1K_V1',
            'default_224_crop_used': False},
    'validation_test': 'no stochastic augmentation; keep existing 384x384',
    'fov_masks': False, 'smoothing': False,
    'augmentation_seed': 'SHA256(seed:epoch:id) first 8 bytes, little endian, mod 2**63',
    'num_workers': 0, 'drop_last': False,
}
SOURCES = {
    'dataset_loader': 'https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html',
    'transforms': 'https://docs.pytorch.org/vision/0.27/transforms.html',
    'normalisation': 'https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html',
    'reproducibility': 'https://docs.pytorch.org/docs/2.12/notes/randomness.html',
    'scope': 'Official implementation references. Numerical augmentation strengths '
             'are project choices, not copied from a DR paper. CP1 section 3.4 applies.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_project(root):
    """Read small metadata files; do not decode or hash all 3,504 images."""
    root = Path(root).resolve()
    audit = root / 'environment' / 'data_audit'
    summary_path = audit / 'preprocessing_v1' / 'preprocessing_summary_v1.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    require(summary.get('status') == 'COMPLETE'
            and summary.get('ready_for_next_step') is True
            and summary.get('errors') == 0 and summary.get('review_required') == 0
            and summary.get('successful_outputs') == 3504
            and summary.get('successful_by_split') == COUNTS,
            'Expected the completed RGB preprocessing summary.')
    rows, hashes, seen_ids, seen_hashes, groups = {}, {}, set(), set(), {}
    required = {'id_code', 'diagnosis', 'split', 'status', 'split_group_id', 'pixel_sha256'}
    for split, count in COUNTS.items():
        path = audit / 'split_v1' / (split + '_v1.csv')
        items = read_csv(path)
        require(len(items) == count, f'{path.name}: expected {count} rows.')
        for r in items:
            require(required <= r.keys(), 'Missing split CSV fields.')
            require(re.fullmatch(r'[0-9a-f]{12}', r['id_code']) is not None,
                    'Invalid image ID in split CSV.')
            require(r['diagnosis'] in ('0', '1', '2', '3', '4')
                    and r['split'] == split and r['status'] == 'keep',
                    'Unexpected label, split or excluded row.')
            require(r['id_code'] not in seen_ids and r['pixel_sha256'] not in seen_hashes,
                    'Repeated image ID or recorded pixel hash across the split CSVs.')
            require(re.fullmatch(r'[0-9a-f]{64}', r['pixel_sha256']) is not None
                    and bool(r['split_group_id']), 'Missing/invalid audit hash or group.')
            seen_ids.add(r['id_code'])
            seen_hashes.add(r['pixel_sha256'])
            require(groups.setdefault(r['split_group_id'], split) == split,
                    'A split group crosses partitions.')
        counts = Counter(int(r['diagnosis']) for r in items)
        require([counts[i] for i in range(5)] == CLASS_COUNTS[split],
                f'{split}: class counts differ from fixed split v1.')
        rows[split] = sorted(items, key=lambda r: r['id_code'])
        hashes[path.relative_to(root).as_posix()] = sha256(path)
    require(len(groups) == 3494, 'Expected 3494 split groups.')
    batch_path = audit / 'preprocessing_v1' / 'preprocessing_manifest_v1.csv'
    records = read_csv(batch_path)
    processed = {r['id_code']: r for r in records}
    require(len(records) == len(processed) == 3504 and set(processed) == seen_ids,
            'RGB batch manifest does not match the fixed split IDs.')
    for split in rows:
        for r in rows[split]:
            p = processed[r['id_code']]
            require(p['status'] == 'ok' and p['split'] == split
                    and p['diagnosis'] == r['diagnosis']
                    and re.fullmatch(r'[0-9a-f]{64}', p['output_sha256']) is not None,
                    'RGB record label, split, status or output hash differs.')
    for path in (summary_path, batch_path):
        hashes[path.relative_to(root).as_posix()] = sha256(path)
    return {'root': root, 'rows': rows, 'processed': processed, 'source_sha256': hashes}


def augmentation_seed(seed, epoch, image_id):
    text = f'{seed}:{epoch}:{image_id}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little') % (2**63)


def training_transform():
    cfg = SETTINGS['training_only']
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop((SIZE, SIZE), scale=cfg['crop_retained_area_range'],
                             ratio=cfg['crop_aspect_ratio'],
                             interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.RandomRotation(cfg['rotation_degrees'], interpolation=InterpolationMode.BILINEAR,
                          expand=False, fill=0),
        v2.RandomHorizontalFlip(cfg['horizontal_flip_probability']),
        v2.RandomVerticalFlip(cfg['vertical_flip_probability']),
        v2.ColorJitter(brightness=cfg['brightness_factor_range'],
                       contrast=cfg['contrast_factor_range'],
                       saturation=cfg['saturation_factor_range'], hue=0.0),
        v2.ToDtype(torch.float32, scale=True), v2.ToPureTensor(),
    ])


class AptosDataset(Dataset):
    """Each item supplies aligned rgb/cnn tensors; augmentation is train-only."""
    def __init__(self, project, split, seed=SEED, verify_file_hash=False):
        require(split in COUNTS, 'Unknown split.')
        self.root = project['root']
        self.rows = project['rows'][split]
        self.processed = project['processed']
        self.split, self.seed, self.epoch = split, int(seed), 0
        self.verify_file_hash = verify_file_hash
        self.to_rgb = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                  v2.ToPureTensor()])
        self.augment = training_transform() if split == 'train' else None
        self.normalise = v2.Normalize(MEAN, STD, inplace=False)

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        require(isinstance(epoch, int) and epoch >= 0, 'Epoch must be a nonnegative integer.')
        self.epoch = epoch

    def read_image(self, image_id):
        path = self.root / 'processed_images_384_v1' / (image_id + '.png')
        if self.verify_file_hash:
            require(sha256(path) == self.processed[image_id]['output_sha256'],
                    f'{image_id}: PNG differs from the completed RGB batch record.')
        with Image.open(path) as source:
            source.load()
            require(source.mode == 'RGB' and source.size == (SIZE, SIZE),
                    f'{image_id}: expected RGB 384x384 PNG; got {source.mode} {source.size}.')
            return source.copy()

    def __getitem__(self, index):
        row = self.rows[index]
        image = self.read_image(row['id_code'])
        if self.augment is not None:
            # Save/restore only the CPU RNG. Do not reseed CUDA/model generators.
            # Each DataLoader worker executes serially; v1 uses num_workers=0.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(
                    augmentation_seed(self.seed, self.epoch, row['id_code']))
                rgb = self.augment(image)
        else:
            rgb = self.to_rgb(image)
        rgb = rgb.contiguous()
        cnn = self.normalise(rgb).contiguous()
        return {'rgb': rgb, 'cnn': cnn, 'label': int(row['diagnosis']),
                'id_code': row['id_code'], 'split': self.split}


def make_loader(dataset, epoch=0, batch_size=4):
    """Call once per epoch. CPU data loading; no whole-dataset GPU allocation."""
    require(isinstance(batch_size, int) and batch_size > 0, 'Invalid batch size.')
    dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(dataset.seed + epoch)
    return DataLoader(dataset, batch_size=batch_size, shuffle=dataset.split == 'train',
                      num_workers=0, drop_last=False, pin_memory=False, generator=generator)


def validate_batch(batch, expected_rows):
    n = len(batch['id_code'])
    for name in ('rgb', 'cnn'):
        value = batch[name]
        require(value.shape == (n, 3, SIZE, SIZE) and value.dtype == torch.float32
                and torch.isfinite(value).all().item(), f'Invalid {name} batch.')
    require(batch['label'].shape == (n,) and batch['label'].dtype == torch.int64,
            'Expected int64 class labels.')
    require(0 <= batch['rgb'].min().item() and batch['rgb'].max().item() <= 1,
            'RGB must be in [0,1].')
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    error = (batch['cnn'] * std + mean - batch['rgb']).abs().max().item()
    require(error < 1e-6, 'RGB/CNN branches differ beyond normalisation.')
    for image_id, label, split in zip(batch['id_code'], batch['label'].tolist(), batch['split']):
        expected = expected_rows[image_id]
        require(label == int(expected['diagnosis']) and split == expected['split'],
                'Batch ID/label/split mismatch.')
    return {'size': n, 'rgb_shape': list(batch['rgb'].shape),
            'cnn_shape': list(batch['cnn'].shape), 'labels': batch['label'].tolist(),
            'ids': batch['id_code'], 'rgb_min': float(batch['rgb'].min()),
            'rgb_max': float(batch['rgb'].max()), 'inverse_normalisation_max_error': error}


class Progress:
    def __init__(self, total):
        self.total, self.done, self.stage = total, 0, 'starting'
        self.started = BOOT_STARTED
        self.stopped = threading.Event()
        self.tty = sys.stdout.isatty()
        self.thread = threading.Thread(target=self.refresh, daemon=True)

    def display(self):
        elapsed = time.perf_counter() - self.started
        eta = f'{elapsed/self.done*(self.total-self.done):.1f}s' if self.done else '--'
        text = (f'Input checks {self.done}/{self.total} | Elapsed {elapsed:.1f}s | '
                f'ETA {eta} | {self.stage}')
        print(('\r' if self.tty else '') + text.ljust(115),
              end='' if self.tty else '\n', flush=True)

    def refresh(self):
        while not self.stopped.wait(1 if self.tty else 5):
            self.display()

    def advance(self, stage):
        self.done += 1
        self.stage = stage

    def __enter__(self):
        self.display()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join()
        self.display()
        if self.tty:
            print()


def save_preview(dataset, batch, output):
    """Show only TRAIN images, before/after the augmentation used in this batch."""
    page = Image.new('RGB', (SIZE*2, (SIZE+46)*len(batch['id_code'])), 'white')
    draw = ImageDraw.Draw(page)
    for index, image_id in enumerate(batch['id_code']):
        y = index * (SIZE+46)
        label = int(batch['label'][index])
        draw.text((8, y+6), f'TRAIN | {image_id} | grade {label}', fill='black')
        draw.text((8, y+24), 'Existing 384x384 RGB', fill='black')
        draw.text((SIZE+8, y+6), 'Augmented RGB shared by both branches', fill='black')
        draw.text((SIZE+8, y+24), 'Seed 42 | epoch 0 | before normalisation', fill='black')
        page.paste(dataset.read_image(image_id), (0, y+46))
        array = (batch['rgb'][index].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
        page.paste(Image.fromarray(array), (SIZE, y+46))
    page.save(output / 'augmentation_preview_v1.jpg', quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='CP2 project folder')
    args = parser.parse_args()
    root = args.root.resolve()
    # A new run folder avoids confusing an old PASS/preview with a failed rerun.
    run_id = datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_') + uuid4().hex[:6]
    output = root / 'environment' / 'data_audit' / VERSION / run_id
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'version': VERSION, 'errors': [],
              'unique_train_images_read': None, 'unique_validation_images_read': None,
              'test_images_read': 0, 'model_training_started': False,
              'augmentation_visually_approved': False, 'output': str(output)}
    write_json(output / 'input_pipeline_summary_v1.json', report)
    try:
        with Progress(7) as progress:
            # Limit preview overhead on machines with many logical cores. This does
            # not change the thread count when the module is imported for training.
            torch.set_num_threads(min(4, torch.get_num_threads()))
            project = load_project(root)
            cuda = torch.cuda.is_available()
            config = {'version': VERSION, 'settings': SETTINGS, 'sources': SOURCES,
                      'source_sha256': project['source_sha256'], 'script_sha256': sha256(__file__),
                      'python': platform.python_version(), 'torch': torch.__version__,
                      'torchvision': torchvision.__version__, 'numpy': np.__version__,
                      'pillow': PIL.__version__, 'cuda_available': cuda,
                      'torch_cuda_build': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(0) if cuda else None,
                      'preview_batch_sizes': {'train': 4, 'validation': 2},
                      'preview_only': True, 'test_csv_metadata_read': True,
                      'test_image_content_read': False}
            write_json(output / 'input_pipeline_config_v1.json', config)
            progress.advance('metadata ready')
            train = AptosDataset(project, 'train', verify_file_hash=True)
            validation = AptosDataset(project, 'validation', verify_file_hash=True)
            train_batch = next(iter(make_loader(train, epoch=0, batch_size=4)))
            report['unique_train_images_read'] = 4
            progress.advance('train batch loaded')
            validation_batch = next(iter(make_loader(validation, batch_size=2)))
            report['unique_validation_images_read'] = 2
            progress.advance('validation batch loaded')
            expected = {r['id_code']: r for split in project['rows'].values() for r in split}
            report['train'] = validate_batch(train_batch, expected)
            report['validation'] = validate_batch(validation_batch, expected)
            progress.advance('labels, shapes and branch alignment checked')
            # Evaluation must preserve the stored RGB pixels, including repeated reads.
            for i, image_id in enumerate(validation_batch['id_code']):
                original = np.asarray(validation.read_image(image_id)).copy()
                raw_bytes = torch.from_numpy(original).permute(2, 0, 1)
                recovered_bytes = (validation_batch['rgb'][i]*255).round().to(torch.uint8)
                require(torch.equal(raw_bytes, recovered_bytes),
                        'Evaluation unexpectedly changed stored RGB pixels.')
                repeated = validation[i]
                require(torch.equal(repeated['rgb'], validation_batch['rgb'][i]),
                        'Evaluation input is not deterministic.')
            cpu_rng_before = torch.get_rng_state().clone()
            replay = next(iter(make_loader(train, epoch=0, batch_size=4)))
            require(replay['id_code'] == train_batch['id_code']
                    and torch.equal(replay['rgb'], train_batch['rgb'])
                    and torch.equal(replay['cnn'], train_batch['cnn']),
                    'Same-seed/epoch training batch did not reproduce.')
            require(torch.equal(cpu_rng_before, torch.get_rng_state()),
                    'Input loading unexpectedly consumed the global CPU RNG.')
            report['evaluation_deterministic'] = True
            report['train_replay_identical'] = True
            report['global_cpu_rng_preserved'] = True
            progress.advance('evaluation and train replay checked')
            if cuda:
                device_batch = train_batch['cnn'].to('cuda')
                require(torch.isfinite(device_batch.square().mean()).item(),
                        'CUDA tensor operation returned a non-finite value.')
                torch.cuda.synchronize()
                require(torch.equal(device_batch.cpu(), train_batch['cnn']),
                        'CUDA transfer changed the batch.')
                del device_batch
                report['cuda_tensor_check'] = 'PASS'
            else:
                report['cuda_tensor_check'] = 'SKIPPED: CPU runtime'
            progress.advance('device check complete')
            save_preview(train, train_batch, output)
            evidence = []
            for split, batch in [('train', train_batch), ('validation', validation_batch)]:
                for image_id, label in zip(batch['id_code'], batch['label'].tolist()):
                    evidence.append({'id_code': image_id, 'diagnosis': label, 'split': split,
                                     'output_sha256': project['processed'][image_id]['output_sha256'],
                                     'augmentation_seed': augmentation_seed(SEED, 0, image_id)
                                     if split == 'train' else '', 'epoch': 0})
            with (output / 'batch_evidence_v1.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(evidence[0]))
                writer.writeheader()
                writer.writerows(evidence)
            progress.advance('training augmentation preview saved')
        report['status'] = 'TECHNICAL_PASS_AWAITING_AUGMENTATION_REVIEW'
        report['split_counts'] = COUNTS
    except Exception as exc:
        report['status'] = 'FAIL'
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['elapsed_seconds'] = round(time.perf_counter() - BOOT_STARTED, 3)
    report['limitation'] = ('Small-batch program checks and a tensor CUDA operation do not '
                            'validate full-model memory, training speed, clinical label preservation '
                            'under augmentation, or classification performance.')
    write_json(output / 'input_pipeline_summary_v1.json', report)
    if report['status'] == 'FAIL':
        print('INPUT PIPELINE CHECK: FAIL')
        for error in report['errors']:
            print(error)
    else:
        print('Split counts:', COUNTS)
        print('Train RGB / CNN shape:', report['train']['rgb_shape'], '/', report['train']['cnn_shape'])
        print('Train labels:', report['train']['labels'])
        print('Validation batch size: 2 | Test image content read: 0')
        print('Aligned branches: PASS | Evaluation deterministic: PASS | Train replay: PASS')
        print('CUDA tensor check:', report['cuda_tensor_check'])
        print('INPUT PIPELINE CHECK: PASS (small-batch technical checks only)')
        print('AUGMENTATION STATUS: AWAITING VISUAL REVIEW')
    print('Total elapsed:', f"{report['elapsed_seconds']:.1f}s")
    print('Saved:', output)
    print('Existing RGB images and split CSVs were read only.')
    print('Return this error summary.' if report['status'] == 'FAIL' else
          'Return this summary and augmentation_preview_v1.jpg.')
    return 1 if report['status'] == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())
