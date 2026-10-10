r"""CP2 section 3.5.1: continuous boundary-strength map and TRAIN preview.

Put beside dataset_v1.py, then run: python boundary_map_v1.py
Uses the existing torch/torchvision/Pillow/NumPy environment; no new packages.
Does not train, segment superpixels, generate FOV masks, or alter source images.

Later integration (regenerate AFTER each epoch's augmentation):
    from boundary_map_v1 import boundary_map
    boundary = boundary_map(batch['rgb'].to(device))  # [B,1,H,W], float32
    features = cnn_model(batch['cnn'].to(device))     # separate unchanged input

Input: unstandardised sRGB float32 in [0,1], [B,3,H,W] (or [3,H,W]).
Output: continuous strength in [0,1]; SAME geometry/device, no autograd.
This is an image-gradient prior, NOT lesion segmentation or a probability.
Saved preview NPYs are evidence for epoch 0, not a training cache.
"""
import argparse
import csv
import json
import platform
import sys
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading boundary-map libraries...', flush=True)

import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import gaussian_blur

VERSION = 'boundary_map_v1'
SIGMA = 1.0
KERNEL_SIZE = 7
FLAT_EPS = 1e-4
DATASET_SHA256 = '0407e4f4f8c19cd36175a8e760cc3722ed21f5dff2e398db6a1a1cc997f05bd6'
SETTINGS = {
    'input': 'unstandardised sRGB float32 [0,1], after shared train augmentation',
    'colour_space': 'CIELAB, D65 white (0.95047,1,1.08883), 2-degree observer',
    'rgb_to_xyz_matrix': [[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]],
    'lab_units': 'native L*,a*,b*; no individual channel rescaling/normalisation',
    'gaussian': {'sigma_pixels': SIGMA, 'kernel_size': KERNEL_SIZE,
                 'support_radius_pixels': 3, 'channels_filtered_separately': True},
    'padding': 'PyTorch reflect (excludes the edge pixel; SciPy equivalent: mirror)',
    'scharr_x_cross_correlation_kernel': [[-3/16, 0, 3/16],
                                         [-10/16, 0, 10/16], [-3/16, 0, 3/16]],
    'scharr_y': 'transpose of x kernel',
    'combined_strength': 'sqrt(sum_c(gx_c**2 + gy_c**2)), c=L*,a*,b*',
    'normalisation': 'per image: raw / raw.max; all zeros if max <= flat_epsilon',
    'flat_epsilon_lab_gradient_units': FLAT_EPS,
    'batch_statistics_used': False, 'label_or_lesion_annotations_used': False,
    'fov_mask': False, 'binary_threshold_or_nonmax_suppression': False,
    'cnn_input_smoothing': False,
    'display_only': 'third preview column sqrt(boundary); NPY/linear column unchanged',
    'status': 'initial project settings; not performance-optimised or validated as lesions',
}
SOURCES = {
    'proposal': 'CP1 sections 3.5.1 and 3.5.2',
    'contour_guidance': 'https://doi.org/10.1016/j.cviu.2018.01.006',
    'contour_guidance_author_manuscript': 'https://arxiv.org/abs/1903.07193',
    'scharr_thesis': 'https://hci.iwr.uni-heidelberg.de/node/5556',
    'lab_convention': 'https://scikit-image.org/docs/0.25.x/api/skimage.color.html#skimage.color.rgb2lab',
    'scharr_operator': 'https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.scharr',
    'gaussian_implementation': 'https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.gaussian_blur.html',
    'padding': 'https://docs.pytorch.org/docs/2.12/generated/torch.nn.ReflectionPad2d.html',
    'scope': 'Giraud et al. supports contour-guided region formation, not this exact '
             'Gaussian/Scharr recipe. Sigma, support, Lab gradient combination and '
             'max normalisation are explicit project implementation choices. '
             'No claim of reproducing SCALP or optimal APTOS settings.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _as_batch(rgb):
    require(isinstance(rgb, torch.Tensor), 'Expected a torch RGB tensor.')
    single = rgb.ndim == 3
    x = rgb.unsqueeze(0) if single else rgb
    require(x.ndim == 4 and x.shape[0] > 0 and x.shape[1] == 3
            and min(x.shape[-2:]) >= 4, 'Expected [B,3,H,W] or [3,H,W], H,W >= 4.')
    require(x.device.type in ('cpu', 'cuda') and x.dtype == torch.float32,
            'Expected float32 on CPU/CUDA; supply batch["rgb"], not the CNN tensor.')
    require(torch.isfinite(x).all().item() and x.min().item() >= 0
            and x.max().item() <= 1, 'Expected finite RGB values in [0,1].')
    return x, single


def _rgb_to_lab(x):
    # sRGB transfer function -> XYZ D65 -> native CIELAB units.
    linear = torch.where(x > 0.04045, ((x + 0.055) / 1.055).pow(2.4), x / 12.92)
    r, g, b = linear.unbind(dim=1)
    xyz = torch.stack((0.412453*r + 0.357580*g + 0.180423*b,
                       0.212671*r + 0.715160*g + 0.072169*b,
                       0.019334*r + 0.119193*g + 0.950227*b), dim=1)
    white = x.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    t = xyz / white
    # The rounded conventional breakpoints match the scikit-image convention.
    f = torch.where(t > 0.008856, t.clamp_min(0).pow(1/3), 7.787*t + 16/116)
    fx, fy, fz = f.unbind(dim=1)
    return torch.stack((116*fy - 16, 500*(fx-fy), 200*(fy-fz)), dim=1)


def _scharr_xy(lab):
    p = F.pad(lab, (1, 1, 1, 1), mode='reflect')
    # Difference first: exactly zero for spatially constant channel values.
    dx = p[:, :, :, 2:] - p[:, :, :, :-2]
    dy = p[:, :, 2:, :] - p[:, :, :-2, :]
    gx = (3*dx[:, :, :-2, :] + 10*dx[:, :, 1:-1, :] + 3*dx[:, :, 2:, :]) / 16
    gy = (3*dy[:, :, :, :-2] + 10*dy[:, :, :, 1:-1] + 3*dy[:, :, :, 2:]) / 16
    return gx, gy


@torch.no_grad()
def boundary_map(rgb, *, return_details=False):
    """Return [B,1,H,W] or [1,H,W]; never resize, augment or change input RGB.

    Fixed float32 math even if called inside an outer mixed-precision context.
    A whole-image max <=1e-4 returns zeros to avoid amplifying numerical noise.
    Max normalisation is per image, NOT per batch/channel. It does not make
    strengths physically comparable across images; bright outer edges can dominate.
    """
    x, single = _as_batch(rgb)
    backend = (torch.backends.cudnn.flags(benchmark=False, deterministic=True,
                                        allow_tf32=False)
               if x.is_cuda else nullcontext())
    with torch.autocast(device_type=x.device.type, enabled=False), backend:
        lab = _rgb_to_lab(x)
        smoothed = gaussian_blur(lab, [KERNEL_SIZE, KERNEL_SIZE], [SIGMA, SIGMA])
        gx, gy = _scharr_xy(smoothed)
        raw = (gx.square() + gy.square()).sum(dim=1, keepdim=True).sqrt()
        peak = raw.amax(dim=(-2, -1), keepdim=True)
        boundary = torch.where(peak > FLAT_EPS, raw / peak.clamp_min(FLAT_EPS),
                               torch.zeros_like(raw)).clamp(0, 1)
    values = {'boundary': boundary, 'raw_strength': raw, 'lab': lab,
              'raw_peak': peak}
    if single:
        values = {key: value[0] for key, value in values.items()}
    return values if return_details else values['boundary']


def numerical_check(device):
    # Black, white, red, green, blue have known conventional D65 Lab values.
    colours = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 0, 0],
                            [0, 1, 0], [0, 0, 1]], dtype=torch.float32,
                           device=device).view(5, 3, 1, 1).expand(5, 3, 12, 12)
    expected = torch.tensor([[0, 0, 0], [100, -.00245494, .00465342],
                              [53.24059, 80.09231, 67.20275],
                              [87.73510, -86.18303, 83.17970],
                              [32.29567, 79.18559, -107.85730]], device=device)
    result = boundary_map(colours, return_details=True)
    error = (result['lab'][:, :, 0, 0] - expected).abs().max().item()
    require(error < .002, 'Known-colour Lab conversion check failed.')
    require(result['boundary'].count_nonzero().item() == 0,
            'A uniform colour must have a zero boundary map.')
    return {'known_colour_lab_max_error': error, 'constant_colours_zero': True}


class Progress:
    def __init__(self, total):
        self.total, self.done, self.stage = total, 0, 'starting'
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.refresh, daemon=True)

    def display(self):
        elapsed = time.perf_counter() - STARTED
        eta = f'{elapsed/self.done*(self.total-self.done):.1f}s' if self.done else '--'
        print(f'Boundary checks {self.done}/{self.total} | Elapsed {elapsed:.1f}s | '
              f'ETA {eta} | {self.stage}', flush=True)

    def refresh(self):
        while not self.stopped.wait(1 if sys.stdout.isatty() else 5):
            self.display()

    def advance(self, stage):
        self.done += 1
        self.stage = stage
        self.display()

    def __enter__(self):
        self.display()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join()


def save_preview(batch, maps, path):
    n, _, height, width = batch['rgb'].shape
    row_height = height + 48
    page = Image.new('RGB', (width*3, row_height*n), 'white')
    draw = ImageDraw.Draw(page)
    for i, image_id in enumerate(batch['id_code']):
        y = i*row_height
        draw.text((8, y+5), f'TRAIN {image_id} | grade {int(batch["label"][i])}', fill='black')
        draw.text((8, y+24), 'Shared augmented RGB | epoch 0', fill='black')
        draw.text((width+8, y+5), 'Boundary strength: linear [0,1]', fill='black')
        draw.text((width+8, y+24), 'Black=0; white=1 | same pixels', fill='black')
        draw.text((width*2+8, y+5), 'Visibility view: sqrt(boundary)', fill='black')
        draw.text((width*2+8, y+24), 'Display only; NOT the SLIC input', fill='black')
        rgb = (batch['rgb'][i].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
        boundary = maps[i, 0].numpy()
        for col, array in enumerate([rgb, (boundary*255).round().astype(np.uint8),
                                     (np.sqrt(boundary)*255).round().astype(np.uint8)]):
            page.paste(Image.fromarray(array).convert('RGB'), (width*col, y+48))
    page.save(path, quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    output = None
    read_ids = []
    try:
        import dataset_v1 as d
        require(d.VERSION == 'input_pipeline_v1' and d.SIZE == 384 and d.SEED == 42,
                'Expected the existing input_pipeline_v1, size 384, seed 42.')
        require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA is unavailable.')
        device = torch.device('cuda' if args.device == 'cuda' or
                              (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        output = (args.root.resolve() / 'environment' / 'data_audit' / VERSION /
                  ('run_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                   + '_' + uuid4().hex[:6]))
        output.mkdir(parents=True, exist_ok=False)
        with Progress(8) as progress:
            project = d.load_project(args.root)
            progress.advance('existing split and RGB metadata checked')
            train = d.AptosDataset(project, 'train', verify_file_hash=True)
            batch = next(iter(d.make_loader(train, epoch=0, batch_size=4)))
            read_ids = batch['id_code']
            expected = {r['id_code']: r for r in project['rows']['train']}
            batch_check = d.validate_batch(batch, expected)
            rgb_before, cnn_before = batch['rgb'].clone(), batch['cnn'].clone()
            progress.advance('4/4 training images loaded; validation/test images read: 0')
            checks = numerical_check(device)
            progress.advance('known colours and constant-image boundary checks passed')
            rgb = batch['rgb'].to(device)
            gpu_input_before = rgb.clone()
            result = boundary_map(rgb, return_details=True)
            maps = result['boundary'].cpu()
            require(maps.shape == (4, 1, d.SIZE, d.SIZE) and torch.isfinite(maps).all().item()
                    and maps.min().item() >= 0 and maps.max().item() <= 1,
                    'Invalid boundary map shape/range.')
            require(torch.equal(rgb, gpu_input_before), 'Boundary extraction changed input.')
            progress.advance(f'4/4 maps generated on {device.type}')
            repeat = boundary_map(rgb)
            replay_error = (repeat - result['boundary']).abs().max().item()
            require(replay_error <= 1e-6, 'Repeated boundary extraction differs.')
            cpu_error = None
            if device.type == 'cuda':
                cpu_error = (boundary_map(batch['rgb']) - maps).abs().max().item()
                require(cpu_error <= 5e-4, 'CPU/CUDA map discrepancy exceeds tolerance.')
            require(torch.equal(batch['rgb'], rgb_before) and
                    torch.equal(batch['cnn'], cnn_before), 'RGB/CNN inputs were modified.')
            checks.update({'repeat_max_error': replay_error, 'cpu_cuda_max_error': cpu_error,
                           'cpu_cuda_tolerance': 5e-4, 'rgb_cnn_inputs_unchanged': True})
            progress.advance('repeatability, input preservation and backend checks passed')
            evidence = []
            for i, image_id in enumerate(read_ids):
                target = output / (image_id + '_boundary.npy')
                np.save(target, maps[i, 0].numpy(), allow_pickle=False)
                restored = np.load(target, allow_pickle=False)
                require(np.array_equal(restored, maps[i, 0].numpy()), 'NPY round-trip failed.')
                evidence.append({'id_code': image_id, 'split': 'train',
                                 'diagnosis': int(batch['label'][i]), 'epoch': 0,
                                 'input_png_sha256': project['processed'][image_id]['output_sha256'],
                                 'raw_gradient_peak': float(result['raw_peak'][i].item()),
                                 'boundary_mean': float(maps[i].mean()),
                                 'boundary_p99': float(torch.quantile(maps[i], .99)),
                                 'flat_image': bool(result['raw_peak'][i].item() <= FLAT_EPS),
                                 'output_npy': target.name, 'output_sha256': d.sha256(target)})
            with (output / 'boundary_evidence_v1.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(evidence[0]))
                writer.writeheader()
                writer.writerows(evidence)
            progress.advance('4/4 float32 evidence maps saved and read back')
            save_preview(batch, maps, output / 'boundary_preview_v1.jpg')
            progress.advance('training comparison preview saved')
            d.write_json(output / 'boundary_config_v1.json', {
                'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'settings': SETTINGS, 'sources': SOURCES, 'seed': d.SEED, 'epoch': 0,
                'input_pipeline_settings': d.SETTINGS,
                'code_sha256': d.sha256(Path(__file__)),
                'dataset_sha256': d.sha256(Path(d.__file__)),
                'reviewed_dataset_source_sha256': DATASET_SHA256,
                'source_metadata_sha256': project['source_sha256'],
                'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                             'torchvision': torchvision.__version__, 'pillow': PIL.__version__,
                             'numpy': np.__version__},
                'device': str(device),
                'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            })
            d.write_json(output / 'boundary_summary_v1.json', {
                'status': 'TECHNICAL_PASS_AWAITING_BOUNDARY_PREVIEW_REVIEW',
                'scope': 'four training images only; not segmentation/training/performance validation',
                'boundary_visually_reviewed': False, 'train_images_read': len(read_ids),
                'validation_images_read': 0, 'test_images_read': 0,
                'split_counts': {s: len(rows) for s, rows in project['rows'].items()},
                'batch_check': batch_check, 'boundary_shape': list(maps.shape),
                'device': str(device), 'checks': checks, 'evidence': evidence,
                'elapsed_seconds': time.perf_counter() - STARTED,
            })
            progress.advance('configuration, method sources and summary saved')
        print(f'\nTrain images: {len(read_ids)} | Validation: 0 | Test: 0')
        print(f'Boundary shape: {list(maps.shape)} | Device: {device}')
        print('RGB/CNN unchanged: PASS | Repeatability: PASS | Numerical checks: PASS')
        print('CPU/CUDA agreement: ' + ('PASS' if cpu_error is not None else 'NOT RUN (CPU)'))
        print('BOUNDARY MAP CHECK: PASS (small-batch technical checks only)')
        print('BOUNDARY PREVIEW: AWAITING VISUAL REVIEW')
        print(f'Total elapsed: {time.perf_counter()-STARTED:.1f}s\nSaved: {output}')
        print('Return this summary and boundary_preview_v1.jpg.')
        print('After review passes: remember to commit boundary_map_v1.py.')
        return 0
    except Exception as exc:
        print(f'\nBOUNDARY MAP CHECK: FAIL | {type(exc).__name__}: {exc}', flush=True)
        if output is not None:
            failure = {'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}',
                       'completed_batch_ids': read_ids, 'test_images_read': 0,
                       'elapsed_seconds': time.perf_counter()-STARTED}
            (output / 'boundary_failure_v1.json').write_text(
                json.dumps(failure, indent=2) + '\n', encoding='utf-8')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
