"""Apply the reviewed preprocessing to the fixed 3,504-image APTOS split.

Run from C:\\CP2: python preprocess_dataset_v1.py
Keep preview_preprocessing_v1.py beside this script. Requires Pillow and NumPy.
Outputs: processed_images_384_v1/<id_code>.png and audit CSV/JSON files.
No augmentation, smoothing, contrast adjustment or normalization is applied.
Fallbacks are saved for review; errors are reported, never silently excluded.
"""
import csv
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

try:
    import preview_preprocessing_v1 as preview
except ModuleNotFoundError as exc:
    raise SystemExit('Keep preview_preprocessing_v1.py beside this script.') from exc


EXPECTED = {'train': 2453, 'validation': 527, 'test': 524}
# The immediately preceding release differs only in progress/timing code.
# Keep all input, preprocessing and software checks when accepting that release.
PRE_TIMER_HASHES = {
    'e1da0ba37df8d7a31189f85a5971812df625bbdd272ef8733ef70f187be6132c',
    '53297d7ee9aa6a9c176dc7d2e15bafbe287426f1426a5bc8eb841e2bc920f325',
}
FIELDS = ['id_code', 'diagnosis', 'split', 'split_group_id', 'source_path',
          'output_path', 'original_width', 'original_height', 'crop_left',
          'crop_top', 'crop_right', 'crop_bottom', 'padding_left', 'padding_top',
          'padding_right', 'padding_bottom', 'output_width', 'output_height',
          'output_sha256', 'status', 'warning', 'error']


def duration(seconds):
    if seconds is None:
        return '--:--:--'
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class ProgressTimer:
    """Refresh an interactive terminal every second, even during a slow image.

    Redirected output receives a new line every 10 seconds to avoid log spam.
    ETA uses mean time per attempted image; errors are included in the count.
    """

    def __init__(self, total):
        self.total = total
        self.done = self.errors = self.review = 0
        self.started = time.perf_counter()
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.interactive = sys.stdout.isatty()
        self.thread = threading.Thread(target=self._refresh, daemon=True)
        self.line_width = 0

    @property
    def elapsed(self):
        return time.perf_counter() - self.started

    def update(self, done, errors, review):
        with self.lock:
            self.done, self.errors, self.review = done, errors, review

    def _display(self):
        with self.lock:
            done, errors, review = self.done, self.errors, self.review
        elapsed = self.elapsed
        speed = done / elapsed if elapsed > 0 else 0
        eta = (self.total - done) / speed if speed else None
        percent = 100 * done / self.total if self.total else 100
        message = (f'Processed {done}/{self.total} ({percent:.1f}%) | '
                   f'Elapsed {duration(elapsed)} | ETA {duration(eta)} | '
                   f'{speed:.2f} img/s | errors={errors} | review={review}')
        if self.interactive:
            self.line_width = max(self.line_width, len(message))
            print('\r' + message.ljust(self.line_width), end='', flush=True)
        else:
            print(message, flush=True)

    def _refresh(self):
        interval = 1 if self.interactive else 10
        while not self.stopped.wait(interval):
            self._display()

    def __enter__(self):
        self._display()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stopped.set()
        self.thread.join()
        self._display()
        if self.interactive:
            print(flush=True)


def same_processing_config(previous, current):
    if previous == current:
        return True
    if previous.get('batch_script_sha256') not in PRE_TIMER_HASHES:
        return False
    expected = dict(previous)
    expected['batch_script_sha256'] = current['batch_script_sha256']
    return expected == current


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_splits(audit):
    rows, sources = [], {}
    required = {'id_code', 'diagnosis', 'split', 'split_group_id', 'pixel_sha256',
                'width', 'height', 'status'}
    for split, count in EXPECTED.items():
        path = audit / 'split_v1' / (split + '_v1.csv')
        items = read_csv(path)
        require(len(items) == count, f'{path.name}: expected {count} rows')
        require(all(required <= row.keys() for row in items), 'Missing manifest fields')
        require(all(r['split'] == split and r['status'] == 'keep' for r in items),
                f'{path.name}: unexpected split or excluded record')
        rows.extend(items)
        sources[path.name] = file_sha(path)
    require(len({r['id_code'] for r in rows}) == len(rows), 'Repeated image ID')
    require(len({r['pixel_sha256'] for r in rows}) == len(rows), 'Repeated pixel hash')
    require(all(re.fullmatch(r'[0-9a-f]{12}', r['id_code']) for r in rows), 'Invalid image ID')
    require(all(r['diagnosis'] in '01234' and len(r['diagnosis']) == 1
                and r['split_group_id'] and r['pixel_sha256'] for r in rows),
            'Invalid label, group ID or pixel hash')
    group_splits = {}
    for row in rows:
        group = row['split_group_id']
        require(group_splits.setdefault(group, row['split']) == row['split'],
                'A group crosses partitions')
    require(len(group_splits) == 3494, 'Expected 3494 split groups')
    eligible_path = audit / 'eligible_images_v1.csv'
    eligible = read_csv(eligible_path)
    require(len(eligible) == len(rows), 'Eligible count differs')
    actual = {r['id_code']: {k: v for k, v in r.items() if k != 'split'} for r in rows}
    require(actual == {r['id_code']: r for r in eligible}, 'Splits differ from eligible records')
    sources['eligible_images_v1.csv'] = file_sha(eligible_path)
    return sorted(rows, key=lambda r: r['id_code']), sources


def write_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def process_one(row, root, output):
    record = {name: '' for name in FIELDS}
    record.update({k: row[k] for k in ['id_code', 'diagnosis', 'split', 'split_group_id']})
    source = root / 'train_images' / (row['id_code'] + '.png')
    destination = output / source.name
    temporary = destination.with_suffix('.png.tmp')
    record['source_path'] = source.relative_to(root).as_posix()
    record['output_path'] = destination.relative_to(root).as_posix()
    try:
        with Image.open(source) as image:
            image.load()
            width, height = image.size
            require((width, height) == (int(row['width']), int(row['height'])),
                    'Original size differs from audited record')
            # This is the exact function used for the reviewed previews.
            processed, box, warning = preview.preprocess(image)
        require(processed.mode == 'RGB' and processed.size == (384, 384),
                'Unexpected preprocessing output')
        left, top, right, bottom = box
        require(0 <= left < right <= width and 0 <= top < bottom <= height,
                'Crop box outside the original image')
        crop_width, crop_height = right-left, bottom-top
        side = max(crop_width, crop_height)
        pad_left, pad_top = (side-crop_width)//2, (side-crop_height)//2
        record.update(original_width=width, original_height=height,
                      crop_left=left, crop_top=top, crop_right=right, crop_bottom=bottom,
                      padding_left=pad_left, padding_top=pad_top,
                      padding_right=side-crop_width-pad_left,
                      padding_bottom=side-crop_height-pad_top, warning=warning)
        processed.save(temporary, format='PNG')
        # Decode the saved file to check actual saved size/mode/content.
        with Image.open(temporary) as saved:
            saved.load()
            require(saved.size == (384, 384) and saved.mode == 'RGB'
                    and saved.tobytes() == processed.tobytes(), 'Saved PNG verification failed')
        temporary.replace(destination)
        record.update(output_width=384, output_height=384,
                      output_sha256=file_sha(destination),
                      status='review' if warning else 'ok')
    except Exception as exc:
        record.update(status='error', error=f'{type(exc).__name__}: {exc}')
        if temporary.exists():
            temporary.unlink()
    return record


def main():
    run_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    print('Checking manifests and preprocessing settings...', flush=True)
    root = Path.cwd()
    audit = root / 'environment' / 'data_audit'
    output = root / 'processed_images_384_v1'
    logs = audit / 'preprocessing_v1'
    require(preview.SIZE == 384 and preview.BLACK_THRESHOLD == 7,
            'Preview settings changed; expected SIZE=384 and BLACK_THRESHOLD=7')
    rows, source_hashes = load_splits(audit)
    config = {
        'version': 'preprocessing_v1', 'source_csv_sha256': source_hashes,
        'preview_script_sha256': file_sha(Path(preview.__file__)),
        'batch_script_sha256': file_sha(Path(__file__)),
        'settings': {'size': 384, 'black_threshold': 7,
                     'margin': 'max(1, round(0.01 * max(original_size)))',
                     'square_method': 'center crop content on black square using padding',
                     'resize': 'Pillow LANCZOS', 'smoothing': False,
                     'normalization': False, 'augmentation': False,
                     'fallback': 'full original if no content or bounding-box area <25%'},
        'python': sys.version.split()[0], 'pillow': PIL.__version__, 'numpy': np.__version__
    }
    config_path = logs / 'preprocessing_config_v1.json'
    if config_path.exists():
        require(same_processing_config(json.loads(config_path.read_text(encoding='utf-8')), config),
                'Existing v1 run uses different inputs/code/software; do not mix versions')
    else:
        require(not output.exists() or not any(output.iterdir()),
                'Output folder is not empty and has no matching run configuration')
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    summary_path = logs / 'preprocessing_summary_v1.json'
    write_json(summary_path, {'status': 'RUNNING', 'expected_images': len(rows)})
    manifest = logs / 'preprocessing_manifest_v1.csv'
    issues = logs / 'preprocessing_issues_v1.csv'
    counts = Counter()
    per_split = Counter()
    with manifest.open('w', encoding='utf-8', newline='') as all_stream, \
            issues.open('w', encoding='utf-8', newline='') as issues_stream:
        all_writer = csv.DictWriter(all_stream, fieldnames=FIELDS)
        issue_writer = csv.DictWriter(issues_stream, fieldnames=FIELDS)
        all_writer.writeheader()
        issue_writer.writeheader()
        with ProgressTimer(len(rows)) as progress:
            for index, row in enumerate(rows, 1):
                record = process_one(row, root, output)
                all_writer.writerow(record)
                counts[record['status']] += 1
                if record['status'] != 'error':
                    per_split[row['split']] += 1
                if record['status'] != 'ok':
                    issue_writer.writerow(record)
                    issues_stream.flush()
                if index % 100 == 0 or index == len(rows):
                    all_stream.flush()
                progress.update(index, counts['error'], counts['review'])
        processing_seconds = progress.elapsed
    expected_names = {r['id_code'] + '.png' for r in rows}
    actual_names = {p.name for p in output.glob('*.png')}
    complete = (counts['error'] == 0 and actual_names == expected_names
                and dict(per_split) == EXPECTED)
    summary = {
        'started_utc': started_utc,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': round(time.perf_counter() - run_started, 3),
        'processing_seconds': round(processing_seconds, 3),
        'average_images_per_second': round(len(rows) / processing_seconds, 3)
        if processing_seconds > 0 else None,
        'status': 'COMPLETE' if complete else 'INCOMPLETE',
        'ready_for_next_step': complete and counts['review'] == 0,
        'expected_images': len(rows), 'successful_outputs': counts['ok'] + counts['review'],
        'errors': counts['error'], 'review_required': counts['review'],
        'successful_by_split': dict(per_split),
        'output_filenames_match': actual_names == expected_names,
        'pixel_audit_recomputed': False,
        'limitation': 'Automated file checks do not prove that all retinal content is preserved.'
    }
    write_json(summary_path, summary)
    print('\nSuccessful outputs:', summary['successful_outputs'])
    print('By split:', dict(per_split))
    print('Errors:', counts['error'])
    print('Review required (fallback used):', counts['review'])
    print('BATCH FILE CHECK:', 'PASS' if complete else 'FAIL')
    print('NEXT STEP:', 'READY' if summary['ready_for_next_step'] else 'CHECK ISSUES CSV')
    print('Total elapsed:', duration(summary['elapsed_seconds']))
    print('Images:', output)
    print('Logs:', logs)
    print('Original images and split manifests unchanged.')
    return 0 if complete else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print('STOP:', exc, file=sys.stderr)
        sys.exit(1)
