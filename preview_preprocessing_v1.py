"""Training-only preprocessing preview. Does not change originals or splits."""
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


SIZE = 384
BLACK_THRESHOLD = 7  # Provisional 8-bit RGB threshold; inspect previews before freezing.
SEED = 42


def preprocess(img):
    rgb = img.convert('RGB')
    arr = np.asarray(rgb)
    # Conservative bounding box: include every pixel with any channel > 7.
    # This detects non-black content, not an anatomical retinal segmentation.
    mask = arr.max(axis=2) > BLACK_THRESHOLD
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    warning = ''
    if not len(xs) or not len(ys):
        box = (0, 0, rgb.width, rgb.height)
        warning = 'NO_CONTENT_ABOVE_THRESHOLD: kept full image'
    else:
        margin = max(1, round(0.01 * max(rgb.size)))
        box = (max(0, int(xs[0]) - margin), max(0, int(ys[0]) - margin),
               min(rgb.width, int(xs[-1]) + 1 + margin),
               min(rgb.height, int(ys[-1]) + 1 + margin))
        retained_fraction = ((box[2]-box[0]) * (box[3]-box[1])) / (rgb.width * rgb.height)
        if retained_fraction < 0.25:
            box = (0, 0, rgb.width, rgb.height)
            warning = 'SMALL_CONTENT_BOX: kept full image for review'
    crop = rgb.crop(box)
    side = max(crop.size)
    square = Image.new('RGB', (side, side), 'black')
    square.paste(crop, ((side-crop.width)//2, (side-crop.height)//2))
    output = square.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return output, box, warning


def main():
    root = Path.cwd()
    audit = root / 'environment' / 'data_audit'
    with (audit / 'split_v1' / 'train_v1.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2453 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Expected the fixed 2453-image training manifest')
    rng = random.Random(SEED)
    chosen = []
    for label in '01234':
        pool = sorted((r for r in rows if r['diagnosis'] == label), key=lambda r: r['id_code'])
        chosen.extend(rng.sample(pool, 2))
    for row in chosen:
        path = root / 'train_images' / (row['id_code'] + '.png')
        if not path.is_file():
            raise FileNotFoundError(path)
    folder = audit / 'preprocessing_preview_v1'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    cell, heading, row_height = 384, 65, 449
    for page in range(2):
        canvas = Image.new('RGB', (cell*2, row_height*5), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(chosen[page*5:(page+1)*5]):
            image_id = row['id_code']
            with Image.open(root / 'train_images' / (image_id + '.png')) as original:
                rgb = original.convert('RGB')
                processed, box, warning = preprocess(rgb)
            processed.save(folder / (image_id + '_384.png'))
            y = i * row_height
            draw.text((8,y+8), f"TRAIN | {image_id} | label {row['diagnosis']}", fill='black')
            draw.text((8,y+26), f'Original {rgb.width}x{rgb.height}', fill='black')
            draw.text((cell+8,y+8), 'Non-black crop + square padding + resize', fill='black')
            draw.text((cell+8,y+26), f'384x384 | threshold={BLACK_THRESHOLD}', fill='black')
            draw.text((cell+8,y+43), 'REVIEW: fallback used' if warning else 'No smoothing or normalization applied', fill='red' if warning else 'black')
            thumb = ImageOps.contain(rgb, (cell, cell), method=Image.Resampling.LANCZOS)
            canvas.paste(thumb, ((cell-thumb.width)//2, y+heading+(cell-thumb.height)//2))
            canvas.paste(processed, (cell,y+heading))
            draw.line((0,y+row_height-1,cell*2,y+row_height-1), fill='gray')
            records.append({'id_code':image_id, 'diagnosis':row['diagnosis'],
                            'split':'train', 'original_width':rgb.width, 'original_height':rgb.height,
                            'crop_left':box[0], 'crop_top':box[1], 'crop_right':box[2], 'crop_bottom':box[3],
                            'black_threshold':BLACK_THRESHOLD, 'output_size':SIZE, 'warning':warning})
        canvas.save(folder / f'preview_{page+1:02d}.jpg', quality=95)
    with (folder / 'preview_manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('Training images previewed:', len(records))
    print('Images requiring fallback review:', sum(bool(r['warning']) for r in records))
    print('Saved:', folder)
    print('Upload preview_01.jpg and preview_02.jpg for review.')
    print('PREVIEW GENERATION: COMPLETE (not preprocessing approval)')
    print('Original images, labels and split manifests unchanged.')


if __name__ == '__main__':
    main()
