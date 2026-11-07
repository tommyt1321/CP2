"""CP2: aligned IDRiD external RGB / ground-truth mask preprocessing.

In C:\\CP2: python idrid_preprocess_v1.py
After interruption: python idrid_preprocess_v1.py --resume
Keep the existing preview_preprocessing_v1.py and idrid_audit_v1.py beside it.
Uses NumPy/Pillow only. No training, augmentation, prediction or new FOV mask.
The default command and --resume both reuse verified completed image records.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import csv
from pathlib import Path, PurePosixPath
import platform
import sys
import time

BOOT = time.monotonic()
if __name__ == "__main__":
    print("Loading aligned IDRiD preprocessing libraries...", flush=True)

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
import idrid_audit_v1 as audit
import preview_preprocessing_v1 as aptos

VERSION = "idrid_preprocess_v1_20261105"
ROOT = Path(__file__).resolve().parent
SIZE = 384
TYPES = ("MA", "HE", "EX", "SE", "OD")
FINAL_SUMMARY_SHA = "14dcbcae99974ed8ce4fa8263328a5ad751215d065bb18d24a1eeea142d53763"
APTOS_FUNCTION_SHA = "4142bb504f6925201d2e85e31c04584ee00fd93f59790950dac7daeaa945d06d"
MASK_DECODER_SHA = "4fe0219b27afcfea8fecece52ef4aa5f9c1760e690ca0b851ba4748c725d75d4"
PREVIEW_KEYS = tuple(f"segmentation/{part}/IDRiD_{n}" for part, n in
                     (("training", 1), ("training", 18), ("training", 36),
                      ("testing", 55), ("testing", 68), ("testing", 81)))
COLOURS = {"MA": (0, 255, 255), "HE": (255, 70, 70), "EX": (255, 230, 0),
           "SE": (80, 255, 80), "OD": (210, 100, 255)}
MASK_LIMIT = ("Nearest-neighbour 384px masks can lose small lesions even when a mask remains nonempty. "
              "Pixel counts do not measure survival of every lesion. Original mask paths/hashes and crop/pad "
              "geometry are retained for later original-resolution localisation analysis.")
METHODS = [
    {"method": "RGB transform", "source": "Existing CP2 preview_preprocessing_v1.preprocess, directly called",
     "details": "Any RGB channel >7; original-long-side 1% margin; empty/<25% bbox fallback; centred black square; LANCZOS 384px.",
     "provenance": "Project-specific accepted APTOS settings, not an IDRiD paper requirement."},
    {"method": "Aligned binary masks", "source": "Pillow official Image.resize and resampling documentation",
     "url": "https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize",
     "filter_reference": "https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters",
     "accessed": "2026-11-05", "details": "Reuse RGB bbox/pad exactly; nearest-neighbour; 0/255 PNG; no mask-derived crop or dilation.",
     "limitation": MASK_LIMIT},
    {"method": "Original mask interpretation", "source": "Existing CP2 idrid_audit_v1.binary_mask_array (r3)",
     "details": "Validate binary scalar, indexed, RGB or opaque RGBA encoding before deriving nonzero support; absent stays absent."},
    {"method": "Dataset", "source": "Porwal et al., IDRiD (2018)", "doi": "10.3390/data3030025",
     "details": "MA/HE/EX/SE are lesion annotations; OD is anatomy. Cohorts and uncertainty policy are inherited unchanged."},
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def code_hash(path):
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def function_hash(function):
    tree = ast.parse(inspect.getsource(function))
    return hashlib.sha256(ast.dump(tree.body[0], include_attributes=False).encode()).hexdigest()


def under(base, relative, must_exist=False):
    rel = PurePosixPath(relative)
    require(relative and not rel.is_absolute() and ".." not in rel.parts
            and "\\" not in relative and ":" not in relative, "Unsafe relative path: " + relative)
    p = base.joinpath(*rel.parts)
    require(not p.is_symlink() and p.resolve().is_relative_to(base.resolve()), "Path escapes its root: " + str(p))
    if must_exist:
        require(p.is_file(), "Missing input/output: " + str(p))
    return p


def protect_output(output, protected):
    require(not output.is_symlink(), "Output cannot be a symlink.")
    for p in protected:
        a, b = output.resolve(), p.resolve()
        require(not a.is_relative_to(b) and not b.is_relative_to(a), "Output overlaps protected folder: " + str(p))


class Progress(audit.Progress):
    def render(self):
        with self.lock:
            now = time.monotonic()
            eta = audit.clock((now-self.phase_start)*(self.total-self.done)/self.done) if self.done else "--"
            line = (f"IDRiD preprocess | {self.phase} | {self.done}/{self.total} | "
                    f"RGB {self.images}/588 | Masks {self.masks}/363 | "
                    f"Elapsed {audit.clock(now-BOOT)} | Phase ETA {eta}")
        print("\r" + line.ljust(200), end="", flush=True)


@contextmanager
def exclusive_run(logs):
    """OS lock releases on exit/termination; a stale lock file does not block resume."""
    logs.mkdir(parents=True, exist_ok=True)
    handle = (logs / ".idrid_preprocess.lock").open("a+b")
    locked = False
    try:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as e:
            raise ValueError("Another preprocessing process holds this run. Stop it before resuming.") from e
        yield
    finally:
        if locked:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def validate_tables(primary, classes, segments, masks):
    pk = {r["sample_key"] for r in primary}
    ck = {r["sample_key"] for r in classes}
    require(len(primary) == len(pk) == 502 and len(classes) == len(ck) == 507, "DR cohort counts changed.")
    excluded = {f"grading/training/IDRiD_{n}" for n in (216, 228, 244, 260, 318)}
    require(pk < ck and ck-pk == excluded, "Primary/sensitivity membership changed.")
    require(Counter(int(r["dr_grade"]) for r in primary) == dict(enumerate([164, 23, 163, 91, 61])),
            "Primary grades changed.")
    require(len(segments) == len({r["sample_key"] for r in segments}) == 81, "Segmentation membership changed.")
    require(len(masks) == 363 and Counter(r["lesion"] for r in masks) == {"MA":81,"HE":80,"EX":81,"SE":40,"OD":81},
            "Mask counts changed.")
    require(sum(bool(r["grading_sample_key"]) for r in segments) == 3
            and sum(r["dr_grade"] == "" for r in segments) == 78, "Segmentation DR links changed.")
    all_rows = classes + segments
    for r in all_rows:
        package = "segmentation" if "grading_link_status" in r else "grading"
        require(r["sample_key"] == f'{package}/{r["official_split"]}/{r["canonical_id"]}', "Scoped identity mismatch.")
        require(r["official_split"] in ("training", "testing") and (int(r["width"]), int(r["height"])) == (4288,2848),
                "Unexpected source dimensions/partition.")
    require(len({r["image_path"] for r in all_rows}) == 588, "Repeated source paths.")
    require(len({r["rgb_sha256"] for r in all_rows}) == 585, "Recorded exact A/B content links changed.")
    by_mask = {(r["official_split"], r["canonical_id"], r["lesion"]): r for r in masks}
    require(len(by_mask) == 363, "Duplicate mask identity.")
    missing = 0
    for r in segments:
        for kind in TYPES:
            m = by_mask.get((r["official_split"], r["canonical_id"], kind))
            if m is None:
                require(r[kind+"_availability"] == "absent_not_inferred" and not r[kind+"_mask_path"],
                        "Missing mask has been imputed.")
                missing += 1
            else:
                require(r[kind+"_availability"] == "present" and r[kind+"_mask_path"] == m["path"], "Mask link changed.")
    require(missing == 42, "Missing mask count changed.")
    require(set(PREVIEW_KEYS) <= {r["sample_key"] for r in segments}, "Fixed preview samples missing.")
    return by_mask


def load_inputs(root, final_dir):
    summary_path = final_dir / "idrid_final_cohort_summary_v1.json"
    require(audit.sha_file(summary_path) == FINAL_SUMMARY_SHA, "Use the reviewed final_aptos_idrid_v1 summary uploaded on 2026-11-05.")
    s = read_json(summary_path)
    state_path = final_dir / "final_cohort_state_v1.json"
    state = read_json(state_path)
    require(state["status"] == "COMPLETE" and state["summary_sha256"] == FINAL_SUMMARY_SHA
            and state["signature"] == s["signature"], "Final cohort is incomplete or changed.")
    sources = {summary_path: FINAL_SUMMARY_SHA, state_path: audit.sha_file(state_path)}
    for name, expected in s["output_sha256"].items():
        p = under(final_dir, name, True)
        require(audit.sha_file(p) == expected, "Final cohort output changed: " + name)
        sources[p] = expected
    primary, classes, segments, masks = [read_csv(final_dir / name) for name in
        ("idrid_classification_primary_v1.csv", "idrid_classification_sensitivity_v1.csv",
         "idrid_segmentation_primary_v1.csv", "idrid_mask_catalog_v1.csv")]
    by_mask = validate_tables(primary, classes, segments, masks)
    require(aptos.SIZE == SIZE and aptos.BLACK_THRESHOLD == 7 and function_hash(aptos.preprocess) == APTOS_FUNCTION_SHA,
            "The accepted APTOS preprocessing function/settings differ. Do not substitute a new crop.")
    require(audit.VERSION == "idrid_audit_v1_20261030_r3" and function_hash(audit.binary_mask_array) == MASK_DECODER_SHA,
            "Expected the reviewed r3 binary-mask decoder.")
    ap = root / "environment/data_audit/preprocessing_v1"
    config_path, result_path = ap / "preprocessing_config_v1.json", ap / "preprocessing_summary_v1.json"
    cfg, result = read_json(config_path), read_json(result_path)
    settings = cfg["settings"]
    require(settings == {"size":384, "black_threshold":7, "margin":"max(1, round(0.01 * max(original_size)))",
                        "square_method":"center crop content on black square using padding", "resize":"Pillow LANCZOS",
                        "smoothing":False, "normalization":False, "augmentation":False,
                        "fallback":"full original if no content or bounding-box area <25%"}, "APTOS settings differ.")
    require(result["status"] == "COMPLETE" and result["successful_outputs"] == 3504 and result["errors"] == 0,
            "Existing APTOS preprocessing did not complete.")
    require(PIL.__version__ == cfg["pillow"], f'Pillow differs from accepted APTOS processing ({cfg["pillow"]} vs {PIL.__version__}). Return this message before changing packages.')
    for p in (config_path, result_path):
        sources[p] = audit.sha_file(p)
    return dict(primary=primary, classes=classes, segments=segments, masks=masks, by_mask=by_mask,
                sources=sources, aptos_config=cfg, summary=s)


def verify_inputs(raw, evidence, progress):
    paths = {r["image_path"]:r["file_sha256"] for r in evidence["classes"] + evidence["segments"]}
    paths.update({r["path"]:r["file_sha256"] for r in evidence["masks"]})
    require(len(paths) == 951, "Expected 588 selected source RGB files plus 363 existing masks.")
    stats = {}
    progress.begin("verify 951 selected original hashes", len(paths))
    for rel, expected in paths.items():
        p = under(raw, rel, True)
        before = p.stat()
        require(audit.sha_file(p) == expected, "Original hash mismatch: " + rel)
        after = p.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Original changed while checking: " + rel)
        stats[rel] = (after.st_size, after.st_mtime_ns)
        progress.step()
    return stats


def decode_original(raw, relative, expected):
    # Decode exactly the bytes whose hash was checked, not a second file open.
    blob = under(raw, relative, True).read_bytes()
    require(hashlib.sha256(blob).hexdigest() == expected, "Original changed before decoding: " + relative)
    with Image.open(io.BytesIO(blob)) as im:
        require(getattr(im, "n_frames", 1) == 1 and int(im.getexif().get(274, 1)) == 1, "Unexpected frames/orientation: " + relative)
        im.load()
        return im.copy()


def pixel_hash(arr, prefix):
    h, w = arr.shape[:2]
    return hashlib.sha256(f"{prefix}\0{w}\0{h}\0".encode() + arr.astype(np.uint8).tobytes(order="C")).hexdigest()


def geometry(size, box):
    left, top, right, bottom = map(int, box)
    require(0 <= left < right <= size[0] and 0 <= top < bottom <= size[1], "Invalid RGB crop geometry.")
    w, h = right-left, bottom-top
    side = max(w, h)
    px, py = (side-w)//2, (side-h)//2
    return {"crop_left":left,"crop_top":top,"crop_right":right,"crop_bottom":bottom,
            "pad_left":px,"pad_top":py,"pad_right":side-w-px,"pad_bottom":side-h-py,
            "square_side":side,"scale_to_384":SIZE/side}


def transform_mask(support, geom):
    # Use image geometry, never a bbox derived from the mask.
    box = tuple(geom[k] for k in ("crop_left","crop_top","crop_right","crop_bottom"))
    arr = np.asarray(support, dtype=bool)
    before = int(arr.sum())
    cropped = arr[box[1]:box[3], box[0]:box[2]]
    kept = int(cropped.sum())
    square = Image.new("L", (geom["square_side"], geom["square_side"]), 0)
    square.paste(Image.fromarray(cropped.astype(np.uint8)*255), (geom["pad_left"],geom["pad_top"]))
    output = square.resize((SIZE,SIZE), Image.Resampling.NEAREST)
    values = np.asarray(output)
    require(np.all((values == 0) | (values == 255)), "Mask interpolation created non-binary labels.")
    after = int(np.count_nonzero(values))
    return output, {"positive_pixels_original":before, "positive_pixels_in_crop":kept,
                    "positive_pixels_outside_crop":before-kept, "positive_pixels_384":after,
                    "expected_area_pixels_384":kept*(SIZE/geom["square_side"])**2,
                    "entire_mask_lost_after_crop":bool(before and not kept),
                    "entire_mask_lost_after_resize":bool(kept and not after),
                    "small_lesion_survival":"not_established_by_aggregate_pixel_counts"}


def save_image(path, im, fmt="PNG"):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    im.save(stream, format=fmt, **({"quality":95} if fmt == "JPEG" else {}))
    audit.atomic_bytes(path, stream.getvalue())
    if fmt == "PNG":
        with Image.open(path) as loaded:
            require(loaded.mode == im.mode and loaded.size == im.size
                    and np.array_equal(np.asarray(loaded),np.asarray(im)), "PNG readback mismatch: " + str(path))
    return audit.sha_file(path)


def tint(rgb, support, colour):
    a = np.array(rgb, dtype=np.uint8, copy=True)
    a[support] = np.rint(0.48*a[support] + 0.52*np.asarray(colour)).astype(np.uint8)
    return Image.fromarray(a)


def make_preview(path, key, original, processed, native_overlay, masks, mask_stats, geom):
    # Original overlay is for visual orientation only; numerical masks are separate PNGs.
    cell, header = 384, 54
    canvas = Image.new("RGB", (3*cell, 3*(cell+header)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        try: font = ImageFont.truetype("arial.ttf", 16)
        except OSError: font = ImageFont.load_default(size=16)
    panels = [("Original RGB", key, original),
              ("Original + lesion annotations", "MA cyan; HE red; EX yellow; SE green", native_overlay),
              ("Processed RGB", "384x384; same APTOS transform", processed)]
    for kind in TYPES:
        if kind in masks:
            panel = tint(processed, np.asarray(masks[kind]) > 0, COLOURS[kind])
            stats = mask_stats[kind]
            detail = f'384px positives: {stats["positive_pixels_384"]}; cropped: {stats["positive_pixels_outside_crop"]}'
        else:
            panel = Image.new("RGB", (cell,cell), (235,235,235))
            ImageDraw.Draw(panel).text((20,160), "ABSENT: not inferred", fill="black", font=font)
            detail = "No mask output was fabricated"
        panels.append((kind + (" (anatomy)" if kind == "OD" else " (lesion)"), detail, panel))
    for index, (title, detail, panel) in enumerate(panels):
        x, y = (index%3)*cell, (index//3)*(cell+header)
        draw.text((x+8,y+6), title, fill="black", font=font)
        draw.text((x+8,y+29), detail, fill="black", font=font)
        thumbnail = ImageOps.contain(panel, (cell,cell), method=Image.Resampling.LANCZOS)
        canvas.paste(thumbnail, (x+(cell-thumbnail.width)//2,y+header+(cell-thumbnail.height)//2))
    x, y = 2*cell+10, 2*(cell+header)+12
    lines = ["GROUND TRUTH ONLY; no model/XAI", "Overlays are display copies.",
             "RGB and masks share crop and padding.", "Masks: NEAREST; binary PNGs 0/255.",
             "Small lesions may vanish at 384px.", "Original overlay is downscaled for display.",
             "OD is anatomy, not a lesion.", f'Crop: ({geom["crop_left"]}, {geom["crop_top"]},',
             f'       {geom["crop_right"]}, {geom["crop_bottom"]})',
             f'Pad left/top: {geom["pad_left"]}/{geom["pad_top"]}',
             "Review the original masks if uncertain."]
    for line in lines:
        draw.text((x,y),line,fill="black",font=font); y += 28
    return save_image(path, canvas, "JPEG")


def prepare_run(logs, images, config):
    state_path = logs / "preprocessing_state_v1.json"
    sig = config["signature"]
    if state_path.exists():
        state = read_json(state_path)
        require(state["signature"] == sig, "Existing run has different code/inputs/software. Do not mix versions.")
        require(state["status"] in ("BUILDING","COMPLETE"), "Unknown run state.")
        if state["status"] == "COMPLETE":
            summary_path = logs/"idrid_preprocessing_summary_v1.json"
            require(audit.sha_file(summary_path) == state["summary_sha256"], "Completed summary changed.")
            summary = read_json(summary_path)
            require(summary["signature"] == sig, "Completed summary belongs to another run.")
            for name, expected in summary["output_sha256"].items():
                require(audit.sha_file(under(logs,name,True)) == expected, "Completed evidence changed: " + name)
            inventory = read_json(logs/"idrid_preprocessing_files_v1.json")
            require(inventory["image_root"] == str(images), "Completed output root changed.")
            for rel, expected in inventory["record_files"].items():
                require(audit.sha_file(under(logs,rel,True)) == expected, "Completed image record changed: " + rel)
    else:
        require(not any(p.name != ".idrid_preprocess.lock" for p in logs.iterdir()), "Nonempty log folder has no owned state.")
        require(not images.exists() or not any(images.iterdir()), "Nonempty image folder has no owned run.")
        state = {"status":"BUILDING","signature":sig,"config":config}
        audit.save_json(state_path, state)
    images.mkdir(parents=True,exist_ok=True)
    owner = images / "preprocessing_owner_v1.json"
    if owner.exists():
        require(read_json(owner)["signature"] == sig, "Image folder belongs to a different run.")
    else:
        require(not any(images.iterdir()), "Image folder has files without ownership metadata.")
        audit.save_json(owner, {"signature":sig,"logs":str(logs),"version":VERSION})
    return state


def check_record(record, row, sig, images, logs):
    require(record["signature"] == sig and record["sample_key"] == row["sample_key"]
            and record["input_row"] == row, "Saved record identity/input mismatch: " + row["sample_key"])
    for base, field in ((images,"image_files"),(logs,"preview_files")):
        for rel, expected in record[field].items():
            require(audit.sha_file(under(base, rel, True)) == expected, "Completed output changed: " + rel)


def process_image(row, raw, images, logs, evidence, sig, primary_keys):
    key = row["sample_key"]
    receipt_path = under(logs, "records/"+key+".json")
    if receipt_path.exists():
        record = read_json(receipt_path)
        check_record(record, row, sig, images, logs)
        return record, True
    original = decode_original(raw, row["image_path"], row["file_sha256"])
    require(original.mode == "RGB" and original.size == (int(row["width"]), int(row["height"])), "RGB source encoding/dimensions changed: " + key)
    require(pixel_hash(np.asarray(original), "IDRID_RGB8_V1") == row["rgb_sha256"], "Decoded RGB hash differs: " + key)
    processed, box, warning = aptos.preprocess(original)
    geom = geometry(original.size, box)
    require(processed.mode == "RGB" and processed.size == (SIZE,SIZE), "APTOS RGB output differs.")
    rgb_rel = "rgb/"+key+".png"
    rgb_sha = save_image(under(images,rgb_rel),processed)
    image_files = {rgb_rel:rgb_sha}
    rgb_row = {**row, "processed_rgb_path":rgb_rel, "processed_rgb_sha256":rgb_sha,
               "output_width":SIZE,"output_height":SIZE, **geom,"crop_warning":warning,
               "primary_classification":key in primary_keys,"sensitivity_classification":key.startswith("grading/"),
               "segmentation_primary":key.startswith("segmentation/")}
    preview = key in PREVIEW_KEYS
    native_overlay = original.copy() if preview else None
    previews, output_masks, stats_by_type, masks_rows = {}, {}, {}, []
    if key.startswith("segmentation/"):
        for kind in TYPES:
            source = evidence["by_mask"].get((row["official_split"],row["canonical_id"],kind))
            common = {"sample_key":key, "official_split":row["official_split"], "canonical_id":row["canonical_id"],
                      "mask_type":kind,"role":"anatomy" if kind == "OD" else "lesion",
                      "availability":row[kind+"_availability"], "original_mask_path":row[kind+"_mask_path"],
                      "original_mask_sha256":"", "processed_mask_path":"", "processed_mask_sha256":""}
            if source is None:
                masks_rows.append(common)
                continue
            im = decode_original(raw, source["path"], source["file_sha256"])
            require(im.size == original.size, "Mask/RGB dimensions differ: " + source["path"])
            arr, encoding, palette = audit.binary_mask_array(im)
            support = arr != 0
            require(encoding == source["value_encoding"] and int(support.sum()) == int(source["positive_pixels"])
                    and pixel_hash(support, "IDRID_BINARY_SUPPORT_V1") == source["binary_support_sha256"],
                    "Decoded mask differs from reviewed audit: " + source["path"])
            transformed, stats = transform_mask(support, geom)
            mask_rel = "masks/"+key+"_"+kind+".png"
            digest = save_image(under(images,mask_rel),transformed)
            image_files[mask_rel] = digest
            masks_rows.append({**common,"original_mask_sha256":source["file_sha256"],
                               "original_binary_support_sha256":source["binary_support_sha256"],
                               "processed_mask_path":mask_rel,"processed_mask_sha256":digest,
                               "value_encoding":encoding, **stats, **geom})
            if preview:
                output_masks[kind], stats_by_type[kind] = transformed, stats
                if kind != "OD":
                    native_overlay = tint(native_overlay, support, COLOURS[kind])
            del arr, support, im
    if preview:
        rel = f"idrid_alignment_preview_{PREVIEW_KEYS.index(key)+1:02d}.jpg"
        previews[rel] = make_preview(logs/rel,key,original,processed,native_overlay,output_masks,stats_by_type,geom)
    record = {"signature":sig,"sample_key":key,"input_row":row,"rgb_row":rgb_row,
              "mask_rows":masks_rows,"image_files":image_files,"preview_files":previews}
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    audit.save_json(receipt_path, record)  # Commit after every PNG/preview has been saved and verified.
    return record, False


def save_table(logs, name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    audit.save_csv(logs/name,rows,fields)
    return audit.sha_file(logs/name)


def finalise(logs, images, evidence, records, config, raw, raw_stats, progress, state):
    rgb = [r["rgb_row"] for r in records]
    mask_rows = [m for r in records for m in r["mask_rows"]]
    files = {k:v for r in records for k,v in r["image_files"].items()}
    previews = {k:v for r in records for k,v in r["preview_files"].items()}
    require(len(rgb) == 588 and len(files) == 951 and len(mask_rows) == 405 and len(previews) == 6, "Incomplete batch output.")
    require(Counter(r["availability"] for r in mask_rows) == {"present":363,"absent_not_inferred":42},
            "Present/missing mask states changed in output records.")
    require(len({r["sample_key"] for r in rgb}) == 588, "Repeated output sample identities.")
    require({p.relative_to(images).as_posix() for p in images.rglob("*.png")} == set(files), "Unexpected/missing processed PNGs.")
    progress.begin("verify output hashes and source preservation", len(files)+len(raw_stats)+len(evidence["sources"]))
    for rel, digest in files.items():
        require(audit.sha_file(under(images,rel,True)) == digest, "Output hash changed: " + rel)
        progress.step()
    for rel, expected_stat in raw_stats.items():
        st = under(raw,rel,True).stat()
        require((st.st_size,st.st_mtime_ns) == expected_stat, "Original file changed during run: " + rel)
        progress.step()
    for p, digest in evidence["sources"].items():
        require(audit.sha_file(p) == digest, "Source metadata changed during run: " + str(p))
        progress.step()
    summary_path = logs/"idrid_preprocessing_summary_v1.json"
    if state["status"] == "COMPLETE":
        require(audit.sha_file(summary_path) == state["summary_sha256"], "Completed summary changed.")
        summary = read_json(summary_path)
        for name, expected in summary["output_sha256"].items():
            require(audit.sha_file(under(logs,name,True)) == expected, "Completed evidence changed: " + name)
        return summary
    primary_keys = {r["sample_key"] for r in evidence["primary"]}
    primary = [r for r in rgb if r["sample_key"] in primary_keys]
    classes = [r for r in rgb if r["sample_key"].startswith("grading/")]
    segments = [r for r in rgb if r["sample_key"].startswith("segmentation/")]
    require((len(primary),len(classes),len(segments)) == (502,507,81), "Output cohort membership changed.")
    tables = {"idrid_rgb_manifest_384_v1.csv":rgb,"idrid_mask_manifest_384_v1.csv":mask_rows,
              "idrid_classification_primary_384_v1.csv":primary,"idrid_classification_sensitivity_384_v1.csv":classes,
              "idrid_segmentation_384_v1.csv":segments}
    issues = []
    for r in rgb:
        if r["crop_warning"]:
            issues.append({"sample_key":r["sample_key"],"mask_type":"","issue":"RGB_CROP_FALLBACK","details":r["crop_warning"]})
    for r in mask_rows:
        if r["availability"] != "present":
            continue
        for field, issue in (("positive_pixels_outside_crop","MASK_FOREGROUND_OUTSIDE_RGB_CROP"),
                             ("entire_mask_lost_after_resize","WHOLE_POSITIVE_MASK_VANISHED_AT_384")):
            if r[field]:
                issues.append({"sample_key":r["sample_key"],"mask_type":r["mask_type"],"issue":issue,"details":str(r[field])})
    progress.begin("save manifests, geometry and methods", len(tables)+5)
    hashes = dict(previews)
    for name, rows in tables.items():
        hashes[name] = save_table(logs,name,rows); progress.step()
    name = "idrid_preprocessing_review_v1.csv"
    audit.save_csv(logs/name,issues,["sample_key","mask_type","issue","details"])
    hashes[name] = audit.sha_file(logs/name); progress.step()
    for name, data in (("idrid_preprocessing_config_v1.json",config),("method_sources_v1.json",METHODS)):
        audit.save_json(logs/name,data); hashes[name] = audit.sha_file(logs/name); progress.step()
    receipt_hashes = {"records/"+r["sample_key"]+".json":audit.sha_file(under(logs,"records/"+r["sample_key"]+".json",True)) for r in records}
    name = "idrid_preprocessing_files_v1.json"
    audit.save_json(logs/name,{"image_root":str(images),"image_files":files,"record_files":receipt_hashes})
    hashes[name] = audit.sha_file(logs/name); progress.step()
    summary = {"status":"PREPROCESSING_FILE_CHECK_PASS","script_version":VERSION,"signature":config["signature"],
               "created_at":datetime.now(timezone.utc).isoformat(),"technical_scope":"File/hash/geometry/binary-label checks; visual alignment awaiting review.",
               "processed_rgb_files":588,"distinct_source_RGB_hashes":585,"classification_primary":502,
               "classification_sensitivity":507,"segmentation":81,"processed_existing_masks":363,
               "absent_mask_type_entries_not_imputed":42,"mask_manifest_rows":405,"original_hashes_verified_this_stage":951,
               "grade_links_preserved":3,"segmentation_unknown_grade_preserved":78,
               "source_audit_status_unchanged":"FAIL","source_screen_status_unchanged":"REVIEW_REQUIRED",
               "source_final_cohort_summary_sha256":FINAL_SUMMARY_SHA,"same_APTOS_RGB_function":True,
               "mask_interpolation":"Pillow NEAREST, binary 0/255", "mask_resolution_limitation":MASK_LIMIT,
               "automatic_review_issue_count":len(issues),"automatic_review_counts":dict(Counter(r["issue"] for r in issues)),
               "visual_alignment":"AWAITING_REVIEW","ready_for_external_inference":False,
               "raw_files_modified_by_script":0,"original_input_hashes_verified_before_processing":True,
               "source_stats_unchanged_at_finish":True,"input_metadata_hashes_unchanged":True,
               "APTOS_image_files_read":0,"model_forwards":0,"optimizer_steps":0,
               "image_root":str(images),"saved":str(logs),"elapsed_seconds_this_process":time.monotonic()-BOOT,
               "preview_selection":list(PREVIEW_KEYS),"output_sha256":hashes,
               "next_step":"Review six alignment previews and any crop/resolution issues; then specify frozen-model external evaluation/localisation."}
    audit.save_json(summary_path,summary)
    audit.save_json(logs/"preprocessing_state_v1.json", {"status":"COMPLETE","signature":config["signature"],
                    "summary_sha256":audit.sha_file(summary_path)})
    progress.step()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--final-cohort-dir",type=Path)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--log-dir",type=Path)
    parser.add_argument("--resume",action="store_true",help="Reuse verified per-image commits; default command is equally restart-safe.")
    args = parser.parse_args()
    root = args.project_root.resolve()
    base = root/"environment/data_audit"
    raw = root/"IDRiD/raw"
    final_dir = (args.final_cohort_dir or base/"idrid_final_cohort_v1/final_aptos_idrid_v1").resolve()
    images = (args.output_dir or root/"IDRiD/processed_384_v1").resolve()
    logs = (args.log_dir or base/"idrid_preprocessing_v1/aligned_384_v1").resolve()
    try:
        protected = [raw,root/"train_images",root/"processed_images_384_v1",final_dir,
                     *[base/n for n in ("split_v1","preprocessing_v1","training_v1","evaluation_v1","xai_test_v1",
                                        "idrid_audit_v1","idrid_cohort_v1","idrid_overlap_v1","idrid_overlap_review_v1")]]
        protect_output(images,protected+[logs]); protect_output(logs,protected+[images])
        audit.load_libraries()
        with Progress() as progress:
            progress.begin("verify frozen cohort and accepted APTOS transform",1)
            evidence = load_inputs(root,final_dir)
            config = {"version":VERSION,"project_root":str(root),"raw_root":str(raw),"image_root":str(images),"log_root":str(logs),
                      "cohort_summary_sha256":FINAL_SUMMARY_SHA,"source_sha256":{str(p):h for p,h in evidence["sources"].items()},
                      "code_LF_normalized_sha256":{Path(m.__file__).name:code_hash(Path(m.__file__)) for m in (aptos,audit)},
                      "script_LF_normalized_sha256":code_hash(Path(__file__)),
                      "python":platform.python_version(),"pillow":PIL.__version__,"numpy":np.__version__,
                      "APTOS_settings":evidence["aptos_config"]["settings"],"preview_keys":list(PREVIEW_KEYS),
                      "mask_interpolation":"NEAREST","absent_masks":"not_imputed","neural_execution":False}
            config["signature"] = signature(config)
            progress.step()
            with exclusive_run(logs):
                state = prepare_run(logs,images,config)
                raw_stats = verify_inputs(raw,evidence,progress)
                records, reused = [], 0
                primary_keys = {r["sample_key"] for r in evidence["primary"]}
                progress.begin("process or verify saved RGB/masks",588)
                for row in evidence["classes"]+evidence["segments"]:
                    record, was_reused = process_image(row,raw,images,logs,evidence,config["signature"],primary_keys)
                    records.append(record); reused += was_reused
                    with progress.lock:
                        progress.masks += sum(r["availability"] == "present" for r in record["mask_rows"])
                    progress.step("image")
                summary = finalise(logs,images,evidence,records,config,raw,raw_stats,progress,state)
    except (Exception,KeyboardInterrupt) as exc:
        print(f"\nIDRID PREPROCESSING STOPPED | {type(exc).__name__}: {exc}")
        print("Verified image commits are retained in the same folder. Resolve the issue, then:")
        print("python idrid_preprocess_v1.py --resume")
        return 1
    print("\nIDRID PREPROCESSING FILE CHECK: PASS (technical checks only)")
    print(f"RGB: 588 | Existing masks: 363 | Absent entries: 42 (not imputed) | Reused image commits: {reused}")
    print("Primary DR: 502 | Sensitivity DR: 507 (shared RGB outputs) | Segmentation: 81")
    print("Same APTOS RGB function | Shared crop/pad geometry | Masks: NEAREST 0/255")
    print(f'Automatic review entries: {summary["automatic_review_issue_count"]} | ALIGNMENT PREVIEW: AWAITING REVIEW')
    print("Small lesions may disappear at 384px; original masks and transform geometry remain available.")
    print("APTOS pixels read: 0 | Model forward: 0 | Optimizer steps: 0 | Raw/cohort files unchanged by script.")
    print(f"Elapsed this process: {audit.clock(time.monotonic()-BOOT)}\nImages: {images}\nLogs: {logs}")
    print("Return idrid_preprocessing_summary_v1.json, idrid_preprocessing_review_v1.csv and idrid_alignment_preview_01..06.jpg.")
    print("After output review: remember to commit idrid_preprocess_v1.py; keep generated data out of Git.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
