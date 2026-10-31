"""Read-only APTOS/IDRiD overlap screening before external evaluation.

Run beside idrid_cohort_v1.py and idrid_audit_v1.py:
    python idrid_overlap_v1.py
After Ctrl+C or a resolved temporary error:
    python idrid_overlap_v1.py --resume

Uses existing NumPy/Pillow only. No model, GPU, training, relabelling or image
deletion. Perceptual candidates require review; this cannot establish patient
independence. Small internal thumbnails are for screening, not model input.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

if __name__ == "__main__":
    print("Loading overlap-screening libraries...", flush=True)
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFile, ImageOps

import idrid_audit_v1 as audit
import idrid_cohort_v1 as cohort

ImageFile.LOAD_TRUNCATED_IMAGES = False
VERSION = "idrid_overlap_v1_20261030"
PROJECT = Path(__file__).resolve().parent
COHORT_RUN = "run_20261030_153346_cef194"
COHORT_SUMMARY_SHA256 = "12252226076dbc20cbf0704aab9b3cac9c1f1da4fed5306afcd7999f9206937c"
COUNTS = {"train": 2453, "validation": 527, "test": 524}
CLASS_COUNTS = {"train": [1257, 236, 646, 124, 190],
                "validation": [270, 51, 138, 27, 41], "test": [269, 51, 138, 26, 40]}
SETTINGS = {
    "exact": "SHA256(CP2_CROSS_RGB8_V1 + NUL + width + NUL + height + NUL + native RGB8 row-major bytes)",
    "exact_orientation": "Native stored orientation; no EXIF rotation in exact hash.",
    "perceptual": "Two 64-bit DCT-II hashes: full view and non-dark bounding box; grayscale 32x32 Lanczos, top-left 8x8 coefficients including DC, threshold strictly above median.",
    "perceptual_orientation": "Apply EXIF orientation only to in-memory screening views.",
    "bbox": "max(R,G,B)>7; no margin; full-view fallback if empty or either side is below 25% of the original side. Screening only.",
    "hamming_threshold": 6,
    "comparison": "Minimum over all four full/bbox view pairs; no rotation or mirror search.",
    "nearest_reference": "Also save one nearest APTOS record per IDRiD file, even above threshold; this alone is not a candidate.",
    "decision": "Never automatically exclude or relabel a perceptual match. No prediction-based filtering.",
    "cache_every": 25, "preview_pairs": 12,
}
SOURCES = [
    {"title": "ImageHash phash implementation", "url": "https://github.com/JohannesBuchner/imagehash/blob/master/imagehash/__init__.py",
     "role": "DCT/median perceptual-hash reference; this script uses a NumPy DCT matrix, not the ImageHash package."},
    {"title": "SciPy DCT-II definition", "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html",
     "role": "Unnormalised DCT-II definition and independent numerical validation; SciPy is not a runtime dependency."},
    {"title": "CP2 overlap-screening policy", "role": "Threshold 6, two screening views and review handling are project choices; neither sensitivity nor patient independence is established."},
]
require = cohort.require
_BASIS = 2.0 * np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(32)[None, :] + 1) / 64.0)
_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
PAIR_FIELDS = ("match_type aptos_id aptos_split aptos_diagnosis aptos_path idrid_key idrid_package "
               "idrid_official_split idrid_source_id idrid_dr_grade idrid_path idrid_cohort_decision "
               "idrid_candidate exact_rgb hamming64 best_aptos_view best_idrid_view affects_external_candidate").split()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_inputs(root, cohort_run, audit_run):
    """Validate fixed lists without loading any neural module or image pixels."""
    hashes = {}

    def csv_file(p):
        hashes[str(p)] = audit.sha_file(p)
        return read_csv(p)

    def json_file(p):
        hashes[str(p)] = audit.sha_file(p)
        return json.loads(p.read_text(encoding="utf-8-sig"))

    summary_path = cohort_run / "idrid_cohort_summary_v1.json"
    require(audit.sha_file(summary_path) == COHORT_SUMMARY_SHA256,
            "Use the reviewed cohort run_20261030_153346_cef194; its summary has changed or is missing.")
    cs = json_file(summary_path)
    require(cs["status"] == "MANIFEST_CHECK_PASS" and cs["source_audit_status"] == "FAIL" and
            cs["verified_original_image_and_mask_files"] == 960, "Unexpected cohort status/scope.")
    source, idrid_images, masks, crosswalk, source_hashes = cohort.read_inputs(audit_run)
    for name, digest in source_hashes.items():
        require(cs["input_sha256"][name] == digest, "Cohort/source-audit mismatch: " + name)
        hashes[str(audit_run / name)] = digest
    derived = {}
    for name in ("idrid_classification_decisions_v1.csv", "idrid_classification_external_v1.csv",
                 "idrid_segmentation_external_v1.csv"):
        derived[name] = csv_file(cohort_run / name)
        require(hashes[str(cohort_run / name)] == cs["output_sha256"][name], "Changed cohort CSV: " + name)
    decisions, classes, segments = cohort.build_cohorts(idrid_images, masks, crosswalk)
    # Compare string-normalised values with the verified derivation, not only counts.
    for name, expected in zip(derived, (decisions, classes, segments)):
        require(derived[name] == [{k: str(v) for k, v in x.items()} for x in expected],
                "Cohort content differs from its reviewed source: " + name)
    decision_by_key = {x["sample_key"]: x["decision"] for x in decisions}
    grade_by_key = {x["sample_key"]: x["dr_grade"] for x in segments}
    idrid = []
    for x in idrid_images:
        key = cohort.source_key(x)
        decision = decision_by_key[key] if x["package"] == "grading" else "segmentation_candidate"
        idrid.append(dict(dataset="idrid", key=key, source_id=x["source_id"], package=x["package"],
                          official_split=x["official_split"], relative_path=x["path"],
                          width=int(x["width"]), height=int(x["height"]),
                          diagnosis=x["dr_grade"] if x["package"] == "grading" else grade_by_key[key],
                          usage="external", candidate=decision in ("keep", "segmentation_candidate"),
                          cohort_decision=decision, expected_file_sha256=x["file_sha256"],
                          audited_rgb_sha256=x["rgb_sha256"]))

    base = root / "environment" / "data_audit"
    labels = csv_file(root / "train.csv")
    images = csv_file(base / "image_manifest.csv")
    old_decisions = csv_file(base / "data_decisions_v1.csv")
    eligible = csv_file(base / "eligible_images_v1.csv")
    split_summary = json_file(base / "split_v1" / "split_summary_v1.json")
    for name, digest in split_summary["input_file_sha256"].items():
        p = base / name
        require(p.parent == base and p.is_file(), "Unsafe/missing APTOS split source: " + name)
        actual = audit.sha_file(p)
        require(actual == digest, "APTOS split source changed: " + name)
        hashes[str(p)] = actual
    require(split_summary["seed"] == 42 and split_summary["eligible_images"] == 3504 and
            split_summary["totals"] == COUNTS, "APTOS split settings changed.")
    lookup = {x["id_code"]: x for x in images}
    label_map = {x["id_code"]: x["diagnosis"] for x in labels}
    require(len(images) == len(lookup) == len(labels) == len(label_map) == len(old_decisions) == 3662,
            "Expected the 3662 original APTOS records.")
    require(set(lookup) == set(label_map) == {x["id_code"] for x in old_decisions}, "APTOS identity sets differ.")
    for ident, x in lookup.items():
        require(re.fullmatch(r"[0-9a-f]{12}", ident) and x["diagnosis"] == label_map[ident] in {"0", "1", "2", "3", "4"},
                "APTOS ID/label mismatch: " + ident)
    kept = {x["id_code"]: x for x in old_decisions if x["status"] == "keep"}
    require(len(eligible) == len(kept) == 3504 and {x["id_code"]: x for x in eligible} == kept,
            "APTOS eligible/keep lists differ.")
    assigned, groups, pixel_hashes = {}, {}, set()
    for split, count in COUNTS.items():
        rows = csv_file(base / "split_v1" / f"{split}_v1.csv")
        require(len(rows) == count and [Counter(x["diagnosis"] for x in rows)[str(k)] for k in range(5)] == CLASS_COUNTS[split],
                "APTOS split counts changed: " + split)
        for x in rows:
            ident = x["id_code"]
            require(ident in kept and ident not in assigned and x["split"] == split and
                    {k: v for k, v in x.items() if k != "split"} == kept[ident], "APTOS split row differs from the fixed eligible record.")
            require(all(x[k] == lookup[ident][k] for k in ("diagnosis", "width", "height", "pixel_sha256")),
                    "APTOS split row differs from the original manifest.")
            require(x["pixel_sha256"] not in pixel_hashes, "Repeated eligible APTOS content hash.")
            pixel_hashes.add(x["pixel_sha256"])
            require(groups.setdefault(x["split_group_id"], split) == split, "APTOS group crosses splits.")
            assigned[ident] = split
    require(set(assigned) == set(kept) and len(groups) == 3494, "Incomplete APTOS split coverage/groups.")
    aptos = [dict(dataset="aptos", key="aptos/" + k, source_id=k, package="aptos",
                  official_split="", relative_path=f"{k}.png", width=int(x["width"]), height=int(x["height"]),
                  diagnosis=x["diagnosis"], usage=assigned.get(k, "excluded_not_used"),
                  candidate=k in assigned, cohort_decision="", expected_file_sha256="",
                  audited_rgb_sha256="", old_aptos_pixel_sha256=x["pixel_sha256"])
             for k, x in sorted(lookup.items())]
    return aptos, sorted(idrid, key=lambda x: x["key"]), hashes


def common_rgb_hash(image):
    w, h = image.size
    digest = hashlib.sha256(f"CP2_CROSS_RGB8_V1\0{w}\0{h}\0".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def phash64(image):
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coeff = _BASIS @ gray @ _BASIS.T
    bits = (coeff > np.median(coeff)).reshape(-1)
    return np.packbits(bits, bitorder="big").tobytes().hex(), bool(gray.std() > 0)


def content_view(image):
    array = np.asarray(image)
    active = np.max(array, axis=2) > 7
    xs = np.flatnonzero(active.any(axis=0)); ys = np.flatnonzero(active.any(axis=1))
    w, h = image.size
    if not len(xs) or not len(ys):
        return image.copy(), [0, 0, w, h], True
    box = [int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1]
    fallback = (box[2] - box[0] < 0.25 * w or box[3] - box[1] < 0.25 * h)
    if fallback:
        box = [0, 0, w, h]
    return image.crop(box), box, bool(fallback)


def describe_blob(blob, expected):
    with Image.open(io.BytesIO(blob)) as source:
        require(getattr(source, "n_frames", 1) == 1, "Expected one image frame: " + expected["key"])
        source.load()
        require(source.size == (expected["width"], expected["height"]), "Raw image dimensions changed: " + expected["key"])
        rgb = source.convert("RGB")
        if expected["audited_rgb_sha256"]:
            w, h = rgb.size
            old = hashlib.sha256(f"IDRID_RGB8_V1\0{w}\0{h}\0".encode())
            old.update(rgb.tobytes())
            require(old.hexdigest() == expected["audited_rgb_sha256"], "IDRiD decoded RGB differs from its audit: " + expected["key"])
        oriented = ImageOps.exif_transpose(source).convert("RGB")
        cropped, box, fallback = content_view(oriented)
        full_hash, full_ok = phash64(oriented)
        crop_hash, crop_ok = phash64(cropped)
        return dict(common_rgb_sha256=common_rgb_hash(rgb), mode=source.mode,
                    exif_orientation=int(source.getexif().get(274, 1)),
                    phash_full=full_hash, phash_bbox=crop_hash,
                    phash_full_valid=full_ok, phash_bbox_valid=crop_ok,
                    bbox=box, bbox_fallback=fallback)


class Progress(audit.Progress):
    def render(self):
        with self.lock:
            now = time.monotonic()
            eta = audit.clock((now - self.phase_start) * (self.total - self.done) / self.done) if self.done else "--"
            line = f"IDRiD overlap | {self.phase} | {self.done}/{self.total} | Elapsed {audit.clock(now-self.start)} | Phase ETA {eta}"
        print("\r" + line.ljust(180), end="", flush=True)


def fingerprint_all(rows, roots, cache, cache_path, progress):
    progress.begin("hash and fingerprint original images", len(rows))
    result, stats, reused = [], {}, 0
    try:
        for i, x in enumerate(rows):
            p = cohort.raw_path(roots[x["dataset"]], x["relative_path"])
            before = p.stat()
            blob = p.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            after = p.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Raw input changed while reading: " + str(p))
            if x["expected_file_sha256"]:
                require(digest == x["expected_file_sha256"], "IDRiD original file changed: " + str(p))
            cached = cache["records"].get(x["key"])
            if cached is not None:
                require(cached.get("record_sha256") == json_hash({k: v for k, v in cached.items() if k != "record_sha256"}),
                        "Fingerprint cache record is damaged: " + x["key"])
                require(cached["file_sha256"] == digest and cached["source_record_sha256"] == json_hash(x),
                        "Input differs from the resumable cache: " + x["key"])
                item = cached
                reused += 1
            else:
                item = {**x, **describe_blob(blob, x), "file_sha256": digest, "source_record_sha256": json_hash(x)}
                item["record_sha256"] = json_hash(item)
                cache["records"][x["key"]] = item
            result.append(item)
            stats[str(p)] = [after.st_size, after.st_mtime_ns]
            progress.step("image")
            if (i + 1) % SETTINGS["cache_every"] == 0:
                audit.save_json(cache_path, cache)
    finally:
        # One fixed checkpoint file, including on Ctrl+C; no new interrupt files.
        audit.save_json(cache_path, cache)
    return result, stats, reused


def distances_to_aptos(a_hashes, a_valid, b):
    b_hashes = np.array([list(bytes.fromhex(b[k])) for k in ("phash_full", "phash_bbox")], dtype=np.uint8)
    b_valid = np.array([b["phash_full_valid"], b["phash_bbox_valid"]], dtype=bool)
    distances = _POPCOUNT[np.bitwise_xor(a_hashes[:, :, None, :], b_hashes[None, None, :, :])].sum(axis=3)
    valid = a_valid[:, :, None] & b_valid[None, None, :]
    distances = np.where(valid, distances, 255).reshape(len(a_hashes), 4)
    return distances.min(axis=1), distances.argmin(axis=1)


def compare(aptos, idrid, progress):
    a_hashes = np.array([[list(bytes.fromhex(x[k])) for k in ("phash_full", "phash_bbox")] for x in aptos], dtype=np.uint8)
    a_valid = np.array([[x["phash_full_valid"], x["phash_bbox_valid"]] for x in aptos], dtype=bool)
    exact_lookup = defaultdict(list)
    for i, x in enumerate(aptos):
        exact_lookup[x["common_rgb_sha256"]].append(i)
    pairs, nearest = [], []
    progress.begin("compare IDRiD against all APTOS originals", len(idrid))
    for b in idrid:
        distances, views = distances_to_aptos(a_hashes, a_valid, b)
        exact = set(exact_lookup.get(b["common_rgb_sha256"], []))
        close = set(np.flatnonzero(distances <= SETTINGS["hamming_threshold"]).tolist())
        best = int(np.argmin(distances))
        selected = exact | close | ({best} if distances[best] < 255 else set())
        nearest.append(dict(idrid_key=b["key"], idrid_candidate=b["candidate"],
                            nearest_aptos_id=aptos[best]["source_id"] if distances[best] < 255 else "",
                            minimum_hamming64=int(distances[best]) if distances[best] < 255 else "",
                            exact_rgb_pairs=len(exact), perceptual_candidate_pairs=len(close - exact)))
        for j in sorted(selected):
            a = aptos[j]
            kind = "exact_rgb" if j in exact else "perceptual_candidate" if j in close else "nearest_reference"
            v = int(views[j]); valid = distances[j] < 255
            pairs.append(dict(match_type=kind, aptos_id=a["source_id"], aptos_split=a["usage"],
                              aptos_diagnosis=a["diagnosis"], aptos_path=a["relative_path"],
                              idrid_key=b["key"], idrid_package=b["package"], idrid_official_split=b["official_split"],
                              idrid_source_id=b["source_id"], idrid_dr_grade=b["diagnosis"], idrid_path=b["relative_path"],
                              idrid_cohort_decision=b["cohort_decision"], idrid_candidate=b["candidate"], exact_rgb=j in exact,
                              hamming64=int(distances[j]) if valid else "",
                              best_aptos_view=("full", "bbox")[v // 2] if valid else "",
                              best_idrid_view=("full", "bbox")[v % 2] if valid else "",
                              affects_external_candidate=kind != "nearest_reference" and b["candidate"] and a["usage"] != "excluded_not_used"))
        progress.step()
    order = {"exact_rgb": 0, "perceptual_candidate": 1, "nearest_reference": 2}
    pairs.sort(key=lambda x: (not x["affects_external_candidate"], order[x["match_type"]],
                              x["hamming64"] if x["hamming64"] != "" else 255, x["idrid_key"], x["aptos_id"]))
    return pairs, nearest


def save_previews(out, pairs, roots, fingerprint_by_key):
    selected, seen = [], set()
    for pair in pairs:
        a = fingerprint_by_key["aptos/" + pair["aptos_id"]]
        b = fingerprint_by_key[pair["idrid_key"]]
        content_pair = (a["common_rgb_sha256"], b["common_rgb_sha256"])
        if content_pair not in seen:
            seen.add(content_pair); selected.append(pair)
        if len(selected) >= SETTINGS["preview_pairs"]:
            break
    files = []
    for start in range(0, len(selected), 4):
        page = selected[start:start + 4]
        canvas = Image.new("RGB", (640, 65 + 235 * len(page)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "APTOS / IDRiD: screening comparisons", fill="black")
        draw.text((10, 28), "Only a preview subset. Similarity is not a confirmed duplicate.", fill="black")
        for line, pair in enumerate(page):
            y = 60 + line * 235
            draw.text((10, y), f"{pair['match_type']} | Hamming={pair['hamming64']} | active={pair['affects_external_candidate']}", fill="black")
            for column, key, dataset, view in [(0, "aptos/" + pair["aptos_id"], "aptos", pair["best_aptos_view"]),
                                               (1, pair["idrid_key"], "idrid", pair["best_idrid_view"])]:
                row = fingerprint_by_key[key]
                path = cohort.raw_path(roots[dataset], row["relative_path"])
                blob = path.read_bytes()
                require(hashlib.sha256(blob).hexdigest() == row["file_sha256"], "Preview source changed: " + key)
                with Image.open(io.BytesIO(blob)) as im:
                    image = ImageOps.exif_transpose(im).convert("RGB")
                if view == "bbox":
                    image = image.crop(row["bbox"])
                image.thumbnail((300, 180), Image.Resampling.LANCZOS)
                x = 10 + 320 * column
                canvas.paste(image, (x + (300-image.width)//2, y + 22 + (180-image.height)//2))
                label = row["source_id"] + " | " + (row["usage"] if dataset == "aptos" else row["package"] + "/" + row["official_split"])
                draw.text((x, y + 205), label, fill="black")
                draw.text((x, y + 219), "view=" + view, fill="black")
        name = f"overlap_preview_{start//4+1:02d}.jpg"
        buffer = io.BytesIO(); canvas.save(buffer, format="JPEG", quality=92)
        audit.atomic_bytes(out / name, buffer.getvalue()); files.append(name)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    base = root / "environment" / "data_audit"
    cohort_run = base / "idrid_cohort_v1" / COHORT_RUN
    audit_run = base / "idrid_audit_v1" / cohort.REVIEWED_RUN
    out = (args.output_dir or base / "idrid_overlap_v1" / "overlap_aptos_idrid_v1").resolve()
    roots = {"aptos": root / "train_images", "idrid": root / "IDRiD" / "raw"}
    for protected in (*roots.values(), cohort_run, audit_run, base / "split_v1"):
        require(not out.is_relative_to(protected.resolve()), "Output must be separate from input directories.")
    start = time.monotonic()
    run_owned = False
    attempt_at = datetime.now().astimezone().isoformat()
    try:
        with Progress() as progress:
            progress.begin("validate reviewed cohort and fixed APTOS splits", 1)
            aptos_rows, idrid_rows, input_hashes = load_inputs(root, cohort_run, audit_run)
            config = dict(script_version=VERSION, script_sha256=audit.sha_file(Path(__file__)),
                          project_root=str(root), input_sha256=input_hashes, settings=SETTINGS,
                          numpy=np.__version__, pillow=PIL.__version__,
                          aptos_raw_count=len(aptos_rows), idrid_raw_count=len(idrid_rows))
            signature = json_hash(config)
            cache_path = out / "fingerprint_cache_v1.json"
            if args.resume:
                require(cache_path.is_file(), "No resumable cache found. Start without --resume if this is a new run.")
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                require(cache["config_sha256"] == signature, "Sources, script or libraries differ from the saved run. Do not mix cached fingerprints.")
                expected_keys = {x["key"] for x in aptos_rows + idrid_rows}
                require(set(cache["records"]) <= expected_keys, "Unexpected records in overlap cache.")
            else:
                require(not out.exists(), f"Output already exists. Resume with --resume, or choose a new --output-dir: {out}")
                out.mkdir(parents=True)
                cache = {"config_sha256": signature, "records": {}}
                audit.save_json(out / "overlap_config_v1.json", {**config, "created_at": datetime.now().astimezone().isoformat()})
                audit.save_json(cache_path, cache)
            run_owned = True
            audit.save_json(out / "overlap_run_state_v1.json", {"status": "RUNNING", "attempt_started_at": attempt_at})
            progress.step()
            all_rows, stats, reused = fingerprint_all(aptos_rows + idrid_rows, roots, cache, cache_path, progress)
            aptos = all_rows[:len(aptos_rows)]; idrid = all_rows[len(aptos_rows):]
            pairs, nearest = compare(aptos, idrid, progress)
            progress.begin("save comparison tables and preview subset", 1)
            by_key = {x["key"]: x for x in all_rows}
            previews = save_previews(out, pairs, roots, by_key)
            for dataset, rows in (("aptos", aptos), ("idrid", idrid)):
                serial = [{**x, "bbox": json.dumps(x["bbox"])} for x in rows]
                audit.save_csv(out / f"{dataset}_fingerprints_v1.csv", serial, list(serial[0]))
            audit.save_csv(out / "idrid_overlap_pairs_v1.csv", pairs, PAIR_FIELDS)
            audit.save_csv(out / "idrid_nearest_aptos_v1.csv", nearest, list(nearest[0]))
            audit.save_json(out / "method_sources_v1.json", SOURCES)
            require(all(audit.sha_file(Path(p)) == digest for p, digest in input_hashes.items()), "Source metadata changed during screening.")
            require(all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == value for p, value in stats.items()),
                    "A raw file changed during screening; no completed result accepted.")
            flagged = [x for x in pairs if x["affects_external_candidate"]]
            descriptor_review = [x["key"] for x in all_rows if not x["phash_full_valid"] or not x["phash_bbox_valid"] or x["bbox_fallback"]]
            output_names = ["aptos_fingerprints_v1.csv", "idrid_fingerprints_v1.csv", "idrid_overlap_pairs_v1.csv",
                            "idrid_nearest_aptos_v1.csv", "overlap_config_v1.json", "method_sources_v1.json"] + previews
            summary = dict(status="REVIEW_REQUIRED" if flagged or descriptor_review else "SCREEN_COMPLETE_NO_FLAGGED_ACTIVE_PAIR",
                           script_version=VERSION, saved=str(out), cohort_run=str(cohort_run),
                           source_cohort_status="MANIFEST_CHECK_PASS", source_audit_status="FAIL",
                           aptos_original_images=len(aptos), aptos_split_counts=dict(Counter(x["usage"] for x in aptos)),
                           idrid_original_files=len(idrid), idrid_unique_rgb=len({x["common_rgb_sha256"] for x in idrid}),
                           idrid_classification_candidates=sum(x["package"] == "grading" and x["candidate"] for x in idrid),
                           idrid_segmentation_candidates=sum(x["package"] == "segmentation" and x["candidate"] for x in idrid),
                           all_cross_dataset_file_pairs=len(aptos)*len(idrid), fingerprint_cache_reused=reused,
                           exact_rgb_pairs=sum(x["match_type"] == "exact_rgb" for x in pairs),
                           perceptual_candidate_pairs=sum(x["match_type"] == "perceptual_candidate" for x in pairs),
                           nearest_reference_rows=sum(x["match_type"] == "nearest_reference" for x in pairs),
                           flagged_active_pairs=len(flagged), flagged_active_idrid_files=len({x["idrid_key"] for x in flagged}),
                           flagged_active_by_aptos_split=dict(Counter(x["aptos_split"] for x in flagged)),
                           descriptor_review_keys=descriptor_review, preview_files=previews, preview_is_subset=True,
                           input_sha256=input_hashes, output_sha256={n: audit.sha_file(out/n) for n in output_names},
                           source_metadata_unchanged=True, raw_file_stat_checks_pass=True,
                           model_forwards=0, optimizer_steps=0, raw_files_modified=0, cohort_rows_changed=0,
                           ready_for_external_inference=False, elapsed_seconds=time.monotonic()-start,
                           limits=["No same-patient conclusion: identity metadata is unavailable.",
                                   "A pHash candidate can be a false match; no candidate is automatically excluded.",
                                   "No flagged pair does not establish absence of resized, recompressed, rotated, cropped or otherwise altered duplicates.",
                                   "Old APTOS and IDRiD pixel hash strings are not directly compared. Both datasets receive the same new exact hash.",
                                   "APTOS labels/dimensions/splits are checked against existing manifests; current original pixels are freshly fingerprinted, not validated against an assumed legacy APTOS hash formula.",
                                   "Nearest reference rows above the threshold are diagnostic examples, not confirmed overlaps.",
                                   "Preview pages show at most 12 distinct content pairs; the CSV contains every detected pair."])
            audit.save_json(out / "idrid_overlap_summary_v1.json", summary)
            audit.save_json(out / "overlap_run_state_v1.json", {
                "status": "COMPLETE", "attempt_started_at": attempt_at,
                "summary_sha256": audit.sha_file(out / "idrid_overlap_summary_v1.json")})
            progress.step()
        print(f"\nOVERLAP SCREEN COMPLETE | {summary['status']}")
        print(f"APTOS originals: {len(aptos)} | IDRiD files: {len(idrid)} | Compared file pairs: {len(aptos)*len(idrid)}")
        print(f"Exact RGB pairs: {summary['exact_rgb_pairs']} | Perceptual candidates: {summary['perceptual_candidate_pairs']}")
        print(f"Flagged active pairs: {len(flagged)} | Descriptor review images: {len(descriptor_review)}")
        print("Similarity candidates require review. No files or cohort decisions were changed; model forward/optimizer steps: 0.")
        print(f"Elapsed: {audit.clock(time.monotonic()-start)}\nSaved: {out}")
        print("Return idrid_overlap_summary_v1.json, idrid_overlap_pairs_v1.csv and overlap_preview_*.jpg.")
        print("Remember to commit idrid_overlap_v1.py after this stage is checked; keep raw data out of Git.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nOVERLAP SCREEN STOPPED | {type(exc).__name__}: {exc}")
        print("No completed screen is accepted from this attempt.")
        if run_owned:
            try:
                audit.save_json(out / "overlap_run_state_v1.json", {
                    "status": "STOPPED", "attempt_started_at": attempt_at,
                    "error": f"{type(exc).__name__}: {exc}",
                    "previous_summary_retained": (out / "idrid_overlap_summary_v1.json").exists()})
            except OSError as save_error:
                print(f"Could not update run-state file: {save_error}")
        if (out / "fingerprint_cache_v1.json").exists():
            print("Fix the reported issue, then rerun with --resume (and the same optional paths). One fixed cache file is retained.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
