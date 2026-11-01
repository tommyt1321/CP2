"""Export every candidate from the completed APTOS/IDRiD overlap screen.

In C:\\CP2: python idrid_overlap_review_v1.py
Continue an interrupted export: python idrid_overlap_review_v1.py --resume

Reuses the reviewed summary/CSV/fingerprints; does not repeat the full scan.
Requires Pillow and the existing idrid_audit_v1.py / idrid_cohort_v1.py.
Creates review copies only. No model, exclusion, grading or raw-image edit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

if __name__ == "__main__":
    print("Loading overlap-review export libraries...", flush=True)
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
import idrid_audit_v1 as audit
import idrid_cohort_v1 as cohort

VERSION = "idrid_overlap_review_v1_20261031"
PROJECT = Path(__file__).resolve().parent
REVIEWED_SUMMARY_SHA256 = "75aa39f5b671d4d05a8169beda6449a86b6ac3c119d6b5a7b52f99675a161f9b"
SOURCE_FILES = ("idrid_overlap_summary_v1.json", "idrid_overlap_pairs_v1.csv",
                "aptos_fingerprints_v1.csv", "idrid_fingerprints_v1.csv",
                "overlap_config_v1.json", "method_sources_v1.json", "overlap_run_state_v1.json")
require = cohort.require


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def flag(value):
    require(value in ("True", "False"), "Invalid CSV Boolean: " + repr(value))
    return value == "True"


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Progress(audit.Progress):
    def render(self):
        with self.lock:
            now = time.monotonic()
            eta = audit.clock((now-self.phase_start)*(self.total-self.done)/self.done) if self.done else "--"
            line = (f"Overlap review export | {self.phase} | {self.done}/{self.total} | "
                    f"Elapsed {audit.clock(now-self.start)} | Phase ETA {eta}")
        print("\r" + line.ljust(180), end="", flush=True)


def load_inputs(source):
    hashes = {name: audit.sha_file(source/name) for name in SOURCE_FILES}
    require(hashes[SOURCE_FILES[0]] == REVIEWED_SUMMARY_SHA256,
            "Expected the reviewed completed overlap summary; source run differs.")
    summary = read_json(source/SOURCE_FILES[0])
    state = read_json(source/"overlap_run_state_v1.json")
    require(state["status"] == "COMPLETE" and state["summary_sha256"] == hashes[SOURCE_FILES[0]],
            "The source overlap run is not COMPLETE with this summary.")
    for name in SOURCE_FILES[1:-1]:
        require(hashes[name] == summary["output_sha256"][name], "Source output changed: " + name)
    require(summary["status"] == "REVIEW_REQUIRED" and summary["source_audit_status"] == "FAIL"
            and summary["source_cohort_status"] == "MANIFEST_CHECK_PASS",
            "Unexpected source status; this script does not override it.")
    config = read_json(source/"overlap_config_v1.json")
    by_key = {}
    for dataset, count_field in (("aptos", "aptos_original_images"), ("idrid", "idrid_original_files")):
        rows = read_csv(source/f"{dataset}_fingerprints_v1.csv")
        require(len(rows) == summary[count_field], "Fingerprint count mismatch: " + dataset)
        for row in rows:
            require(row["dataset"] == dataset and row["key"] not in by_key, "Repeated/wrong fingerprint key.")
            for field in ("candidate", "phash_full_valid", "phash_bbox_valid", "bbox_fallback"):
                flag(row[field])
            row["bbox"] = json.loads(row["bbox"])
            require(len(row["bbox"]) == 4 and all(type(v) is int for v in row["bbox"]), "Invalid stored crop box.")
            by_key[row["key"]] = row
    all_pairs = read_csv(source/"idrid_overlap_pairs_v1.csv")
    seen, pairs = set(), []
    for pair in all_pairs:
        keys = ("aptos/"+pair["aptos_id"], pair["idrid_key"])
        require(keys not in seen, "Repeated candidate/reference pair.")
        seen.add(keys)
        a, b = (by_key[k] for k in keys)
        for actual, expected in ((pair["aptos_split"], a["usage"]), (pair["aptos_path"], a["relative_path"]),
                                 (pair["aptos_diagnosis"], a["diagnosis"]), (pair["idrid_path"], b["relative_path"]),
                                 (pair["idrid_source_id"], b["source_id"]), (pair["idrid_package"], b["package"]),
                                 (pair["idrid_official_split"], b["official_split"]),
                                 (pair["idrid_dr_grade"], b["diagnosis"]),
                                 (pair["idrid_cohort_decision"], b["cohort_decision"]),
                                 (pair["idrid_candidate"], b["candidate"])):
            require(actual == expected, "Pair/fingerprint metadata mismatch: " + repr(keys))
        exact = a["common_rgb_sha256"] == b["common_rgb_sha256"]
        require(flag(pair["exact_rgb"]) == exact, "Exact-RGB flag mismatch.")
        distances = []
        for av in ("full", "bbox"):
            for bv in ("full", "bbox"):
                valid = flag(a[f"phash_{av}_valid"]) and flag(b[f"phash_{bv}_valid"])
                distance = (int(a[f"phash_{av}"], 16) ^ int(b[f"phash_{bv}"], 16)).bit_count() if valid else 255
                distances.append((distance, av, bv))
        best = min(distances, key=lambda x: x[0])
        require((int(pair["hamming64"]), pair["best_aptos_view"], pair["best_idrid_view"]) == best,
                "Stored candidate/view does not match its fingerprints.")
        kind = "exact_rgb" if exact else ("perceptual_candidate" if best[0] <= config["settings"]["hamming_threshold"] else "nearest_reference")
        require(pair["match_type"] == kind, "Candidate type does not match the screening rule.")
        active = kind != "nearest_reference" and flag(b["candidate"]) and a["usage"] != "excluded_not_used"
        require(flag(pair["affects_external_candidate"]) == active, "Active flag mismatch.")
        if kind != "nearest_reference":
            pairs.append(pair)
    types = Counter(x["match_type"] for x in all_pairs)
    for kind, field in (("exact_rgb", "exact_rgb_pairs"), ("perceptual_candidate", "perceptual_candidate_pairs"),
                        ("nearest_reference", "nearest_reference_rows")):
        require(types[kind] == summary[field], "Summary/pair counts disagree: " + kind)
    active = [p for p in pairs if flag(p["affects_external_candidate"])]
    require(len(active) == summary["flagged_active_pairs"] and
            len({p["idrid_key"] for p in active}) == summary["flagged_active_idrid_files"] and
            dict(Counter(p["aptos_split"] for p in active)) == summary["flagged_active_by_aptos_split"],
            "Summary active scope disagrees with the CSV.")
    pairs.sort(key=lambda p: (not flag(p["affects_external_candidate"]), p["match_type"] != "exact_rgb",
                              int(p["hamming64"]), p["idrid_key"], p["aptos_id"]))
    return summary, pairs, by_key, hashes


def load_font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    raise RuntimeError("No readable Arial/DejaVuSans font found for legible review captions.")


class Views:
    def __init__(self, roots, by_key):
        self.roots, self.by_key = roots, by_key

    @lru_cache(maxsize=16)
    def get(self, key):
        row = self.by_key[key]
        path = cohort.raw_path(self.roots[row["dataset"]], row["relative_path"])
        blob = path.read_bytes()
        require(hashlib.sha256(blob).hexdigest() == row["file_sha256"], "Original changed: " + key)
        with Image.open(io.BytesIO(blob)) as source:
            require(source.size == (int(row["width"]), int(row["height"])), "Source dimensions changed.")
            image = ImageOps.exif_transpose(source).convert("RGB")
        x0, y0, x1, y1 = row["bbox"]
        require(0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height, "Crop exceeds oriented image.")
        result = {"full": image, "bbox": image.crop(row["bbox"])}
        for im in result.values():
            im.thumbnail((768, 600), Image.Resampling.LANCZOS)
        return result


def paste_fit(canvas, image, box):
    x, y, w, h = box
    thumbnail = image.copy()
    thumbnail.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas.paste(thumbnail, (x+(w-thumbnail.width)//2, y+(h-thumbnail.height)//2))


def pair_picture(number, pair, views, font):
    canvas = Image.new("RGB", (1600, 1480), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), f"Pair {number:03d} | Hamming={pair['hamming64']} | active={pair['affects_external_candidate']}", font=font, fill="black")
    draw.text((20, 45), "Candidate only; duplicate status PENDING. Compare vessel branches and shared structures.", font=font, fill="black")
    draw.text((20, 85), f"APTOS {pair['aptos_id']} | {pair['aptos_split']}", font=font, fill="black")
    draw.text((820, 85), f"{pair['idrid_source_id']} | {pair['idrid_package']}/{pair['idrid_official_split']}", font=font, fill="black")
    for index, view in enumerate(("full", "bbox")):
        y = 130+index*665
        draw.text((20, y), "Full oriented source view" if view == "full" else "Stored non-dark bounding-box view (screening only)", font=font, fill="black")
        for column, key in ((0, "aptos/"+pair["aptos_id"]), (1, pair["idrid_key"])):
            paste_fit(canvas, views.get(key)[view], (20+column*800, y+35, 760, 600))
    draw.text((20, 1450), "Display copies preserve aspect ratio; no colour adjustment, warping, model inference or cohort change.", font=font, fill="black")
    return canvas


def sheet_picture(start, page, views, font):
    canvas = Image.new("RGB", (1280, 110+len(page)*440), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), "APTOS / IDRiD - all candidate pairs, review pending", font=font, fill="black")
    draw.text((20, 45), "Screening views; open pairs/pair_NNN.jpg for larger full + crop views.", font=font, fill="black")
    for offset, pair in enumerate(page):
        y = 95+offset*440
        draw.text((20, y), f"Pair {start+offset+1:03d} | Hamming={pair['hamming64']} | active={pair['affects_external_candidate']}", font=font, fill="black")
        for column, key, view in ((0, "aptos/"+pair["aptos_id"], pair["best_aptos_view"]),
                                  (1, pair["idrid_key"], pair["best_idrid_view"])):
            paste_fit(canvas, views.get(key)[view], (20+column*640, y+35, 600, 330))
            label = pair["aptos_id"]+" | "+pair["aptos_split"] if column == 0 else pair["idrid_source_id"]+" | "+pair["idrid_package"]+"/"+pair["idrid_official_split"]
            draw.text((20+column*640, y+370), label, font=font, fill="black")
            draw.text((20+column*640, y+400), "view="+view, font=font, fill="black")
    return canvas


def jpeg_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, subsampling=0)
    return buffer.getvalue()


def export(root, source, out, resume=False):
    roots = {"aptos": root/"train_images", "idrid": root/"IDRiD"/"raw"}
    require(out != root, "Use a separate export directory.")
    protected = (*roots.values(), source, root/"environment"/"data_audit"/"split_v1",
                 root/"environment"/"data_audit"/"idrid_cohort_v1", root/"environment"/"data_audit"/"idrid_audit_v1")
    for path in protected:
        require(not out.is_relative_to(path.resolve()) and not path.resolve().is_relative_to(out), "Export directory overlaps inputs.")
    start, owned, state = time.monotonic(), False, None
    try:
        with Progress() as progress:
            progress.begin("verify completed scan metadata", 1)
            summary, pairs, by_key, hashes = load_inputs(source)
            font = load_font(22)
            config = dict(version=VERSION, script_sha256=audit.sha_file(Path(__file__)),
                          helper_sha256={p.name:audit.sha_file(p) for p in (Path(audit.__file__), Path(cohort.__file__))},
                          source_dir=str(source), input_sha256=hashes, project_root=str(root),
                          pillow=PIL.__version__, font_sha256=audit.sha_file(Path(font.path)),
                          views=["full", "stored_bbox"], pair_size=[1600, 1480], jpeg_quality=95,
                          jpeg_subsampling=0, colour_adjustment=False, pair_order="active first, distance, IDRiD key, APTOS ID")
            run_signature = signature(config)
            state_path = out/"review_export_state_v1.json"
            if resume:
                require(state_path.is_file(), "No export checkpoint exists; start without --resume.")
                state = read_json(state_path)
                require(state["config_sha256"] == run_signature and read_json(out/"review_export_config_v1.json") == config,
                        "Source, script, helpers or render configuration changed; do not mix exports.")
            else:
                require(not out.exists(), f"Export directory exists: {out}. Use --resume.")
                out.mkdir(parents=True)
                for sub in ("pairs", "contact_sheets", "source_evidence"):
                    (out/sub).mkdir()
                state = dict(config_sha256=run_signature, completed={}, status="RUNNING")
                audit.save_json(out/"review_export_config_v1.json", config)
            owned = True
            state["status"] = "RUNNING"
            state["last_attempt_at"] = datetime.now().astimezone().isoformat()
            audit.save_json(state_path, state)
            progress.step()
            selected = sorted({"aptos/"+p["aptos_id"] for p in pairs} | {p["idrid_key"] for p in pairs})
            progress.begin("verify candidate original-file hashes", len(selected))
            raw_stats = {}
            for key in selected:
                row = by_key[key]
                path = cohort.raw_path(roots[row["dataset"]], row["relative_path"])
                before = path.stat()
                require(audit.sha_file(path) == row["file_sha256"], "Original differs from completed scan: " + key)
                after = path.stat()
                require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Original changed during reading.")
                raw_stats[path] = (after.st_size, after.st_mtime_ns)
                progress.step()
            expected_names = {f"pairs/pair_{i+1:03d}.jpg" for i in range(len(pairs))}
            expected_names |= {f"contact_sheets/review_{i//4+1:03d}.jpg" for i in range(0, len(pairs), 4)}
            require(set(state["completed"]) <= expected_names, "Unexpected file in export checkpoint.")
            for name, digest in state["completed"].items():
                require(audit.sha_file(out/name) == digest, "Completed display copy changed: " + name)
            views = Views(roots, by_key)

            def save_picture(name, picture):
                audit.atomic_bytes(out/name, jpeg_bytes(picture))
                state["completed"][name] = audit.sha_file(out/name)
                audit.save_json(state_path, state)

            progress.begin("export candidate pairs and contact sheets", len(pairs))
            for first in range(0, len(pairs), 4):
                page = pairs[first:first+4]
                for offset, pair in enumerate(page):
                    number = first+offset+1
                    name = f"pairs/pair_{number:03d}.jpg"
                    if name not in state["completed"]:
                        save_picture(name, pair_picture(number, pair, views, font))
                    progress.step()
                name = f"contact_sheets/review_{first//4+1:03d}.jpg"
                if name not in state["completed"]:
                    save_picture(name, sheet_picture(first, page, views, font))
            views.get.cache_clear()
            progress.begin("save review template and evidence ZIP", 1)
            review_rows = [dict(pair_number=i+1, **p, evidence_image=f"pairs/pair_{i+1:03d}.jpg",
                               review_status="pending", reviewer="", review_reason="") for i, p in enumerate(pairs)]
            # Immutable template: keep future manual decisions in a separate copy.
            audit.save_csv(out/"idrid_overlap_review_template_v1.csv", review_rows, list(review_rows[0]))
            for name in SOURCE_FILES:
                audit.atomic_bytes(out/"source_evidence"/name, (source/name).read_bytes())
            readme = """APTOS / IDRiD overlap review package

All detected candidate FILE PAIRS are included, including the four inactive pairs.
The nearest_reference-only rows are not candidates and are not rendered.
Start with contact_sheets; pair numbers map to larger pairs/pair_NNN.jpg files.
Each large pair shows both the full oriented images and the stored bounding-box
views. They preserve aspect ratio. These JPEG display copies are not model inputs
or original-resolution evidence. Open the original paths for uncertain details.

No image is declared duplicate or cleared by this export. The CSV template starts
with every review_status=pending. Copy it to idrid_overlap_review_decisions_v1.csv
before editing. Use visually_distinct, suspected_overlap, uncertain, or pending;
record reviewer and structural evidence. Colour, label, black border, optic-disc
side or pHash distance alone must not determine identity. Compare multiple vessel
branches/landmarks within shared visible regions. Patient identity is unavailable.
Do not use model predictions/performance to decide which samples remain.

The source screen remains REVIEW_REQUIRED; the original IDRiD audit retains FAIL.
This package neither changes cohort lists nor authorizes external evaluation.
After review, any cohort action needs its own recorded evidence and decision.

The template is regenerated on resume; a separate decisions CSV is never touched.
The ZIP contains the export evidence and blank template, not later manual edits.
"""
            audit.atomic_bytes(out/"README_review_v1.txt", readme.encode())
            require(all(audit.sha_file(source/name) == digest for name, digest in hashes.items()), "Screen metadata changed during export.")
            require(all((p.stat().st_size, p.stat().st_mtime_ns) == st for p, st in raw_stats.items()), "Candidate originals changed during export.")
            output_names = sorted(state["completed"]) + ["idrid_overlap_review_template_v1.csv", "README_review_v1.txt", "review_export_config_v1.json"]
            output_names += ["source_evidence/"+n for n in SOURCE_FILES]
            result = dict(status="EXPORT_COMPLETE_REVIEW_PENDING", source_screen_status=summary["status"],
                          source_audit_status=summary["source_audit_status"], candidate_pairs=len(pairs),
                          active_pairs=sum(flag(p["affects_external_candidate"]) for p in pairs),
                          unique_aptos_images=len({p["aptos_id"] for p in pairs}),
                          unique_idrid_files=len({p["idrid_key"] for p in pairs}),
                          verified_original_files=len(selected), larger_display_pair_files=len(pairs),
                          contact_sheets=(len(pairs)+3)//4, pending_reviews=len(pairs),
                          model_forwards=0, optimizer_steps=0, cohort_changes=0, raw_files_modified=0,
                          ready_for_external_inference=False, input_sha256=hashes,
                          output_sha256={n:audit.sha_file(out/n) for n in output_names},
                          elapsed_seconds_before_zip=time.monotonic()-start)
            audit.save_json(out/"idrid_overlap_review_summary_v1.json", result)
            output_names.append("idrid_overlap_review_summary_v1.json")
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for name in output_names:
                    archive.write(out/name, name)
            archive_path = out/"idrid_overlap_review_bundle_v1.zip"
            audit.atomic_bytes(archive_path, buffer.getvalue())
            state.update(status="COMPLETE", summary_sha256=audit.sha_file(out/"idrid_overlap_review_summary_v1.json"),
                         zip_sha256=audit.sha_file(archive_path))
            audit.save_json(state_path, state)
            progress.step()
        print(f"\nREVIEW EXPORT COMPLETE | REVIEW PENDING | Pairs: {len(pairs)} | Active: {result['active_pairs']}")
        print(f"Verified candidate originals: {len(selected)} | Full-dataset rescan: 0 | Model forward: 0")
        print(f"Large pair images: {len(pairs)} | Contact sheets: {result['contact_sheets']} | Cohort changes: 0")
        print(f"Elapsed: {audit.clock(time.monotonic()-start)} | ZIP: {archive_path.stat().st_size/1048576:.1f} MiB")
        print(f"Saved: {out}\nReturn idrid_overlap_review_bundle_v1.zip for review.")
        print("After checking this export, remember to commit idrid_overlap_review_v1.py; keep images/ZIPs out of Git.")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nREVIEW EXPORT STOPPED | {type(exc).__name__}: {exc}")
        if owned:
            state.update(status="STOPPED", error=f"{type(exc).__name__}: {exc}")
            try:
                audit.save_json(out/"review_export_state_v1.json", state)
            except OSError as save_error:
                print(f"Could not save stop state: {save_error}")
        print("Resolve the reported issue, then use --resume with the same paths. This exporter does not modify originals or cohort lists.")
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    base = root/"environment"/"data_audit"
    source = (args.source_dir or base/"idrid_overlap_v1"/"overlap_aptos_idrid_v1").resolve()
    out = (args.output_dir or base/"idrid_overlap_review_v1"/"review_aptos_idrid_v1").resolve()
    return export(root, source, out, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
