"""Prepare auditable IDRiD external-evaluation candidates from the reviewed r3 run.

Run in C:\\CP2: python idrid_cohort_v1.py
Requires the existing idrid_audit_v1.py. No extra packages or GPU are needed.
All official grading partitions remain external to CP2 model development.
No new split, training, inference, image transformation or raw-file deletion.
The source audit remains FAIL; its known DR conflicts are resolved only in the
derived classification candidate list, with every excluded record retained.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import time
from uuid import uuid4

import idrid_audit_v1 as audit

VERSION = "idrid_cohort_v1_20261030"
PROJECT = Path(__file__).resolve().parent
REVIEWED_RUN = "run_20261030_010512_d71f2d"
REVIEWED_SUMMARY_SHA256 = "dc81d5903cc8913ae4eeeda5a8a172ca47200b421d73fd49bdd8bd667fe40308"
INPUT_FILES = ("idrid_image_manifest_v1.csv", "idrid_mask_manifest_v1.csv",
               "idrid_package_crosswalk_v1.csv")
LESIONS = ("MA", "HE", "EX", "SE", "OD")
CLASS_FIELDS = "sample_key official_split canonical_id source_id image_path width height dr_grade dme_grade file_sha256 rgb_sha256"
DECISION_FIELDS = CLASS_FIELDS + " decision reason representative_key duplicate_group_size conflicting_dr_labels conflicting_dme_labels"
SOURCES = [
    {"title": "IDRiD official data description", "url": "https://idrid.grand-challenge.org/Data/",
     "role": "516 grading images, 81 annotated images, available mask counts; not the source of our exclusion policy"},
    {"title": "Porwal et al. (2018), Indian Diabetic Retinopathy Image Dataset (IDRiD)",
     "doi": "10.3390/data3030025", "role": "Dataset description and grading/lesion annotation roles"},
    {"title": "CP1 proposal, sections 3.3.3 and 3.10.3", "role": "External evaluation only, selected A0-A3 checkpoints, no IDRiD tuning or fine-tuning"},
    {"title": "CP2 project data-quality choice", "role": "Exclude whole exact-RGB groups with conflicting DR grades; retain one deterministic representative from groups with the same DR grade. This is a project decision, not an official IDRiD requirement."},
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_key(row):
    return f"{row['package']}/{row['official_split']}/{row['canonical_id']}"


def numeric_id(row):
    return int(row["canonical_id"].split("_")[-1])


def stable_order(row):
    return (0 if row["official_split"] == "training" else 1,
            numeric_id(row), row["path"])


def representative_order(row):
    # Deterministic, independent of predictions. In the reviewed data all
    # same-DR duplicate groups occur within the official training partition.
    return (numeric_id(row), 0 if row["official_split"] == "testing" else 1, row["path"])


def validate_metadata(summary, images, masks, crosswalk):
    require(summary["script_version"] == "idrid_audit_v1_20261030_r3", "Expected the reviewed r3 audit.")
    require(summary["stopped_reason"] is None and summary["input_files_unchanged"] is True,
            "Source audit stopped or did not verify unchanged inputs.")
    require(summary["decoded_masks"] == 363 and summary["failed_mask_files"] == 0,
            "Mask decoding must be complete before candidate-list generation.")
    errors = [x for x in summary["issues"] if x["severity"] == "ERROR"]
    require(len(errors) == 3 and all(x["code"] == "WITHIN_PACKAGE_EXACT_DUPLICATES" for x in errors),
            "Unexpected audit errors; do not bypass them.")
    require(len(images) == 597 and len({source_key(x) for x in images}) == 597,
            "Original image identities are incomplete or ambiguous.")
    require(len({x["path"] for x in images}) == 597, "Duplicate original image paths.")
    by_key = {source_key(x): x for x in images}
    require(Counter((x["package"], x["official_split"]) for x in images) ==
            {("grading", "training"): 413, ("grading", "testing"): 103,
             ("segmentation", "training"): 54, ("segmentation", "testing"): 27}, "Image partition counts changed.")
    for x in images:
        require((int(x["width"]), int(x["height"])) == (4288, 2848), "Unexpected image dimensions.")
        if x["package"] == "grading":
            require(x["label_status"] == "matched" and x["dr_grade"] in {"0", "1", "2", "3", "4"},
                    f"Invalid or unmatched DR label: {source_key(x)}")
            require(x["dme_grade"] in {"0", "1", "2"}, "Invalid DME metadata.")
    require(len(masks) == len({(x["official_split"], x["canonical_id"], x["lesion"]) for x in masks}) == 363,
            "Incomplete or duplicate mask identities.")
    for m in masks:
        owner = by_key.get(f"segmentation/{m['official_split']}/{m['canonical_id']}")
        require(owner is not None and (m["width"], m["height"]) == (owner["width"], owner["height"]),
                f"Mask owner/size mismatch: {m['path']}")
        require(m["lesion"] in LESIONS and m["role"] == ("anatomy" if m["lesion"] == "OD" else "lesion"),
                "Mask role/type mismatch.")
        require(m["alignment_status"] == "id_split_dimensions_match_not_visual_validation" and m["exif_orientation"] == "1",
                f"Unresolved mask alignment metadata: {m['path']}")
    require(len(crosswalk) == 81 and len({x["segmentation_path"] for x in crosswalk}) == 81,
            "Incomplete A/B crosswalk.")
    require(Counter(x["match_status"] for x in crosswalk) == {"unique_exact_rgb": 3, "no_exact_rgb_match": 78},
            "A/B crosswalk differs from the reviewed audit.")


def read_inputs(run):
    summary_path = run / "idrid_audit_summary_v1.json"
    require(audit.sha_file(summary_path) == REVIEWED_SUMMARY_SHA256,
            "This is not the reviewed audit summary. Use run_20261030_010512_d71f2d; do not silently substitute another run.")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    rows, hashes = {}, {summary_path.name: REVIEWED_SUMMARY_SHA256}
    for name in INPUT_FILES:
        path = run / name
        actual = audit.sha_file(path)
        require(actual == summary["output_sha256"][name], f"Audit CSV checksum mismatch: {name}")
        hashes[name] = actual
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows[name] = list(csv.DictReader(handle))
    images, masks, crosswalk = (rows[name] for name in INPUT_FILES)
    validate_metadata(summary, images, masks, crosswalk)
    return summary, images, masks, crosswalk, hashes


def class_record(x):
    return dict(sample_key=source_key(x), official_split=x["official_split"], canonical_id=x["canonical_id"],
                source_id=x["source_id"], image_path=x["path"], width=x["width"], height=x["height"],
                dr_grade=int(x["dr_grade"]), dme_grade=int(x["dme_grade"]),
                file_sha256=x["file_sha256"], rgb_sha256=x["rgb_sha256"])


def build_cohorts(images, masks, crosswalk):
    grading = sorted((x for x in images if x["package"] == "grading"), key=stable_order)
    groups = defaultdict(list)
    for row in grading:
        groups[row["rgb_sha256"]].append(row)
    decisions = []
    for row in grading:
        group = groups[row["rgb_sha256"]]
        conflict = len({x["dr_grade"] for x in group}) > 1
        representative = min(group, key=representative_order)
        if conflict:
            decision, reason = "exclude_dr_conflict", "Exact RGB group has conflicting DR labels; exclude every member without guessing a grade."
        elif row["path"] != representative["path"]:
            decision, reason = "exclude_same_dr_duplicate", "Additional exact RGB copy with the same DR grade."
        else:
            decision, reason = "keep", "Unique RGB image." if len(group) == 1 else "Deterministic representative of an exact RGB group with one DR grade."
        decisions.append({**class_record(row), "decision": decision, "reason": reason,
                          "representative_key": "" if conflict else source_key(representative),
                          "duplicate_group_size": len(group), "conflicting_dr_labels": conflict,
                          "conflicting_dme_labels": len({x["dme_grade"] for x in group}) > 1})
    fields = CLASS_FIELDS.split()
    classification = [{k: x[k] for k in fields} for x in decisions if x["decision"] == "keep"]
    require(Counter(x["decision"] for x in decisions) ==
            {"keep": 507, "exclude_dr_conflict": 6, "exclude_same_dr_duplicate": 3}, "Unexpected decision counts.")
    require(len({x["rgb_sha256"] for x in classification}) == 507, "Classification candidates still contain exact duplicates.")
    require(Counter(x["dr_grade"] for x in classification) == {0: 167, 1: 23, 2: 165, 3: 91, 4: 61},
            "Unexpected candidate DR distribution.")
    masks_by_key = {(x["official_split"], x["canonical_id"], x["lesion"]): x for x in masks}
    links = {x["segmentation_path"]: x for x in crosswalk}
    b_by_path = {x["path"]: x for x in grading}
    segmentation = []
    for x in sorted((x for x in images if x["package"] == "segmentation"), key=stable_order):
        link = links[x["path"]]
        match = link["match_status"] == "unique_exact_rgb"
        if match:
            target = b_by_path.get(link["grading_path"])
            require(target is not None and target["rgb_sha256"] == x["rgb_sha256"] and
                    target["dr_grade"] == link["dr_grade"], "A/B crosswalk label/content mismatch.")
        else:
            require(not link["grading_path"] and not link["dr_grade"], "Unmatched A image was assigned a B label.")
        record = dict(sample_key=source_key(x), official_split=x["official_split"], canonical_id=x["canonical_id"],
                      source_id=x["source_id"], image_path=x["path"], width=x["width"], height=x["height"],
                      file_sha256=x["file_sha256"], rgb_sha256=x["rgb_sha256"],
                      grading_link_status=link["match_status"],
                      grading_sample_key=source_key(target) if match else "", dr_grade=link["dr_grade"] if match else "")
        for lesion in LESIONS:
            m = masks_by_key.get((x["official_split"], x["canonical_id"], lesion))
            record[f"{lesion}_availability"] = "present" if m else "absent_not_inferred"
            record[f"{lesion}_mask_path"] = m["path"] if m else ""
        segmentation.append(record)
    require(len(segmentation) == 81 and len({x["rgb_sha256"] for x in segmentation}) == 81,
            "Segmentation image identities changed.")
    require(sum(x[f"{k}_availability"] == "absent_not_inferred" for x in segmentation for k in LESIONS) == 42,
            "Missing mask states changed.")
    return decisions, classification, segmentation


def raw_path(raw, relative):
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and ".." not in rel.parts and "\\" not in relative and ":" not in relative,
            f"Unsafe raw relative path: {relative}")
    p = raw.joinpath(*rel.parts)
    require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(raw.resolve()),
            f"Raw file missing or outside raw root: {relative}")
    return p


def verify_raw_files(raw, images, masks, progress):
    rows = [(x, "image") for x in images] + [(x, "mask") for x in masks]
    require(len({x["path"] for x, _ in rows}) == len(rows), "Repeated raw paths across image/mask inputs.")
    progress.begin("verify original image/mask hashes", len(rows))
    for row, kind in rows:
        p = raw_path(raw, row["path"])
        before = p.stat()
        digest = audit.sha_file(p)
        after = p.stat()
        require(digest == row["file_sha256"] and
                (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                f"Raw file changed since audit: {row['path']}")
        progress.step(kind)
    return len(rows)


class Progress(audit.Progress):
    def render(self):
        with self.lock:
            now = time.monotonic()
            eta = audit.clock((now - self.phase_start) * (self.total - self.done) / self.done) if self.done else "--"
            line = (f"IDRiD cohorts | {self.phase} | Files {self.done}/{self.total} | "
                    f"Elapsed {audit.clock(now-self.start)} | Phase ETA {eta}")
        print("\r" + line.ljust(180), end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit-run", type=Path, default=PROJECT / "environment" / "data_audit" / "idrid_audit_v1" / REVIEWED_RUN)
    parser.add_argument("--raw-root", type=Path, default=PROJECT / "IDRiD" / "raw")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "environment" / "data_audit" / "idrid_cohort_v1")
    args = parser.parse_args()
    run, raw, output_root = args.audit_run.resolve(), args.raw_root.resolve(), args.output_root.resolve()
    if output_root == raw or output_root.is_relative_to(raw) or output_root == run or output_root.is_relative_to(run):
        parser.error("Output must be separate from raw data and the source audit run.")
    out = output_root / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6])
    started = time.monotonic()
    print("Preparing IDRiD external-evaluation candidate lists...", flush=True)
    try:
        with Progress() as progress:
            progress.begin("verify reviewed audit metadata", 1)
            source_summary, images, masks, crosswalk, input_hashes = read_inputs(run)
            decisions, classification, segmentation = build_cohorts(images, masks, crosswalk)
            progress.step()
            verified = verify_raw_files(raw, images, masks, progress)
            out.mkdir(parents=True, exist_ok=False)
            progress.begin("save candidate lists and decisions", 7)
            outputs = [
                ("idrid_classification_decisions_v1.csv", decisions, DECISION_FIELDS.split()),
                ("idrid_classification_external_v1.csv", classification, CLASS_FIELDS.split()),
                ("idrid_segmentation_external_v1.csv", segmentation, list(segmentation[0])),
                ("idrid_mask_catalog_v1.csv", masks, list(masks[0])),
            ]
            output_hashes = {}
            for name, rows, columns in outputs:
                audit.save_csv(out / name, rows, columns)
                output_hashes[name] = audit.sha_file(out / name)
                progress.step()
            config = dict(script_version=VERSION, script_sha256=audit.sha_file(Path(__file__)),
                          created_at=datetime.now().astimezone().isoformat(), source_audit=str(run), raw_root=str(raw),
                          reviewed_summary_sha256=REVIEWED_SUMMARY_SHA256, input_sha256=input_hashes,
                          identity_key="package / official_split / canonical_id",
                          classification_scope="Candidates from both official grading partitions; all remain external to model development. No new 70/15/15 split.",
                          dr_conflict_rule="Exclude every member of any exact RGB group with conflicting DR grades; no relabelling.",
                          same_dr_duplicate_rule="Keep minimum numeric ID; tie: testing before training, then relative path. Prediction-independent.",
                          dme_rule="DME is metadata only; DR decisions must not be presented as DME curation or evaluation.",
                          segmentation_scope="All 81 annotated image records with available masks. OD is anatomy, not lesion; missing masks remain unknown.",
                          crosswalk_rule="Only verified exact RGB links carry a DR label; 78 unresolved entries stay unlabelled.",
                          no_files_deleted=True, no_labels_modified=True, no_training_or_inference=True,
                          audit_status_policy="Original audit FAIL is retained. PASS below applies only to derived lists and checks, not full external-study readiness.")
            audit.save_json(out / "idrid_cohort_config_v1.json", config)
            progress.step()
            audit.save_json(out / "method_sources_v1.json", SOURCES)
            progress.step()
            for name in ("idrid_cohort_config_v1.json", "method_sources_v1.json"):
                output_hashes[name] = audit.sha_file(out / name)
            require(all(audit.sha_file(run / name) == digest for name, digest in input_hashes.items()),
                    "Input audit metadata changed during this run.")
            summary = dict(status="MANIFEST_CHECK_PASS", script_version=VERSION, saved=str(out),
                           scope="Derived candidate lists, decisions and raw-file hashes only; no external performance measured.",
                           source_audit_status=source_summary["status"], source_audit=str(run),
                           classification_original_records=len(decisions), classification_candidates=len(classification),
                           decisions=dict(Counter(x["decision"] for x in decisions)),
                           classification_dr_counts={str(k): n for k, n in sorted(Counter(x["dr_grade"] for x in classification).items())},
                           classification_source_partition_counts=dict(Counter(x["official_split"] for x in classification)),
                           segmentation_candidates=len(segmentation), available_masks=len(masks),
                           available_mask_counts=dict(Counter(x["lesion"] for x in masks)),
                           masks_not_provided=42, segmentation_images_with_exact_grade_link=3, unresolved_grade_links=78,
                           verified_original_image_and_mask_files=verified, source_audit_files_unchanged=True,
                           excluded_sample_keys=[x["sample_key"] for x in decisions if x["decision"] != "keep"],
                           input_sha256=input_hashes, output_sha256=output_hashes,
                           elapsed_seconds=time.monotonic()-started, model_forwards=0, optimizer_steps=0,
                           raw_files_modified=0, preprocessed_images_created=0, ready_for_inference=False,
                           remaining_steps=["Review the derived counts and decisions before using the candidates.",
                                            "Check APTOS-IDRiD overlap; exact IDRiD deduplication does not prove patient independence.",
                                            "Apply the fixed image transform with aligned masks and inspect registration.",
                                            "Fix external scoring/localisation settings before seeing external predictions; do not tune A0-A3 on IDRiD."],
                           limits=["The 507 candidates are a project-specific curated set, not the unmodified official 516 or 103-image benchmark.",
                                   "Same patient, recompression and near-duplicate relationships are not resolved by exact RGB hashes.",
                                   "Available mask metadata and dimensions do not establish anatomical annotation correctness."])
            audit.save_json(out / "idrid_cohort_summary_v1.json", summary)
            progress.step()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nCOHORT PREPARATION STOPPED | {type(exc).__name__}: {exc}")
        print(f"No completed manifest run is accepted. Partial output, if any: {out}")
        return 1
    print("\nCOHORT MANIFEST CHECK: PASS (candidate lists and hashes only)")
    print("DR classification: 516 records -> 507 candidates | DR conflict exclusions: 6 | Extra same-DR copies: 3")
    print(f"DR counts [0,1,2,3,4]: {[summary['classification_dr_counts'][str(k)] for k in range(5)]}")
    print("Official source partitions retained: 405 training / 102 testing; ALL candidates are external to CP2 training.")
    print("Segmentation directory: 81 images | Available masks: 363 | Absent mask/type entries: 42 (not imputed)")
    print("Exact A/B grade links: 3 | Unresolved: 78 | Raw source audit remains FAIL with its original findings.")
    print(f"Verified original file hashes: {verified} | Model forward: 0 | Optimizer steps: 0 | Raw files modified: 0")
    print(f"Elapsed: {audit.clock(time.monotonic()-started)}\nSaved: {out}")
    print("Return idrid_cohort_summary_v1.json and idrid_classification_decisions_v1.csv.")
    print("Next: check dataset overlap and prepare aligned preprocessing before external evaluation.")
    print("After review: remember to commit idrid_cohort_v1.py; keep raw data out of Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
