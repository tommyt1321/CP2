"""Freeze the accepted IDRiD external-evaluation membership policy.

In C:\\CP2: python idrid_final_cohort_v1.py
Repeat the same command after interruption; --resume is also accepted.
Use the supplied review_inputs CSV/JSON unchanged. Requires existing
idrid_audit_v1.py, idrid_cohort_v1.py and idrid_overlap_review_v1.py.
No new dependencies, GPU, image transform, model forward or training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import time

import idrid_audit_v1 as audit
import idrid_cohort_v1 as cohort
import idrid_overlap_review_v1 as review

VERSION = "idrid_final_cohort_v1_20261101"
PROJECT = Path(__file__).resolve().parent
AUDIT_RUN = "run_20261030_010512_d71f2d"
COHORT_RUN = "run_20261030_153346_cef194"
COHORT_SUMMARY_SHA256 = "12252226076dbc20cbf0704aab9b3cac9c1f1da4fed5306afcd7999f9206937c"
REVIEW_CSV = "idrid_overlap_review_decisions_v1.csv"
REVIEW_JSON = "idrid_overlap_visual_review_summary_v1.json"
REVIEW_CSV_SHA256 = "7216b3a932f5bd760b166165d012b3d455ae66675a733dfc9048e10362d21681"
REVIEW_JSON_SHA256 = "16462b240a9f0d93d5605b5ca1d77e87e7c018fbcaba474996d2e0899fe586f9"
UNCERTAIN_PAIRS = (27, 33, 41, 54, 65, 69)
EXCLUDE_KEYS = {"grading/training/IDRiD_" + str(n) for n in (216, 228, 244, 260, 318)}
require = cohort.require

POLICY = {
    "policy_id": "idrid_external_overlap_precaution_v1_20261101",
    "approved_in_chat": "2026-11-01 Asia/Kuala_Lumpur; user instructed proceed after the five-file primary-exclusion/sensitivity proposal",
    "primary": "Exclude unresolved active overlap candidates and their exact-RGB equivalents; uncertainty is not a confirmed duplicate.",
    "sensitivity": "Retain the original 507 curated grading candidates, including the five unresolved files. This is a supplementary cohort, not a second split.",
    "base_rules": "Retain existing exclusions of six DR-conflict records and three extra same-DR exact copies in both classification cohorts.",
    "risk_propagation": "Propagate unresolved active overlap through the recorded common exact-RGB hash across all IDRiD source files, including A/B packages; never join by bare numeric filename.",
    "inference": "Use the same existing validation-selected A0-A3 checkpoints and scoring protocol for primary and sensitivity results. No IDRiD training, tuning, checkpoint selection or result-dependent cohort selection.",
    "segmentation": "Keep available MA/HE/EX/SE lesion masks and OD anatomy separately. Missing masks stay absent_not_inferred; unmatched A/B DR labels remain empty.",
    "limits": "A handled overlap flag is not proof of patient independence or absence of every transformed duplicate. Source audit FAIL and screen REVIEW_REQUIRED are historical evidence and remain unchanged.",
}


class Progress(audit.Progress):
    def render(self):
        with self.lock:
            now = time.monotonic()
            eta = audit.clock((now-self.phase_start)*(self.total-self.done)/self.done) if self.done else "--"
            line = (f"IDRiD final cohort | {self.phase} | Files {self.done}/{self.total} | "
                    f"Elapsed {audit.clock(now-self.start)} | Phase ETA {eta}")
        print("\r" + line.ljust(180), end="", flush=True)


def digest_code(path):
    # Git can change LF/CRLF without changing Python source meaning.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def same_rows(actual, expected, label):
    require(actual == [{k: str(v) for k, v in r.items()} for r in expected],
            "Source-derived rows differ: " + label)


def load_evidence(root, overlap_dir, review_dir):
    """Read verified metadata only. This function never opens image content."""
    base = root / "environment/data_audit"
    audit_dir = base / "idrid_audit_v1" / AUDIT_RUN
    cohort_dir = base / "idrid_cohort_v1" / COHORT_RUN
    source, images, masks, crosswalk, audit_hashes = cohort.read_inputs(audit_dir)
    source_files = {audit_dir / n: h for n, h in audit_hashes.items()}
    cp = cohort_dir / "idrid_cohort_summary_v1.json"
    require(audit.sha_file(cp) == COHORT_SUMMARY_SHA256, "The reviewed candidate-cohort summary changed.")
    cs = review.read_json(cp)
    require(cs["status"] == "MANIFEST_CHECK_PASS" and cs["source_audit_status"] == "FAIL",
            "Unexpected source cohort status.")
    require(cs["input_sha256"] == audit_hashes, "Cohort was not derived from this exact reviewed audit.")
    source_files[cp] = COHORT_SUMMARY_SHA256
    decisions, classes, segments = cohort.build_cohorts(images, masks, crosswalk)
    for name, expected in (("idrid_classification_decisions_v1.csv", decisions),
                           ("idrid_classification_external_v1.csv", classes),
                           ("idrid_segmentation_external_v1.csv", segments)):
        p = cohort_dir / name
        require(audit.sha_file(p) == cs["output_sha256"][name], "Candidate file changed: " + name)
        same_rows(review.read_csv(p), expected, name)
        source_files[p] = cs["output_sha256"][name]

    screen, pairs, fingerprints, screen_hashes = review.load_inputs(overlap_dir)
    source_files.update({overlap_dir / n: h for n, h in screen_hashes.items()})
    screen_config = review.read_json(overlap_dir / "overlap_config_v1.json")
    require(screen_config["input_sha256"] == screen["input_sha256"], "Screen/config source hashes differ.")
    # Translate the recorded Windows root to --project-root, without accepting traversal.
    old_root = PureWindowsPath(screen_config["project_root"])
    for old_path, expected in screen["input_sha256"].items():
        relative = PureWindowsPath(old_path).relative_to(old_root)
        require(relative.parts and ".." not in relative.parts and not relative.is_absolute(), "Unsafe screen source path.")
        p = root.joinpath(*relative.parts).resolve()
        require(p.is_relative_to(root) and p.is_file(), "Missing screen source: " + str(relative))
        require(audit.sha_file(p) == expected, "Screen source changed: " + str(relative))
        source_files[p] = expected
    require(not screen["descriptor_review_keys"], "Descriptor failures still need a separate handling decision.")
    require(len(images) == 597 and len(fingerprints) == 3662+597, "Incomplete fingerprint inventory.")
    for row in images:
        f = fingerprints[cohort.source_key(row)]
        require(f["file_sha256"] == row["file_sha256"] and f["relative_path"] == row["path"] and
                (f["width"], f["height"]) == (row["width"], row["height"]),
                "IDRiD fingerprint/audit identity mismatch: " + row["path"])
    for name, expected in ((REVIEW_CSV, REVIEW_CSV_SHA256), (REVIEW_JSON, REVIEW_JSON_SHA256)):
        p = review_dir / name
        require(p.is_file(), "Missing supplied review input: " + str(p) + ". Extract the starter ZIP into C:\\CP2, preserving folders.")
        require(audit.sha_file(p) == expected, "Review input differs from the accepted original-file follow-up: " + name)
        source_files[p] = expected
    reviewed = review.read_csv(review_dir / REVIEW_CSV)
    rs = review.read_json(review_dir / REVIEW_JSON)
    require(rs["status"] == "ORIGINAL_FOLLOWUP_COMPLETE_WITH_UNCERTAIN_PAIRS" and
            rs["input_source_summary_sha256"] == screen_hashes["idrid_overlap_summary_v1.json"] and
            rs["output_sha256"][REVIEW_CSV] == REVIEW_CSV_SHA256,
            "Review is not linked to this source screen/decision CSV.")
    require(len(reviewed) == len(pairs) == 127, "Expected all 127 reviewed candidate pairs.")
    for number, (r, p) in enumerate(zip(reviewed, pairs, strict=True), 1):
        require(int(r["pair_number"]) == number, "Review pair order changed.")
        require(all(r[k] == v for k, v in p.items()), "Review/source mapping differs at pair " + str(number))
        a, b = fingerprints["aptos/"+r["aptos_id"]], fingerprints[r["idrid_key"]]
        require(r["aptos_original_file_sha256"] == a["file_sha256"] and
                r["idrid_original_file_sha256"] == b["file_sha256"], "Review original hashes differ.")
        require(r["review_status"] in {"visually_distinct", "uncertain"} and r["review_reason"].strip(),
                "Pending/unsupported review decision at pair " + str(number))
    uncertain = [r for r in reviewed if r["review_status"] == "uncertain"]
    require(tuple(int(r["pair_number"]) for r in uncertain) == UNCERTAIN_PAIRS and
            {r["idrid_key"] for r in uncertain} == EXCLUDE_KEYS and
            all(review.flag(r["affects_external_candidate"]) for r in uncertain),
            "Uncertainty scope differs from the accepted five-file policy.")
    require(Counter(r["review_status"] for r in reviewed) == {"visually_distinct": 121, "uncertain": 6},
            "Review counts changed.")
    originals = rs["original_followup"]["originals"]
    require(len(originals) == len({x["key"] for x in originals}) == 9, "Original follow-up inventory differs.")
    for x in originals:
        require(x["sha256"] == x["expected_file_sha256"] == fingerprints[x["key"]]["file_sha256"],
                "Follow-up original does not match screen fingerprint.")
    return dict(source_audit=source, source_screen=screen, source_cohort=cs, images=images, masks=masks,
                crosswalk=crosswalk, classes=classes, segments=segments, base_decisions=decisions,
                reviewed=reviewed, fingerprints=fingerprints, source_files=source_files)


def derive_final(evidence):
    """Apply the accepted rule to identities/hashes, never predictions."""
    fp = evidence["fingerprints"]
    direct = defaultdict(list)
    risks = defaultdict(list)
    for r in evidence["reviewed"]:
        direct[r["idrid_key"]].append(r)
        if r["review_status"] == "uncertain" and review.flag(r["affects_external_candidate"]):
            risks[fp[r["idrid_key"]]["common_rgb_sha256"]].append(r)

    def annotation(key):
        flagged = direct[key]
        risk = risks.get(fp[key]["common_rgb_sha256"], [])
        return {
            "screen_pair_numbers": ";".join(r["pair_number"] for r in flagged),
            "unresolved_pair_numbers": ";".join(r["pair_number"] for r in risk),
            "risk_source_keys": ";".join(sorted({r["idrid_key"] for r in risk})),
            "overlap_handling": ("exclude_uncertain_overlap_precaution" if risk else
                                 "reviewed_visually_distinct" if flagged else "not_flagged_by_screen"),
            "risk_propagated_from_another_file": any(r["idrid_key"] != key for r in risk),
        }

    decisions = []
    for base in evidence["base_decisions"]:
        r = dict(base)
        a = annotation(r["sample_key"])
        candidate = r["decision"] == "keep"
        eligible = candidate and not a["unresolved_pair_numbers"]
        decisions.append({**r, **a, "primary_eligible": eligible, "sensitivity_eligible": candidate,
                          "final_action": "keep_primary_and_sensitivity" if eligible else
                          "sensitivity_only_uncertain_overlap" if candidate else r["decision"]})
    primary_keys = {r["sample_key"] for r in decisions if r["primary_eligible"]}
    primary = [r.copy() for r in evidence["classes"] if r["sample_key"] in primary_keys]
    sensitivity = [r.copy() for r in evidence["classes"]]
    exclusions = [r for r in decisions if r["sensitivity_eligible"] and not r["primary_eligible"]]
    require({r["sample_key"] for r in exclusions} == EXCLUDE_KEYS, "Exact-content propagation changed the agreed exclusion scope; review before proceeding.")
    require(len(primary) == len({r["rgb_sha256"] for r in primary}) == 502 and len(sensitivity) == 507,
            "Classification membership totals differ.")
    require([Counter(r["dr_grade"] for r in primary)[k] for k in range(5)] == [164, 23, 163, 91, 61],
            "Primary grade counts differ.")
    require(Counter(r["official_split"] for r in primary) == {"training": 400, "testing": 102},
            "Primary source partitions differ.")

    seg_decisions, segments = [], []
    for r in evidence["segments"]:
        a = annotation(r["sample_key"])
        # Existing exact A/B links must agree with the common-RGB risk propagation.
        linked = r["grading_sample_key"]
        if linked:
            require(fp[linked]["common_rgb_sha256"] == fp[r["sample_key"]]["common_rgb_sha256"],
                    "Recorded A/B exact link differs from the screen's common pixel hash.")
        eligible = not a["unresolved_pair_numbers"]
        seg_decisions.append({**r, **a, "primary_eligible": eligible,
                              "final_action": "keep_segmentation" if eligible else "exclude_uncertain_overlap_precaution"})
        if eligible:
            segments.append(r.copy())
    require(len(segments) == 81, "New segmentation exclusions need an explicit recorded policy; expected no affected A/B link.")
    require(sum(bool(r["grading_sample_key"]) for r in segments) == 3 and
            sum(r["dr_grade"] == "" for r in segments) == 78, "A/B labels changed.")
    require(sum(r[k+"_availability"] == "absent_not_inferred" for r in segments for k in cohort.LESIONS) == 42,
            "Missing mask states changed.")
    require(len(evidence["masks"]) == 363, "Mask inventory changed.")
    require({r["sample_key"] for r in sensitivity} - primary_keys == EXCLUDE_KEYS,
            "Primary/sensitivity difference is not exactly the five accepted files.")
    return {"idrid_classification_primary_v1.csv": primary,
            "idrid_classification_sensitivity_v1.csv": sensitivity,
            "idrid_classification_decisions_final_v1.csv": decisions,
            "idrid_overlap_precaution_exclusions_v1.csv": exclusions,
            "idrid_segmentation_primary_v1.csv": segments,
            "idrid_segmentation_decisions_final_v1.csv": seg_decisions,
            "idrid_mask_catalog_v1.csv": evidence["masks"],
            "idrid_package_crosswalk_v1.csv": evidence["crosswalk"]}


def verify_candidate_aptos(root, evidence, progress):
    keys = sorted({"aptos/"+r["aptos_id"] for r in evidence["reviewed"]})
    require(len(keys) == 77, "Candidate APTOS source count changed.")
    progress.begin("verify candidate APTOS original hashes", len(keys))
    for key in keys:
        r = evidence["fingerprints"][key]
        p = cohort.raw_path(root / "train_images", r["relative_path"])
        before = p.stat()
        digest = audit.sha_file(p)
        after = p.stat()
        require(digest == r["file_sha256"] and
                (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "Candidate APTOS original changed: " + key)
        progress.step()
    return len(keys)


def check_sources(source_files):
    for p, expected in source_files.items():
        require(audit.sha_file(p) == expected, "Source evidence changed during finalization: " + str(p))


def protect_output(output, input_dirs):
    require(not output.is_symlink(), "Output cannot be a symlink.")
    output = output.resolve()
    for p in input_dirs:
        p = p.resolve()
        require(not output.is_relative_to(p) and not p.is_relative_to(output),
                "Output overlaps a protected input folder: " + str(p))


def publish(output, evidence, tables, config, progress, elapsed):
    """Same folder on rerun; COMPLETE is written last. No interrupt-file series."""
    sig = config["signature"]
    state_path = output / "final_cohort_state_v1.json"
    summary_path = output / "idrid_final_cohort_summary_v1.json"
    if output.exists() and any(output.iterdir()):
        require(state_path.is_file(), "Nonempty output folder has no owned state; choose a new --output-dir.")
        state = review.read_json(state_path)
        require(state.get("signature") == sig, "Existing output belongs to different inputs/code/policy; it will not be overwritten.")
        if state.get("status") == "COMPLETE":
            require(audit.sha_file(summary_path) == state["summary_sha256"], "Completed summary changed.")
            summary = review.read_json(summary_path)
            for name, expected in summary["output_sha256"].items():
                require(Path(name).name == name and audit.sha_file(output/name) == expected,
                        "Completed output changed: " + name)
            require(summary["signature"] == sig, "Completed signature differs.")
            return summary, True
        require(state.get("status") == "BUILDING", "Unknown output state.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        audit.save_json(state_path, {"status": "BUILDING", "signature": sig, "script_version": VERSION})
    hashes = {}
    progress.begin("save frozen lists and policy", len(tables)+5)
    for name, rows in tables.items():
        require(rows, "Unexpected empty final table: " + name)
        audit.save_csv(output/name, rows, list(rows[0]))
        same_rows(review.read_csv(output/name), rows, name + " readback")
        hashes[name] = audit.sha_file(output/name)
        progress.step()
    audit.save_json(output / "idrid_final_cohort_config_v1.json", config)
    progress.step()
    audit.save_json(output / "idrid_external_cohort_policy_v1.json", POLICY)
    progress.step()
    audit.save_json(output / "method_sources_v1.json", [
        {"title": "CP2 accepted overlap handling, 2026-11-01", "role": "Project-specific five-file precaution and same-model full-cohort sensitivity analysis; not an official IDRiD requirement."},
        {"title": "Original IDRiD cohort method sources", "role": "Inherited source metadata; same exact-duplicate/DR-conflict rules retained.", "sources": cohort.SOURCES},
        {"title": "Accepted original-file overlap review", "role": "AI-assisted review with unresolved source blur; no confirmed duplicate or patient-identity claim.",
         "review_csv_sha256": REVIEW_CSV_SHA256, "review_summary_sha256": REVIEW_JSON_SHA256},
    ])
    progress.step()
    for name in ("idrid_final_cohort_config_v1.json", "idrid_external_cohort_policy_v1.json", "method_sources_v1.json"):
        hashes[name] = audit.sha_file(output/name)
    check_sources(evidence["source_files"])
    summary = {
        "status": "FINAL_COHORT_CHECK_PASS", "script_version": VERSION, "signature": sig,
        "created_at": datetime.now(timezone.utc).isoformat(), "saved": str(output),
        "scope": "Membership and source hashes frozen; image/mask preprocessing and external inference have not started.",
        "classification_original_records": 516, "base_curated_classification": 507,
        "primary_classification": 502, "sensitivity_classification": 507,
        "precaution_excluded_files": 5, "precaution_excluded_sample_keys": sorted(EXCLUDE_KEYS),
        "primary_dr_counts_0_to_4": [164, 23, 163, 91, 61],
        "sensitivity_dr_counts_0_to_4": [167, 23, 165, 91, 61],
        "primary_official_source_partitions": {"training": 400, "testing": 102},
        "sensitivity_official_source_partitions": {"training": 405, "testing": 102},
        "inherited_classification_exclusions": {"DR_conflict_records": 6, "extra_same_DR_exact_copies": 3},
        "segmentation_primary": 81, "segmentation_source_partitions": {"training": 54, "testing": 27},
        "segmentation_exclusions_from_overlap": 0, "available_masks": 363,
        "available_mask_types": dict(Counter(r["lesion"] for r in evidence["masks"])),
        "absent_mask_type_entries_not_imputed": 42, "exact_AB_grade_links": 3, "unresolved_AB_grade_links": 78,
        "reviewed_pairs": 127, "review_counts": {"visually_distinct": 121, "uncertain": 6},
        "unresolved_pair_numbers": list(UNCERTAIN_PAIRS), "confirmed_overlap_pairs": 0,
        "overlap_handling": "PRECAUTION_POLICY_APPLIED_TO_PRIMARY_MEMBERSHIP; uncertainty is retained in review records",
        "source_audit_status_unchanged": "FAIL", "source_screen_status_unchanged": "REVIEW_REQUIRED",
        "verified_IDRiD_original_image_and_mask_hashes": 960, "verified_candidate_APTOS_original_hashes": 77,
        "all_3662_APTOS_originals_rehashed": False, "full_pair_rescans": 0,
        "input_metadata_unchanged": True, "raw_files_modified": 0, "old_candidate_lists_modified": 0,
        "new_AB_grade_links": 0, "model_forwards": 0, "optimizer_steps": 0, "preprocessed_images_created": 0,
        "ready_for_aligned_preprocessing": True, "ready_for_external_inference": False,
        "next_step": "Apply the locked 384x384 image transform and matching mask geometry, preserve missing masks, inspect alignment; freeze inference/localisation settings before external predictions.",
        "limitations": [POLICY["limits"], "The 502/507 cohorts are project-specific curated external sets, not the unmodified 516-image or 103-image official benchmark.",
                        "For 78 segmentation images no DR grade was established; these are not silently included in classification scoring."],
        "output_sha256": hashes, "elapsed_seconds": elapsed(),
    }
    audit.save_json(summary_path, summary)
    progress.step()
    audit.save_json(state_path, {"status": "COMPLETE", "signature": sig, "script_version": VERSION,
                                 "summary_sha256": audit.sha_file(summary_path)})
    progress.step()
    return summary, False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--overlap-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Same safe rerun behavior as the default command.")
    args = parser.parse_args()
    started = time.monotonic()
    root = args.project_root.resolve()
    base = root / "environment/data_audit"
    review_dir = (args.review_dir or base / "idrid_final_cohort_v1/review_inputs").resolve()
    overlap_dir = (args.overlap_dir or base / "idrid_overlap_v1/overlap_aptos_idrid_v1").resolve()
    output = args.output_dir or base / "idrid_final_cohort_v1/final_aptos_idrid_v1"
    print("Preparing the accepted IDRiD primary and sensitivity cohorts...", flush=True)
    try:
        protect_output(output, [root / "IDRiD/raw", root / "train_images", root / "processed_images_384_v1",
                                base / "idrid_audit_v1", base / "idrid_cohort_v1", base / "split_v1",
                                base / "training_v1", base / "evaluation_v1", review_dir, overlap_dir])
        output = output.resolve()
        with Progress() as progress:
            progress.begin("verify source and accepted review metadata", 1)
            evidence = load_evidence(root, overlap_dir, review_dir)
            tables = derive_final(evidence)
            progress.step()
            verified = cohort.verify_raw_files(root / "IDRiD/raw", evidence["images"], evidence["masks"], progress)
            aptos_verified = verify_candidate_aptos(root, evidence, progress)
            require((verified, aptos_verified) == (960, 77), "Original hash verification is incomplete.")
            code = {Path(m.__file__).name: digest_code(Path(m.__file__)) for m in (audit, cohort, review)}
            code[Path(__file__).name] = digest_code(Path(__file__))
            source_hashes = {str(p): h for p, h in sorted(evidence["source_files"].items(), key=lambda x: str(x[0]))}
            signature = review.signature({"version": VERSION, "policy": POLICY, "source_sha256": source_hashes, "code_normalized_sha256": code})
            config = {"script_version": VERSION, "signature": signature, "project_root": str(root),
                      "raw_root": str(root / "IDRiD/raw"), "review_dir": str(review_dir), "overlap_dir": str(overlap_dir),
                      "source_sha256": source_hashes, "code_LF_normalized_sha256": code,
                      "policy": POLICY, "cohort_counts": {n: len(r) for n, r in tables.items()},
                      "checkpoint_loading": "none; use existing validation-selected best checkpoints at later inference stage"}
            summary, reused = publish(output, evidence, tables, config, progress, lambda: time.monotonic()-started)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nFINAL COHORT STOPPED | {type(exc).__name__}: {exc}")
        print("No new complete run is accepted. Existing source evidence and raw files are retained.")
        print("Resolve the reported issue, then repeat: python idrid_final_cohort_v1.py")
        return 1
    print("\nFINAL COHORT CHECK: PASS (membership and hashes only)")
    if reused:
        print("Existing COMPLETE output verified and reused; no extra run folder created.")
    print("Primary DR: 502 | Sensitivity DR: 507 | Uncertain-overlap precaution: 5 unique IDRiD files")
    print("Primary DR counts [0,1,2,3,4]: [164, 23, 163, 91, 61]")
    print("Primary official source partitions: 400 training / 102 testing; all are external to CP2 training.")
    print("Segmentation: 81 | Available masks: 363 | Absent entries: 42 (not imputed) | New A/B links: 0")
    print("Review: 121 visually_distinct / 6 uncertain | Confirmed overlap: 0")
    print("Source audit remains FAIL; source overlap remains REVIEW_REQUIRED; precautionary policy recorded separately.")
    print("Original hashes verified: IDRiD 960 / candidate APTOS 77 | Full pair rescans: 0")
    print("Preprocessing: 0 | Model forward: 0 | Optimizer steps: 0 | Original/source files unchanged.")
    print(f"Elapsed: {audit.clock(time.monotonic()-started)}\nSaved: {output}")
    print("Return idrid_final_cohort_summary_v1.json and idrid_overlap_precaution_exclusions_v1.csv.")
    print("Next: aligned RGB/mask preprocessing before frozen-model external evaluation.")
    print("After this result passes review: remember to commit idrid_final_cohort_v1.py; keep raw data out of Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
