"""CP2: paired frozen A3 comparison using the reviewed, saved APTOS v3 masks.

In Git Bash, from /c/CP2 with its existing .venv active:
    python -B ~/Downloads/aptos_fov_a3_compare_v3.py compare

Optional: use smoke instead of compare for the six technical checks only.
No installation, training, threshold fitting, or modification of previous runs.
"""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from uuid import uuid4
import zipfile

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image

AUDIT = "53a7afb616a3b46a25bd065a21a160cb80602913e1ce097df7bcf86d806f519d"
PROTOCOL = "a7343fb265c7f037e597ab003a560a75a3d76209d078313f977ac1931a03e63a"
GEOMETRY_SUMMARY = "b134416bf959b7d5dcacd005bf195d1bd9f1eceecd7e2a7c39c439140e6f3ee4"
FOV_CODE = {
    "run_fov.py": "b0dd632de9fd678c671d656841f55d2a8b5dd0d5d3865a606976c32bded91ba3",
    "fov_regions.py": "11e130e464f5259a309a6af3ff40b8879bf68c2b4f5e82c833636f9f35604672",
}
ORIGINAL, CLIP, RESEG = ARMS = ("original_A3", "original_regions_clipped", "FOV_resegmented")
METRICS = ("qwk", "macro_f1", "accuracy")
BROAD = ("82910bba4753", "dc0eea0b68a7")
# Geometry controls, fixed before running this v3 classification comparison.
# The last ID is the smallest SHA256("v3-gate|"+id) among the other 442.
GATE = ("3461dc601cc2", "4704dbb59536", "26453eb7e989", *BROAD, "983c98354e9c")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return sha(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8"))


def read_geometry(root):
    """Match all the contents that identify the inspected run, never just a timestamp."""
    runs = root / "experiments/aptos_fov_geometry_v3/runs"
    check(runs.is_dir(), "Missing v3 geometry runs. Start in C:/CP2 with the completed v3 run present.")
    for path in sorted((p for p in runs.iterdir() if p.is_dir()), reverse=True):
        try:
            raw = {n: (path/n).read_bytes() for n in ("protocol.json", "summary.json", "per_image.csv")}
            if canonical(json.loads(raw["summary.json"])) != GEOMETRY_SUMMARY:
                continue
            if canonical(json.loads(raw["protocol.json"])) != PROTOCOL:
                continue
            rows = sorted(csv.DictReader(io.StringIO(raw["per_image.csv"].decode("utf-8-sig"))),
                          key=lambda row: row["id_code"])
            if canonical(rows) != AUDIT:
                continue
        except (KeyError, FileNotFoundError, UnicodeError, ValueError, csv.Error):
            continue
        return path, {r["id_code"]: r for r in rows}, {path/n: sha(v) for n, v in raw.items()}
    raise ValueError("No saved run matches the reviewed v3 geometry ZIP. No v1/v2 mask fallback.")


def read_mask(data, row):
    with Image.open(io.BytesIO(data)) as im:
        check((im.format, im.mode, im.size) == ("PNG", "L", (384, 384)),
              row["id_code"] + ": wrong mask format")
        pixels = np.array(im)
    m = pixels == 255
    check(np.all((pixels == 0) | m) and int(m.sum()) == int(row["v3_pixels"]) and
          sha(m.tobytes()) == row["v3_raw_sha256"], row["id_code"] + ": v3 mask content differs")
    return m


def collect_masks(path, rows, protected):
    folder = path/"candidate_masks"
    check(folder.is_dir() and {p.stem for p in folder.glob("*.png")} == set(rows),
          "The 527 original v3 PNG masks must be present under the geometry run's candidate_masks/.")
    cache = {}
    for k, r in rows.items():
        p = folder/(k + ".png")
        data = p.read_bytes()
        cache[k] = np.packbits(read_mask(data, r), axis=None)
        protected[p] = sha(data)
    return cache


def one_image(sample, mask, geometry, reference, model, device, base, fov, control):
    import torch
    import boundary_map_v1 as bm
    import superpixels_v1 as sp
    k, y = sample["id_code"], int(sample["label"])
    check(sample["split"] == "validation" and geometry["id_code"] == k, k + ": input ID/split mismatch")
    details = bm.boundary_map(sample["rgb"], return_details=True)
    full = sp.segment_superpixels(details["lab"], details["boundary"][0], boundary_weight=2.)["labels"]
    check(full.shape == mask.shape and np.all(full >= 0), k + ": invalid original geometry")
    cnn = sample["cnn"][None, ...].to(device)
    original = model(cnn, [full], segmentation_method="boundary_slic", return_attention=True)
    original_measures, sel = base.selection(original, full, mask)
    check(reference == (y, original_measures["pred"]), k + ": archived label or original A3 grade did not replay")
    projected = model.regional.extract(cnn)
    if control:
        same = base.masked_forward(model, projected, full)
        torch.testing.assert_close(original["logits"], same["logits"], atol=2e-5, rtol=2e-5)
        a, b = original["vit_attentions"], same["vit_attentions"]
        check(len(a) == len(b) and len(a) > 0, k + ": unexpected attention layer count")
        for t, u in zip(a, b):
            torch.testing.assert_close(t, u, atol=2e-5, rtol=2e-5)
        del same
    del original
    clipped = fov.connected_labels(np.where(mask, full, -1), 1)
    reseg, recovered = fov.masked_slic(sample["rgb"], mask, n_segments=256)
    row = {"id_code": k, "y_true": y, "archived_A3_pred": int(reference[1]),
           "geometry_group": geometry["group"], "v3_action": geometry["action"],
           "v3_reason": geometry["reason"], "known_retained_broad_corners": int(k in BROAD),
           "v3_pixels": int(mask.sum()), "v3_raw_sha256": geometry["v3_raw_sha256"],
           "reseg_recovery_pixels_summed_over_10_iterations": int(recovered)}
    selections = [sel]
    for arm, labels in zip(ARMS, (full, clipped, reseg)):
        if arm == ORIGINAL:
            measures = original_measures
        else:
            check(labels.shape == mask.shape and np.array_equal(labels >= 0, mask),
                  k + ": regional support does not equal the saved v3 mask")
            measures, selected = base.selection(base.masked_forward(model, projected, labels), labels, mask)
            check(measures["outside_fov_selected_pixels"] == 0, k + ": selected pixels beyond estimated FOV")
            selections.append(selected)
        row.update({arm + "_" + key: val for key, val in measures.items()})
    return row, selections


def comparison(rows, metrics_fn):
    truth = np.array([r["y_true"] for r in rows], dtype=int)
    preds = {arm: np.array([r[arm+"_pred"] for r in rows], dtype=int) for arm in ARMS}
    base = preds[ORIGINAL]
    metrics, changes, pooled, diff = {}, {}, {}, {}
    for arm, pred in preds.items():
        cm = np.zeros((5, 5), dtype=int)
        np.add.at(cm, (truth, pred), 1)
        metrics[arm] = metrics_fn(cm)
        n_out = sum(int(r[arm+"_outside_fov_selected_pixels"]) for r in rows)
        n_sel = sum(int(r[arm+"_top_pixels"]) for r in rows)
        pooled[arm] = dict(outside_selected_pixels=n_out, selected_pixels=n_sel,
                           fraction_outside_estimated_FOV=n_out/n_sel)
        if arm != ORIGINAL:
            b, p = base == truth, pred == truth
            changes[arm] = dict(different_grade=int((pred != base).sum()),
                                correct_to_incorrect=int((b & ~p).sum()),
                                incorrect_to_correct=int((~b & p).sum()),
                                different_grade_both_incorrect=int(((base != pred) & ~b & ~p).sum()))
            diff[arm] = {m: metrics[arm][m] - metrics[ORIGINAL][m] for m in METRICS}
    return dict(metrics=metrics, grade_changes_vs_original=changes, delta_vs_original=diff,
                pooled_top10_selected_pixels=pooled)


def new_run(root, mode):
    folder = root / "experiments/aptos_fov_a3_compare_v3/runs"
    check(folder.resolve() == folder, "Refusing a redirected output folder")
    folder.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + mode + "_" + uuid4().hex[:10]
    run = folder/name
    run.mkdir()
    with (run/".gitignore").open("x", encoding="ascii") as f:
        f.write("*\n")
    return run


def preview(run, k, sample, mask, selections, row, base):
    from PIL import ImageDraw
    rgb = (sample["rgb"].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
    panel = base.result_panel(rgb, mask, selections)
    page = Image.new("RGB", (panel.width, panel.height+31), "#f6f7f6")
    title = (f"{k} | validation grade {row['y_true']} | frozen A3 grades: full "
             f"{row[ORIGINAL+'_pred']} / clipped {row[CLIP+'_pred']} / resegmented {row[RESEG+'_pred']}")
    ImageDraw.Draw(page).text((4, 7), title, fill="#193632")
    page.paste(panel, (0, 31))
    with (run/(k+"_top10.png")).open("xb") as f:
        page.save(f, format="PNG")


def make_packet(run):
    downloads = Path.home()/"Downloads"
    downloads.mkdir(exist_ok=True)
    packet = downloads/("CP2_A3_FOV_v3_"+run.name+".zip")
    with zipfile.ZipFile(packet, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as out:
        for name in ("protocol.json", "summary.json", "paired_predictions.csv", *[k+"_top10.png" for k in GATE]):
            out.write(run/name, name)
    return packet


def execute(root, run, mode, device_choice):
    # The installed, unchanged pilot supplies reusable region/graph/rollout functions only.
    pilot = root/"experiments/aptos_fov_v1"
    for name, digest in FOV_CODE.items():
        path = pilot/name
        check(path.is_file(), "Missing original APTOS FOV pilot helper: "+str(path))
        check(sha(path.read_bytes().replace(b"\r\n", b"\n")) == digest,
              "This comparison requires the reviewed pilot helper: "+name)
    sys.path[:0] = [str(root), str(pilot)]
    import run_fov as base
    import fov_regions as fov
    import torch
    import dataset_v1 as dsmod
    import evaluate_v1 as ev
    import attention_rollout_v1 as rollout
    check(base.ROOT == root and Path(fov.__file__).resolve() == pilot/"fov_regions.py" and
          tuple(base.ARMS) == ARMS, "Imported the wrong CP2 experiment modules.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ev.tr.seed_runtime(42)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_choice == "auto" else device_choice)
    geometry_path, geometry, protected = read_geometry(root)
    print("Matching reviewed v3 run:", geometry_path, flush=True)
    masks = collect_masks(geometry_path, geometry, protected)
    project = dsmod.load_project(root)
    ds = dsmod.AptosDataset(project, "validation", verify_file_hash=True)
    index = {row["id_code"]: i for i, row in enumerate(ds.rows)}
    check(len(ds) == len(geometry) == len(index) == 527 and set(index) == set(geometry) and
          set(GATE) <= set(index), "Expected the same 527 APTOS validation IDs as the reviewed geometry.")
    print("527 saved masks verified; checking the formal A3 checkpoint and epoch-21 archive.", flush=True)
    model, reference, locked = base.validated_model(project, device)
    check(not model.training and not any(p.requires_grad for p in model.parameters()), "A3 must be frozen.")
    protected.update(locked)
    protected.update({root/p: h for p, h in project["source_sha256"].items()})
    for p in (Path(base.__file__), Path(fov.__file__), Path(rollout.__file__), Path(__file__).resolve()):
        protected[p] = base.sha(p)
    protocol = dict(version="CP2_frozen_A3_FOV_v3.0", mode=mode, device=str(device),
                    cohort="All 527 locked APTOS validation IDs; the gate is technical and not a second cohort.",
                    primary="original_A3 vs FOV_resegmented (new masked boundary-SLIC partition, target 256)",
                    secondary="original_regions_clipped; disconnected fragments may increase the token count",
                    input="Identical original processed RGB, normalisation, DenseNet weights and canvas; the CNN can see the rim. No lesion removal.",
                    geometry="Saved v3 PNGs only; no automatic fallback, exclusions, or recomputation.",
                    geometry_folder=geometry_path.name, geometry_audit_digest=AUDIT, protocol_digest=PROTOCOL,
                    reviewed_groups={"reviewed85": {"refined":28,"kept":54,"held":3}, "other442":{"kept":442}},
                    known_broad_corners_retained=list(BROAD),
                    preview_caveat="Additional thin halos can occur, e.g. a56729de89e9 on the preview; no anatomical ground truth.",
                    top10=base.POLICY["top10"], resegmented=base.POLICY["resegmented"],
                    gate_ids=list(GATE), technical_gate="Check original archived grade/label and local all-valid adapter logits and attentions on all six; reuse rows for the full run.",
                    performance="Descriptive only: primary QWK, supporting five-class macro-F1/accuracy; deltas = variant minus original on the same images.",
                    interpretation="Zero outside the estimated mask is enforced in the two FOV arms; it is not evidence of improved accuracy or lesion localisation.",
                    frozen_A3_sha256=base.BEST_SHA, comparison_code_sha256=base.sha(Path(__file__)),
                    pilot_code_lf_sha256=FOV_CODE, rollout_source_sha256=protected[Path(rollout.__file__)])
    base.json_new(run/"protocol.json", protocol)
    order = list(GATE) + ([] if mode == "smoke" else sorted(set(index)-set(GATE)))
    records, writer = [], None
    tic = time.monotonic()
    print(f"Technical checks {len(GATE)} | mode {mode} | device {device}", flush=True)
    with torch.inference_mode(), (run/"paired_predictions.csv").open("x", encoding="utf-8", newline="") as f:
        for step, k in enumerate(order, 1):
            sample = ds[index[k]]
            check(sample["id_code"] == k, "Unexpected image order")
            protected[root/"processed_images_384_v1"/(k+".png")] = ds.processed[k]["output_sha256"]
            mask = np.unpackbits(masks[k]).reshape(384, 384).astype(bool)
            gate_case = step <= len(GATE)
            row, selections = one_image(sample, mask, geometry[k], reference[k], model, device, base, fov, gate_case)
            writer = base.append(writer, f, row)
            records.append(row)
            if gate_case:
                preview(run, k, sample, mask, selections, row, base)
            if step == len(GATE):
                print("6/6 technical checks PASS." + (" Continuing with the other 521." if mode == "compare" else ""), flush=True)
            base.tick(step, len(order), tic, gate_case)
    check(len(records) == len(set(order)) == (527 if mode == "compare" else len(GATE)),
          "Incomplete paired output")
    print("Verifying source files, model weights and the saved masks are unchanged...", flush=True)
    for p, digest in protected.items():
        check(base.sha(p) == digest, "Input changed during this run: "+str(p))
    summary = dict(status="COMPARISON_COMPLETE" if mode == "compare" else "SMOKE_ONLY",
                   mode=mode, paired_images=len(records), geometry_masks_verified=len(masks),
                   baseline_archive_grade_matches=len(records), technical_gate_images=len(GATE),
                   protected_files_verified=len(protected), frozen_A3_sha256=base.BEST_SHA,
                   device=str(device), optimizer_steps=0, test_images_decoded=0, IDRiD_images_decoded=0,
                   masks_estimated_not_anatomical_ground_truth=True)
    if mode == "compare":
        summary.update(comparison(records, ev.metrics_from_confusion))
        for arm in ARMS:
            print(arm, " | ".join(f"{k}: {summary['metrics'][arm][k]:.5f}" for k in METRICS), flush=True)
    base.json_new(run/"summary.json", summary)
    return make_packet(run)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("smoke", "compare"), nargs="?", default="smoke")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()
    root = Path.cwd().resolve()
    check((root/"dataset_v1.py").is_file() and (root/"experiments/aptos_fov_geometry_v3/runs").is_dir(),
          "Start from /c/CP2 in the original venv, with your previous v3 geometry run present.")
    run = new_run(root, args.mode)
    print("New separate A3 experiment:", run, flush=True)
    try:
        packet = execute(root, run, args.mode, args.device)
    except Exception as exc:
        with (run/"failure.json").open("x", encoding="utf-8") as f:
            json.dump({"status":"FAILED", "type":type(exc).__name__, "message":str(exc)}, f, indent=2)
        raise
    print("Done. Return this ZIP:", packet, flush=True)


if __name__ == "__main__":
    main()
