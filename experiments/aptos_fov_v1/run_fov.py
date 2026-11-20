"""Read-only A3 sensitivity experiment. Run from an extracted CP2 project copy or checkout."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from fov_regions import MaskFailure, connected_labels, fundus_fov, masked_slic, CROSS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]   # CP2/experiments/aptos_fov_v1/run_fov.py
ARMS = ("original_A3", "original_regions_clipped", "FOV_resegmented")
BEST_SHA = "921e0368a183287939d0142b4d2a1c2b0c3da5db5a3b97791ee7adff4a947219"
POLICY = {
    "version": "aptos_fov_v1.0", "cohort": "APTOS locked validation only; 527, never test/IDRiD",
    "clinical_labels_or_lesion_masks_used_for_fov": False, "target_seed_count": 256,
    "fov": "RGB max > clip(border p10 + 6, 7, 28); binary closing (cross, two iterations); "
           "largest 4-connected component, fill enclosed holes; estimated, visually inspect",
    "arms": list(ARMS), "clipped": "original A3 partition intersected with FOV, splitting disconnected remnants",
    "resegmented": "256 equal-area seeds inside FOV; valid pixels only; 10 updates; compactness 10; "
                   "gamma=2 boundary path; same CP2 Lab/Scharr family; nearest-inside extension for boundary filter; "
                   "normalise peak inside FOV; 4-connected merge below 0.25 x FOV area/256",
    "cnn": "same unchanged 384px RGB and normalisation for all arms; receptive fields may include the rim",
    "geometry": "centroids in original canvas coordinates; area/384^2 (not renormalised by FOV)",
    "top10": "ceil(0.10 * actual region count), CLS excluded; score descending then region ID ascending",
    "metrics": "paired five-class QWK, macro-F1, accuracy; original reference must replay exactly by ID",
    "interpretation": "frozen-weight input/region sensitivity only, not retraining or a new external result",
}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib,"file_digest") else _digest(f)


def _digest(f):
    h = hashlib.sha256()
    for buf in iter(lambda:f.read(1024*1024), b""): h.update(buf)
    return h.hexdigest()


def json_new(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def start_run(mode):
    # There is no output override, resume, latest pointer or existing-file replacement.
    require(HERE == (ROOT / "experiments" / "aptos_fov_v1"), "Keep the distributed folder structure.")
    require((ROOT / "dataset_v1.py").is_file(), "Extract experiments/ under the CP2 project root.")
    base = HERE / "runs"
    require(base.resolve() == base and HERE.resolve() == HERE, "Output must not redirect to another directory.")
    base.mkdir(exist_ok=True)
    run = base / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + mode + "_" + uuid4().hex[:10])
    run.mkdir(exist_ok=False)
    return run


def ordered_sample(rows, n):
    return sorted(range(len(rows)), key=lambda i: hashlib.sha256(
        ("APTOS_FOV_V1_VISUAL|" + rows[i]["id_code"]).encode()).digest())[:n]


def mask_overlay(rgb, mask):
    out = np.asarray(rgb).copy()
    if mask is not None:
        out[~mask] = (out[~mask].astype(np.float32)*.25).astype(np.uint8)
        rim = mask & ~ndi.binary_erosion(mask, structure=CROSS)
        out[rim] = (43, 237, 193)
    return Image.fromarray(out)


def contact_sheet(run, prefix, items):
    if not items: return
    for j in range(0, len(items), 12):
        group = items[j:j+12]
        sheet = Image.new("RGB", (690, 255*math.ceil(len(group)/3)), "#f6f7f6")
        draw = ImageDraw.Draw(sheet)
        for n, (image, title, subtitle) in enumerate(group):
            x = n%3*230; y = n//3*255
            sheet.paste(image.resize((210,210)), (x+10, y+4))
            draw.text((x+10,y+216), title, fill="black")
            draw.text((x+10,y+232), subtitle[:35], fill="#403d3b")
        with (run / (prefix + "_%02d.png" % (j//12+1))).open("xb") as f: sheet.save(f, format="PNG")


def append(writer, stream, row):
    if writer is None:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
    writer.writerow(row)
    stream.flush()
    return writer


def tick(done, total, t0, force=False):
    if force or done == total or done == 1 or done % 10 == 0:
        elapsed = time.monotonic()-t0
        print(f"{done}/{total} | elapsed {elapsed/60:.1f} min | est. remaining {(total-done)*elapsed/max(1,done)/60:.1f} min",
              flush=True)


def preview(run, ds):
    selected = set(ordered_sample(ds.rows, 12))
    examples, flagged = [], []
    failures = flags = 0
    mask_dir = run / "masks"
    mask_dir.mkdir()
    fw = None; tic = time.monotonic()
    with (run/"mask_audit.csv").open("x", encoding="utf-8", newline="") as f:
        for i, row in enumerate(ds.rows):
            name = row["id_code"]
            rgb = np.asarray(ds.read_image(name))
            try:
                m, stats = fundus_fov(rgb)
            except MaskFailure as e:
                m = None; failures += 1
                stats = dict(fov_pixels=None, fov_fraction=None, threshold=None, frame_p10=None,
                             flags="FAILED: "+str(e), second_component_pixels=None)
            if m is not None:
                with (mask_dir / (name+".png")).open("xb") as out:
                    Image.fromarray(m.astype(np.uint8)*255).save(out, format="PNG")
            fw = append(fw, f, {"id_code":name, **stats})
            if stats["flags"]: flags += 1
            if i in selected or stats["flags"]:
                overlay = mask_overlay(rgb, m).resize((210, 210))
                item = (overlay, name, stats["flags"] or f"FOV {100*m.mean():.1f}%")
                if i in selected: examples.append(item)
                if stats["flags"]: flagged.append(item)
            tick(i+1, len(ds), tic)
    contact_sheet(run, "sample", examples)
    contact_sheet(run, "flagged", flagged)
    return {"status": "MASK_REVIEW_REQUIRED" if flags else "MASK_PREVIEW_COMPLETE",
            "images": len(ds), "mask_files":len(ds)-failures, "flagged_including_failures":flags,
            "no_mask":failures, "model_forwards":0, "inspection":"check contour and all warnings before interpreting comparisons"}


def validated_model(project, device):
    import evaluate_v1 as ev
    root = project["root"]
    run = root/"environment/data_audit/training_v1"/ev.RUNS["A3"]
    files = {k:run/f"training_{k}_v1.json" for k in ("config","history","summary")}
    cfg, history, summary = (ev.read_json(files[k]) for k in ("config","history","summary"))
    require(cfg["variant"] == summary["variant"] == "A3" and cfg["mode"] == summary["mode"] == "train" and
            summary["status"] == "TRAINING_COMPLETE", "Not the completed formal A3.")
    require(summary["best_epoch"] == 21 and summary["best_checkpoint"] == "best_epoch_0021.pt", "Wrong best epoch.")
    require(cfg["core"]["settings"] == cfg["settings"] == ev.tr.SETTINGS, "A3 settings differ.")
    require(ev.tr.fingerprint(cfg["core"]) == cfg["core_fingerprint"], "Training metadata mismatch.")
    mods = [Path(m.__file__) for m in (*ev.tr.MODULES, ev.tr)]
    require({p.name:sha(p) for p in mods} == cfg["core"]["code_sha256"], "Training module hashes differ.")
    require(project["source_sha256"] == cfg["core"]["data_sha256"], "Different preprocessing/split manifests.")
    require(cfg["core"]["versions"] == ev.runtime_versions(), "Use the original CP2 training environment.")
    ev.validate_history(cfg, history, summary)
    ckpt = run/summary["best_checkpoint"]
    require(sha(ckpt) == BEST_SHA, "A3 checkpoint does not match its accepted SHA-256.")
    info = dict(config=cfg, history=history, summary=summary, variant="A3", best_path=ckpt, best_sha256=BEST_SHA)
    model, state = ev.load_best_model(info, device)
    del state
    ref_file = run/"validation_epoch_0021.npz"
    with np.load(ref_file, allow_pickle=False) as f:
        ids, y, p = (f[k].tolist() for k in ("id_codes","labels","predictions"))
    refs = {k:(int(a),int(b)) for k,a,b in zip(ids,y,p)}
    rows = project["rows"]["validation"]
    require(len(ids)==len(y)==len(p)==len(refs)==len(rows)
            and all(r["id_code"] in refs and refs[r["id_code"]][0] == int(r["diagnosis"])
                    and 0 <= refs[r["id_code"]][1] < 5 for r in rows), "Invalid archived validation predictions.")
    cm = np.zeros((5,5), dtype=int)
    np.add.at(cm, (np.asarray(y, int), np.asarray(p, int)), 1)
    require(cm.tolist() == summary["best_validation_metrics"]["confusion_matrix"], "Archived predictions differ from epoch 21.")
    protected = {p:sha(p) for p in [*files.values(),ref_file,*mods,Path(ev.__file__)]}
    protected[ckpt] = BEST_SHA
    return model, refs, protected


def masked_forward(model, projected, labels):
    """A3 parameter reuse, one image. The sentinel is discarded before pooling or attention."""
    import torch
    import regional_features_v1 as rf
    import region_adjacency_v1 as rag
    valid = labels >= 0
    n = int(labels.max())+1
    require(n>0 and np.array_equal(np.unique(labels[valid]),np.arange(n)), "Foreground label IDs not contiguous.")
    work = np.where(valid, labels, n)  # temporary ID for existing full-map bookkeeping only
    plan = rf.make_pooling_plan(work)
    plan = rf.RegionPoolingPlan(plan.weights[:n], plan.geometry[:n], plan.areas[:n], plan.feature_hw, plan.image_hw)
    require(int(plan.areas.sum()) == int(valid.sum()), "Foreground pooling has wrong support.")
    regional = rf.pack_regions([rf.pool_regions(projected[0],plan)],[plan])
    graph = rag.build_region_graph(work)
    pairs, edges = graph["undirected_edges"], graph["edge_index"]
    kp = (pairs<n).all(1); ke = (edges<n).all(0)
    e = edges[:,ke]
    graph = dict(num_nodes=n, areas=graph["areas"][:n], undirected_edges=pairs[kp],
        boundary_lengths=graph["boundary_lengths"][kp], edge_index=e,
        directed_boundary_lengths=graph["directed_boundary_lengths"][ke], degree=np.bincount(e[0], minlength=n))
    rag.validate_graph(graph)
    packed = rag.pack_graph_batch(regional,[graph])
    refined = model.graph_stage(packed, return_attention=False)["x"]
    # Same learned token layers and order as v1. Only change to the input contract:
    # real areas sum to the FOV fraction, not to the whole image. No dummy token in ViT.
    x = refined[None,...]; geo = regional["geometry"]
    require(abs(float(geo[0,:,2].sum()) - float(valid.mean())) < 2e-5, "Wrong canvas area scale.")
    maker = model.preparer
    tokens = maker.token_projection(x) + maker.geometry_embedding(geo.to(x.dtype))
    tokens = maker.dropout(torch.cat((maker.cls_token.to(x.dtype).expand(1,-1,-1), tokens), dim=1))
    dev = x.device
    prepared = dict(tokens=tokens, key_padding_mask=torch.zeros((1,n+1), device=dev,dtype=torch.bool),
        token_region_ids=torch.arange(-1,n,device=dev)[None,:], region_counts=regional["region_counts"],
        sequence_lengths=regional["region_counts"]+1)
    enc = model.encoder(prepared, return_attention=True)
    return dict(logits=model.classifier(enc["cls"]), vit_attentions=enc["attentions"],
                key_padding_mask=enc["key_padding_mask"], token_region_ids=enc["token_region_ids"])


def selection(output, labels, mask):
    import attention_rollout_v1 as ar
    cls = ar.attention_rollout(output["vit_attentions"], output["key_padding_mask"])["joint"][0,0]
    ids = ar.as_numpy(output["token_region_ids"][0])
    fg = ids >= 0
    n = int(labels.max())+1
    require(np.array_equal(np.sort(ids[fg]), np.arange(n)), "Rollout token/region mismatch.")
    scores = np.empty(n, np.float64); scores[ids[fg]] = cls[fg]
    k = (n+9)//10
    top = np.lexsort((np.arange(n), -scores))[:k]
    covered = np.isin(labels, top)
    selected = int(covered.sum()); outside = int((covered & ~mask).sum())
    require(selected>0, "No selected pixel.")
    valid = labels >= 0
    if (~valid).any(): require(not (covered & ~valid).any(), "Rollout selected a sentinel.")
    probs = ar.as_numpy(output["logits"][0].double().softmax(-1))
    require(np.isfinite(probs).all() and abs(float(probs.sum())-1)<1e-8, "Invalid probabilities.")
    return dict(pred=int(probs.argmax()), **{f"p{c}":float(v) for c,v in enumerate(probs)},
                regions=n, top_regions=k, top_pixels=selected, outside_fov_selected_pixels=outside,
                outside_fraction_of_selected=outside/selected), covered


def result_panel(rgb, mask, selections):
    canvas = Image.new("RGB", (4*220,264), "#f6f7f6")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(("Estimated FOV",*ARMS)):
        if i == 0: im = mask_overlay(rgb, mask)
        else:
            sel = selections[i-1]
            im = np.asarray(rgb).copy()
            for use, color in ((sel & mask,(41,220,183)),(sel & ~mask,(249,113,125))):
                im[use] = (.35*im[use] + .65*np.array(color)).astype(np.uint8)
            im = Image.fromarray(im)
        canvas.paste(im.resize((216,216)),(i*220+2,2))
        draw.text((i*220+3,222),name,fill="black")
    draw.text((3,246),"Top 10% of regions: teal = selected inside estimated FOV; pink = selected outside FOV. Not a lesion map.",fill="#193632")
    return canvas


def compare(run, ds, project, mode, device_choice):
    import torch
    import evaluate_v1 as ev
    import attention_rollout_v1 as ar
    import boundary_map_v1 as bm
    import superpixels_v1 as sp
    ev.tr.seed_runtime(42)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_choice=="auto" else device_choice)
    model, reference, protected = validated_model(project, device)
    protected[Path(ar.__file__)] = sha(Path(ar.__file__))
    indices = ordered_sample(ds.rows, 3) if mode=="smoke" else list(range(len(ds)))
    sample_set = set(ordered_sample(ds.rows, 6)) if mode=="compare" else set(indices)
    cms = {a:np.zeros((5,5), dtype=np.int64) for a in ARMS}
    changed = {a:0 for a in ARMS[1:]}
    top_counts = {a:np.zeros(2, dtype=np.int64) for a in ARMS}
    problems, replay, warnings = [], [], []
    fw=None; tic=time.monotonic()
    with torch.inference_mode(), (run/"paired_predictions.csv").open("x",encoding="utf-8",newline="") as f:
        for step, i in enumerate(indices,1):
            s = ds[i]; name, truth = s["id_code"], s["label"]
            rgb = (s["rgb"].permute(1,2,0).numpy()*255).round().astype(np.uint8)
            protected[project["root"]/"processed_images_384_v1"/(name+".png")] = ds.processed[name]["output_sha256"]
            try: mask, stats = fundus_fov(rgb)
            except MaskFailure as e:
                problems.append({"id_code":name,"problem":str(e)})
                tick(step,len(indices),tic); continue
            if stats["flags"]: warnings.append({"id_code":name,"flags":stats["flags"]})
            base_details=bm.boundary_map(s["rgb"],return_details=True)
            full = sp.segment_superpixels(base_details["lab"],base_details["boundary"][0],boundary_weight=2.)["labels"]
            clipped = connected_labels(np.where(mask, full, -1), 1)
            reseg, holes = masked_slic(s["rgb"],mask)
            cnn = s["cnn"][None,...].to(device)
            original = model(cnn,[full],segmentation_method="boundary_slic",return_attention=True)
            projected = model.regional.extract(cnn)
            if fw is None:
                # Degenerate all-valid adapter must reproduce the official full-canvas forward, including attention.
                control = masked_forward(model,projected,full)
                torch.testing.assert_close(original["logits"], control["logits"], atol=2e-5, rtol=2e-5)
                for a,b in zip(original["vit_attentions"], control["vit_attentions"]):
                    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
            record={"id_code":name,"y_true":truth,"archived_A3_pred":reference[name][1],**stats,
                    "reseg_recovery_pixels_summed_over_10_iterations":holes}
            selections=[]; base_pred=None
            for arm, lab in zip(ARMS, (full,clipped,reseg)):
                if arm != ARMS[0]:
                    require(np.array_equal(lab>=0,mask), "Region membership not exactly equal to FOV.")
                val = original if arm==ARMS[0] else masked_forward(model,projected,lab)
                measures, chosen = selection(val,lab,mask)
                pred = measures["pred"]
                if arm==ARMS[0]:
                    base_pred=pred
                    if reference[name] != (truth,pred): replay.append(name)
                else:
                    require(measures["outside_fov_selected_pixels"]==0, "FOV arm selected outside.")
                    changed[arm] += int(pred!=base_pred)
                cms[arm][truth,pred] += 1
                top_counts[arm] += [measures["outside_fov_selected_pixels"],measures["top_pixels"]]
                record.update({arm+"_"+k:v for k,v in measures.items()})
                selections.append(chosen)
            fw=append(fw,f,record)
            if i in sample_set:
                with (run/(name+"_top10.png")).open("xb") as out: result_panel(rgb,mask,selections).save(out,format="PNG")
            tick(step,len(indices),tic,mode=="smoke")
    require(all(sha(p)==h for p,h in protected.items()), "An upstream source/checkpoint/validation file changed during the run.")
    eligible = not problems and not replay and len(indices)==len(ds) and mode=="compare"
    result = dict(status=("COMPARISON_COMPLETE" if not warnings else "COMPARISON_COMPLETE_REVIEW_MASKS") if eligible else
                  ("SMOKE_ONLY" if mode=="smoke" and not problems and not replay else "INCOMPLETE_OR_BASELINE_MISMATCH"),
                  requested=len(indices),paired=int(cms[ARMS[0]].sum()),mask_failures=problems,flagged_masks=warnings,
                  archived_baseline_disagreements=replay, frozen_A3_sha256=BEST_SHA,
                  accepted_baseline_replayed=(not replay and not problems),device=str(device),
                  unchanged_protected_files=len(protected),grade_changes_vs_original=changed)
    if eligible:
        metrics={a:ev.metrics_from_confusion(cms[a]) for a in ARMS}
        result["metrics"]=metrics
        result["delta_vs_original"]={a:{k:metrics[a][k]-metrics[ARMS[0]][k] for k in ("qwk","macro_f1","accuracy")} for a in ARMS[1:]}
        result["top10_outside_fraction_pooled_over_selected_pixels"]={a:float(t[0]/t[1]) for a,t in top_counts.items()}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", nargs="?", default="preview", choices=("preview","smoke","compare"))
    ap.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    args=ap.parse_args()
    run=start_run(args.mode)
    print("New experimental output:",run,flush=True)
    sys.path.insert(0, str(ROOT))
    try:
        import torch
        import dataset_v1 as d
        torch.set_num_threads(min(4,torch.get_num_threads()))
        json_new(run/"protocol.json", {**POLICY, "mode":args.mode,
            "experiment_code_sha256":{p.name:sha(p) for p in (Path(__file__),HERE/"fov_regions.py")}})
        project=d.load_project(ROOT)
        ds=d.AptosDataset(project,"validation",verify_file_hash=True)
        require(len(ds)==527,"Wrong validation split.")
        result=preview(run,ds) if args.mode=="preview" else compare(run,ds,project,args.mode,args.device)
        require(all(sha(ROOT/p)==h for p,h in project["source_sha256"].items()), "Source manifests changed.")
        json_new(run/"summary.json",{**result,"mode":args.mode,"optimizer_steps":0,"test_images_opened":0,"IDRiD_images_opened":0})
        print(result["status"],"|",run/"summary.json",flush=True)
        if "metrics" in result:
            for a,v in result["metrics"].items(): print(f"{a}: QWK {v['qwk']:.5f}; macro-F1 {v['macro_f1']:.5f}; accuracy {v['accuracy']:.5f}")
        return 2 if result["status"]=="INCOMPLETE_OR_BASELINE_MISMATCH" or result.get("no_mask",0)>0 else 0
    except Exception as e:
        json_new(run/"failure.json",{"status":"FAILED","type":type(e).__name__,"message":str(e)})
        raise


if __name__=="__main__":
    raise SystemExit(main())
