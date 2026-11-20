"""CP2: compare estimated retinal outlines. Run from C:/CP2 in the existing venv.

python -B ~/Downloads/aptos_fov_corner_preview_v2.py

Read-only APTOS validation RGB/geometry review. No predictions, lesion masks,
weights, optimisation, test images or IDRiD. Old masks and outputs are never updated.
The corner rule was devised after visually inspecting two v1 validation masks.
It is a candidate, not an anatomical reference or a validated model change.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

CROSS = ndi.generate_binary_structure(2, 1)
V1_SHA = "11e130e464f5259a309a6af3ff40b8879bf68c2b4f5e82c833636f9f35604672"
CASES = ("0fc6829da85b", "4a7dc013e802")
CLIP_CASE = "4704dbb59536"
POLICY = {
    "version": "aptos_fov_corner_preview_v2.0",
    "geometry_only": True, "mask_baseline_sha256": V1_SHA,
    "input": "unchanged RGB uint8 384x384, fixed APTOS validation IDs, PNG hashes verified",
    "evidence": "four 16x16 tiles inset by 3 px in the non-padding image bounds; "
                "at least three dark, low-spread, near-neutral tiles whose 95th percentiles agree",
    "rule": "raise v1 threshold only when supported by camera-corner levels: median accepted p95 + 2, "
            "capped at 20 and at v1 + 6. Remove only near-black pixels reachable from the exterior.",
    "protection": "do not cut isolated interior dark areas; do not apply a circle. "
                  "Return v1 for inconclusive corners, removed bright tissue, disconnected bright fragments, "
                  "or changes >15% of v1 area. Substantial changes and holds require review.",
    "selection": "fixed examples chosen by ID hash, both previously reviewed cases, "
                 "and largest changes; no class, prediction or accuracy used",
    "nodes": "one CPU geometry probe on 4704dbb59536, reusing v1 segmentation. Counts only, no classifier or rollout",
    "interpretation": "estimated outline, not expert FOV ground truth. Includes the two development examples.",
}


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda:f.read(1024*1024), b""): h.update(buf)
    return h.hexdigest()


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rect(rgb_max):
    # This bounding box is only for locating background samples, never for cropping the image.
    above = rgb_max > 2
    rows = np.flatnonzero(above.sum(axis=1) >= 5)
    cols = np.flatnonzero(above.sum(axis=0) >= 5)
    if not rows.size or not cols.size: return None
    y0, y1 = int(rows[0]), int(rows[-1])+1
    x0, x1 = int(cols[0]), int(cols[-1])+1
    return (y0,y1,x0,x1) if min(y1-y0,x1-x0) >= 64 else None


def candidate(rgb, old, old_t):
    require(rgb.dtype == np.uint8 and rgb.shape == (384,384,3) and
            old.dtype == bool and old.shape == rgb.shape[:2] and old.any(), "Invalid input/mask")
    maximum = rgb.max(2)
    box = rect(maximum)
    note = dict(action="held", reason="", corner_tiles=0, corner_p95="", threshold=int(old_t),
                attempted_removed=0, removed=0, removed_fraction_of_v1=0.0, review=0)
    def hold(reason):
        note.update(reason=reason, review=1)
        return old.copy(), note
    if box is None: return hold("no_reliable_camera_bounds")
    y0,y1,x0,x1 = box
    qs = []
    for y in (y0+3,y1-19):
        for x in (x0+3,x1-19):
            tile = rgb[y:y+16,x:x+16]
            v = tile.max(2)
            lo, hi = np.percentile(v,[5,95])
            spread = tile.max(2).astype(int) - tile.min(2)
            if hi <= 24 and hi-lo <= 6 and np.percentile(spread,90) <= 6: qs.append(float(hi))
    note["corner_tiles"] = len(qs)
    note["corner_p95"] = "|".join(f"{q:g}" for q in qs)
    if len(qs) < 3 or max(qs)-min(qs) > 4: return hold("inconclusive_camera_corners")
    t = max(int(old_t), min(int(old_t)+6,20,math.ceil(float(np.median(qs)))+2))
    note["threshold"] = t
    if t == old_t:
        note.update(action="kept",reason="v1_threshold_already_covers_corner_level")
        return old.copy(), note
    near = (ndi.median_filter(maximum, size=3) <= t) & (maximum <= t+2)
    exterior = ndi.binary_propagation(~old, mask=(~old)|near, structure=CROSS)
    core = old & ~exterior
    labels, n = ndi.label(core,structure=CROSS)
    if n == 0: return hold("no_connected_foreground")
    sizes = np.bincount(labels.ravel()); sizes[0]=0
    largest = labels==int(sizes.argmax())
    # If the candidate would split away bright islands, keep the old footprint for review.
    loose = core & ~largest
    if np.any(loose & (maximum>t+4)): return hold("detached_bright_pixels")
    # Bridge one-pixel edge irregularities; include interior dark tissue, and stay inside v1.
    padded = np.pad(largest, 2, mode="edge")
    smooth = ndi.binary_closing(padded, structure=CROSS, iterations=1)[2:-2,2:-2]
    new = old & ndi.binary_fill_holes(smooth | largest,structure=CROSS)
    lost = old & ~new
    count = int(lost.sum()); frac = count/int(old.sum())
    note["attempted_removed"] = count
    if frac > .15 or np.any(lost & (maximum>t+4)) or (old[192,192] and not new[192,192]):
        return hold("large_or_unsafe_change")
    if ndi.label(new,structure=CROSS)[1]!=1: return hold("disconnected_candidate")
    note.update(action="refined" if count else "kept",reason="inspect_change" if frac>=.01 else "minor_or_no_change",
                removed=count,removed_fraction_of_v1=frac,review=int(frac>=.01))
    return new,note


def boundary(mask):
    return mask & ~ndi.binary_erosion(mask,structure=CROSS,border_value=0)


def paint(rgb, old, new):
    a=rgb.copy()
    r,b=boundary(old),boundary(new)
    a[r & ~b]=(248,124,184)   # pink: v1 only
    a[b & ~r]=(40,222,216)    # cyan: candidate only
    a[b & r]=(255,222,128)    # gold: same boundary pixel
    return Image.fromarray(a)


def save_sheet(path, keys, thumbs):
    if not keys: return
    w,h=240,275
    sheet=Image.new("RGB",(3*w,47+h*math.ceil(len(keys)/3)),"#f2f5f5")
    d=ImageDraw.Draw(sheet)
    d.text((9,6),"Outline comparison only | pink: v1 | cyan: v2 candidate | gold: both",fill="#172d33")
    d.text((9,24),"Removed = percentage of v1 mask pixels. Check for dark tissue cut off.",fill="#374c51")
    for i,key in enumerate(keys):
        im,caption=thumbs[key]; x=i%3*w+6;y=i//3*h+47
        sheet.paste(im,(x,y));d.text((x,y+230),key,fill="#152b30");d.text((x,y+246),caption,fill="#36484b")
    with path.open("xb") as f: sheet.save(f,format="PNG")


def fragment_stats(full, mask, v1):
    clipped = v1.connected_labels(np.where(mask,full,-1),1)
    ids = clipped[mask]
    areas = np.bincount(ids)
    fragment, first = np.unique(ids,return_index=True)
    parent = full[mask][first]
    require(len(areas)==len(fragment) and np.all(areas>0) and np.array_equal(clipped>=0,mask),"fragment mismatch")
    retained, parts = np.unique(parent,return_counts=True)
    return dict(full_canvas_regions=int(full.max())+1,source_regions_touching_mask=len(retained),
                source_regions_split=int((parts>1).sum()),max_parts_per_source=int(parts.max()),
                extra_parts_over_retained_sources=int((parts-1).sum()),
                final_clipped_regions=len(areas),one_pixel_parts=int((areas==1).sum()),
                parts_at_most_8_pixels=int((areas<=8).sum()),area_in_these_parts=int(areas[areas<=8].sum()),
                clipped_top10_region_count=(len(areas)+9)//10)


def probe(rgb,old,new,v1):
    import torch
    import boundary_map_v1 as bm
    import superpixels_v1 as sp
    torch.set_num_threads(min(4,torch.get_num_threads()))
    with torch.inference_mode():
        t=torch.from_numpy(rgb.copy()).permute(2,0,1).float().div(255).contiguous()
        details=bm.boundary_map(t,return_details=True)
        full=sp.segment_superpixels(details["lab"],details["boundary"][0],boundary_weight=2.)["labels"]
    v1stats=fragment_stats(full,old,v1)
    return {"id_code":CLIP_CASE,"same_full_labels_in_both_arms":True,
            "v1_counts_match_smoke":(v1stats["full_canvas_regions"],v1stats["final_clipped_regions"])==(255,522),
            "v1":v1stats,"v2_candidate":fragment_stats(full,new,v1)}


def outdir(root):
    base=root/"experiments"/"aptos_fov_corner_review_v2"/"runs"
    require(base.resolve()==base,"Refusing redirected output directory")
    base.mkdir(parents=True,exist_ok=True)
    run=base/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:10])
    run.mkdir(exist_ok=False)
    with (run/".gitignore").open("x") as f:f.write("*\n")
    return run


def run_preview(root,run,v1):
    audit=root/"environment"/"data_audit"
    splits=audit/"split_v1"/"validation_v1.csv"
    manifest=audit/"preprocessing_v1"/"preprocessing_manifest_v1.csv"
    pinned={p:digest(p) for p in (splits,manifest,Path(v1.__file__))}
    rows=read_csv(splits)
    ids=sorted(r["id_code"] for r in rows)
    require(len(ids)==len(set(ids))==527 and
            all(r["split"]=="validation" and r["status"]=="keep" and re.fullmatch("[0-9a-f]{12}",r["id_code"]) for r in rows),
            "Expected the original 527 kept APTOS validation IDs")
    records=[r for r in read_csv(manifest) if r["split"]=="validation"]
    hashes={r["id_code"]:r["output_sha256"] for r in records if r["status"]=="ok"}
    require(len(records)==len(hashes)==527 and set(hashes)==set(ids),"Validation RGB manifest mismatch")
    require(set((*CASES,CLIP_CASE))<=set(ids),"Wrong validation manifest")
    json_data={**POLICY,"code_sha256":digest(Path(__file__)),
               "validation_manifest_sha256":pinned[splits],"preprocessing_manifest_sha256":pinned[manifest]}
    write_json(run/"protocol.json",json_data)
    fixed=[*CASES, CLIP_CASE]
    examples=sorted((k for k in ids if k not in fixed),
                    key=lambda k:hashlib.sha256(("FOV_V2_INDEPENDENT_SAMPLE|"+k).encode()).digest())[:9]
    keys=[*fixed,*examples]
    thumbs={}; cards=[]; flags=[]; failures=[]; geom=None
    from collections import Counter
    reasons=Counter(); actions=Counter(); largest=[]
    output_masks=run/"candidate_masks";output_masks.mkdir()
    fieldnames=("id_code","v1_pixels","candidate_pixels","v1_fraction_of_canvas","candidate_fraction_of_canvas",
                "v1_threshold","v1_flags","action","reason","corner_tiles","corner_p95","threshold","attempted_removed",
                "removed","removed_fraction_of_v1","review")
    with (run/"mask_audit_v2.csv").open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader()
        for i,key in enumerate(ids,1):
            p=root/"processed_images_384_v1"/(key+".png")
            require(digest(p)==hashes[key],key+": source PNG hash mismatch")
            with Image.open(p) as im:
                require(im.mode=="RGB" and im.size==(384,384),key+": invalid input")
                rgb=np.asarray(im).copy()
            try:
                old,old_stats=v1.fundus_fov(rgb)
                new,note=candidate(rgb,old,old_stats["threshold"])
            except v1.MaskFailure as e:
                failures.append(key);flags.append(key)
                writer.writerow(dict(id_code=key,action="failed",reason=str(e),review=1))
                thumbs[key]=(Image.fromarray(rgb).resize((222,222)), "FAILED: no v1 footprint")
                continue
            require(not (new & ~old).any() and new.any(),"candidate outside v1 mask")
            if old_stats["flags"]:note["review"]=1
            rec=dict(id_code=key,v1_pixels=int(old.sum()),candidate_pixels=int(new.sum()),
                     v1_fraction_of_canvas=float(old.mean()),candidate_fraction_of_canvas=float(new.mean()),
                     v1_threshold=old_stats["threshold"],v1_flags=old_stats["flags"],**note)
            writer.writerow(rec)
            reasons[note["reason"]]+=1;actions[note["action"]]+=1
            if note["review"]:flags.append(key)
            im=paint(rgb,old,new).resize((222,222))
            thumbs[key]=(im, f"{note['action']} | removed {100*note['removed_fraction_of_v1']:.2f}%")
            largest.append((note["removed_fraction_of_v1"],key))
            with (output_masks/(key+".png")).open("xb") as b:Image.fromarray(new.astype(np.uint8)*255).save(b,format="PNG")
            if key in CASES:cards.append(rec)
            if key==CLIP_CASE:geom=(rgb,old,new)
            if i==1 or i%50==0 or i==len(ids):print(f"RGB/masks {i}/{len(ids)}",flush=True)
    save_sheet(run/"review_01.png",keys,thumbs)
    changed=[key for frac,key in sorted(largest,key=lambda e:(-e[0],e[1])) if frac>0][:12]
    if changed:save_sheet(run/"largest_changes_01.png",changed,thumbs)
    for j in range(0,len(flags),12):
        save_sheet(run/f"flagged_{j//12+1:02d}.png",flags[j:j+12],thumbs)
    print("Checking clipped-region counts for "+CLIP_CASE+" (CPU geometry only)...",flush=True)
    fr=probe(*geom,v1) if geom is not None else {"id_code":CLIP_CASE,"error":"No usable mask"}
    write_json(run/"fragment_check.json",fr)
    require(all(digest(p)==h for p,h in pinned.items()),"Upstream manifests/source changed")
    summary=dict(status="GEOMETRY_REVIEW_REQUIRED",requested=len(ids),mask_pairs=len(largest),failures=failures,
                 actions=dict(actions),reasons=dict(reasons),flagged_for_visual_review=len(flags),
                 case_rows=cards,fragment_check=fr,model_forwards=0,optimizer_steps=0,checkpoint_reads=0,
                 test_images_opened=0,IDRiD_images_opened=0,interpretation="Not a classification run or an expert FOV evaluation.")
    write_json(run/"summary.json",summary)
    print(summary["status"],"|",run,flush=True)


def main():
    root=Path.cwd().resolve()
    src=root/"experiments"/"aptos_fov_v1"/"fov_regions.py"
    require(src.is_file() and (root/"dataset_v1.py").is_file(),"Run from the CP2 root with the original FOV pilot installed")
    # Allow platform line endings; compare the code that is actually imported.
    require(hashlib.sha256(src.read_bytes().replace(b"\r\n",b"\n")).hexdigest()==V1_SHA,"Unexpected v1 source")
    sys.path[:0]=[str(src.parent),str(root)]
    import fov_regions as v1
    require(Path(v1.__file__).resolve()==src,"Unexpected v1 import")
    run=outdir(root);print("New geometry review:",run,flush=True)
    try:run_preview(root,run,v1)
    except Exception as e:
        write_json(run/"failure.json",{"type":type(e).__name__,"message":str(e)})
        raise

if __name__=="__main__":main()
