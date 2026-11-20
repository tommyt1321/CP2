"""CP2 APTOS geometry preview v3. From the CP2 root, in the existing venv:

    python -B ~/Downloads/aptos_fov_geometry_preview_v3.py

Reads existing 384 RGB/v2 masks, creates a new v3 run and a small review ZIP.
No classifier, training, network or changes to earlier outputs.
"""
import math
import numpy as np
from scipy import ndimage as ndi

CROSS = ndi.generate_binary_structure(2, 1)

def outline(m):
    return m & ~ndi.binary_erosion(m, structure=CROSS, border_value=0)

def refine(rgb, base):
    if rgb.shape != (384,384,3) or rgb.dtype != np.uint8 or base.shape != rgb.shape[:2] or base.dtype != bool or not base.any():
        raise ValueError('Expected uint8 384x384 RGB and nonempty boolean v2 mask')
    rec=dict(action='kept',reason='',corner_mask_pixels='',corner_patch_p95_spread='',removed_pixels=0,
             removed_fraction_of_v2=0.0,edge_support='',remaining_corner_pixels='',
             maximum_removed_channel='',probe_before=int(base[60,20]),probe_after=int(base[60,20]))
    def keep(why, action='kept'):
        rec.update(reason=why,action=action)
        return base.copy(),rec
    mx=rgb.max(axis=2); active=mx>2
    yy=np.flatnonzero(active.sum(1)>=5);xx=np.flatnonzero(active.sum(0)>=5)
    if not len(yy) or not len(xx):return keep('unresolved_frame','held')
    y0,y1,x0,x1=int(yy[0]),int(yy[-1])+1,int(xx[0]),int(xx[-1])+1
    if min(y1-y0,x1-x0)<64:return keep('small_frame','held')
    co=[(y,x) for y in (y0+3,y1-19) for x in (x0+3,x1-19)]
    occupied=[int(base[y:y+16,x:x+16].sum()) for y,x in co]
    rec.update(corner_mask_pixels=sum(occupied),remaining_corner_pixels=sum(occupied))
    if sum(occupied)<512:return keep('no_broad_corner_inclusion')
    a=rgb.astype(np.int16)
    meds=[]; spreads=[]
    for y,x in co:
        t=a[y:y+16,x:x+16]
        c=np.median(t,axis=(0,1))
        meds.append(c)
        spreads.append(float(np.percentile(np.abs(t-c).max(2),95)))
    rec['corner_patch_p95_spread']='|'.join(f'{d:g}' for d in spreads)
    if max(spreads)>4 or np.max(meds)>32 or np.max(np.ptp(meds,axis=0))>8:
        return keep('corners_not_uniform_dark_background','held')
    # Each quadrant uses its own camera colour; an enclosed similar colour cannot seed a deletion.
    dist=np.full(base.shape,255.,dtype=np.float32)
    tol=np.zeros(base.shape,dtype=np.float32)
    ym=(y0+y1)//2; xm=(x0+x1)//2
    for i,(ys,xs) in enumerate([(slice(ya,yb),slice(xa,xb)) for ya,yb in ((y0,ym),(ym,y1)) for xa,xb in ((x0,xm),(xm,x1))]):
        dist[ys,xs]=np.max(np.abs(a[ys,xs]-meds[i]),axis=2)
        tol[ys,xs]=min(7,max(4,math.ceil(spreads[i])+3))
    # Only within the actual recorded frame, no circular crop; flood just from verified corners.
    inframe=np.zeros_like(base);inframe[y0:y1,x0:x1]=True
    allow=inframe & (ndi.median_filter(dist,3,mode='nearest')<=tol) & (dist<=tol+2)
    labs,_=ndi.label(allow,structure=CROSS)
    rim=outline(inframe)
    edgeids=np.unique(labs[rim]);edgeids=edgeids[edgeids>0]
    comps=set()
    for y,x in co:
        patch=labs[y:y+16,x:x+16];seen=patch[np.isin(patch,edgeids)]
        if len(seen)<.8*patch.size:return keep('no_exterior_path_from_all_corners','held')
        vals,counts=np.unique(seen,return_counts=True)
        comps.update(int(v) for v,c in zip(vals,counts) if c>=16)
    if not comps:return keep('no_corner_component','held')
    exterior=np.isin(labs,list(comps));remaining=base & ~exterior
    lab,n=ndi.label(remaining,structure=CROSS)
    if n==0:return keep('empty_candidate','held')
    sizes=np.bincount(lab.ravel());sizes[0]=0
    main=lab==int(sizes.argmax());loose=remaining & ~main
    if np.any(loose & (dist>tol+4)):return keep('detached_pixels_unlike_camera','held')
    new=base & ndi.binary_fill_holes(main,structure=CROSS)
    lost=base&~new; count=int(lost.sum());frac=count/int(base.sum())
    if not count:return keep('no_change')
    h,w=y1-y0,x1-x0
    central=lost[y0+h//4:y1-h//4,x0+w//4:x1-w//4]
    if central.any() or frac>.15 or (base[192,192] and not new[192,192]) or np.any(lost & (~inframe | (dist>tol+4))) or ndi.label(new,structure=CROSS)[1]!=1:
        return keep('safety_guard','held')
    # A newly exposed boundary should meet a visible contrast step into the image.
    fresh=outline(new) & ~ndi.binary_dilation(outline(base),structure=CROSS,iterations=2)
    contrast=inframe & new & (dist>=tol+8)
    support=float(np.mean(ndi.maximum_filter(contrast,size=9)[fresh])) if fresh.any() else 0.
    rec['edge_support']=round(support,6)
    if fresh.sum()<32 or support<.75:return keep('weak_new_edge','held')
    rec.update(action='refined',reason='exterior_matches_camera_colour',removed_pixels=count,
               removed_fraction_of_v2=frac,remaining_corner_pixels=sum(int(new[y:y+16,x:x+16].sum()) for y,x in co),
               maximum_removed_channel=int(mx[lost].max()),probe_after=int(new[60,20]))
    return new,rec


import csv
import hashlib
import io
import json
import re
import statistics
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from PIL import Image, ImageDraw

AUDIT_SIG = "2ec1c77e9504d7a0510d13f5bb1948a765a0914f5879405267e35b72c58b35b7"
SUMMARY_SIG = "5a3ddbacb8aaf93921f40760d207a635fd069bb55e872953717b50fe0fa8d88a"
REFERENCE85_SIG = "b3ed7ee673adc0294fcec818449f90ed696014fb2b11c167abcd4489b4d20fa2"
CONTROL = "4704dbb59536"
# Review annotations only: refine() does not receive identifiers, labels or predictions.
KNOWN_BROAD_HOLDS = ["82910bba4753", "dc0eea0b68a7"]
PALETTE = {"v2_only": (249,126,180), "v3_only": (45,230,216), "both": (255,217,127)}
LUT = np.rint(255*(np.arange(256)/255)**.66).astype(np.uint8)

def require(ok,msg):
    if not ok: raise ValueError(msg)

def sha(b):
    return hashlib.sha256(b).hexdigest()

def canonical(x):
    return sha(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode())

def csvrows(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))

def sigline(key,rgb_sha,base,n,note):
    return [key,rgb_sha,sha(base.tobytes()),sha(n.tobytes()),note["action"],note["reason"]]

def known85(rows):
    ids=set()
    for r in rows:
        levels=[float(t) for t in r["corner_p95"].split("|") if t]
        if r["review"]=="1" or (r["action"]=="refined" and levels and statistics.median(levels)>float(r["threshold"])):
            ids.add(r["id_code"])
    ids.add(CONTROL)
    require(len(ids)==85, "Review set differs from the frozen 85 RGB/mask pairs")
    return ids

def find_v2(root):
    base=root/"experiments"/"aptos_fov_corner_review_v2"/"runs"
    require(base.is_dir(),"Start in C:/CP2, with the completed v2 geometry run available")
    for p in sorted((s for s in base.iterdir() if s.is_dir()),reverse=True):
        try:
            meta=json.loads((p/"summary.json").read_text(encoding="utf-8-sig"))
            if canonical(meta)!=SUMMARY_SIG: continue
            rows=sorted(csvrows(p/"mask_audit_v2.csv"),key=lambda r:r["id_code"])
            if canonical(rows)==AUDIT_SIG: return p,rows
        except (KeyError,FileNotFoundError,ValueError,csv.Error):
            continue
    raise ValueError("No completed v2 run matches the reviewed summary and 527-row audit")

def start_run(root):
    base=root/"experiments"/"aptos_fov_geometry_v3"/"runs"
    require(base.resolve()==base,"Refusing redirected v3 output path")
    base.mkdir(parents=True,exist_ok=True)
    token=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:10]
    new=base/token;new.mkdir(exist_ok=False)
    with (new/".gitignore").open("x") as f:f.write("*\n")
    return new

def json_new(path,obj):
    with path.open("x",encoding="utf8") as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write("\n")

def preview_tile(rgb,old,new):
    c=LUT[rgb].copy();a,b=outline(old),outline(new)
    c[a & ~b]=PALETTE["v2_only"];c[b & ~a]=PALETTE["v3_only"];c[a&b]=PALETTE["both"]
    # JPEG bytes for review only, never used in calculations.
    im=Image.fromarray(c).resize((216,216),Image.Resampling.LANCZOS)
    buf=io.BytesIO();im.save(buf,format="JPEG",quality=93)
    return buf.getvalue()

def pages(run,prefix,keys,tiles):
    saved=[]
    for start in range(0,len(keys),12):
        keys1=keys[start:start+12];page=Image.new("RGB",(3*240,54+252*math.ceil(len(keys1)/3)),"#f1f4f2")
        d=ImageDraw.Draw(page)
        d.text((9,5),f"APTOS validation | {prefix} | saved v2: pink / candidate v3: cyan / shared: gold",fill="#173132")
        d.text((9,25),"Full 384 canvas; colours lifted for display. Graded predictions have not been computed.",fill="#3b5051")
        for j,k in enumerate(keys1):
            jpg,caption=tiles[k];x=7+j%3*240;y=54+j//3*252
            with Image.open(io.BytesIO(jpg)) as im:page.paste(im,(x,y))
            d.text((x,y+218),k,fill="#153336");d.text((x,y+233),caption,fill="#30494a")
        name=f"{prefix}_{start//12+1:02d}.jpg"
        with (run/name).open("xb") as f:page.save(f,format="JPEG",quality=91)
        saved.append(name)
    return saved

def review_zip(run,files):
    dest=Path.home()/"Downloads"
    dest.mkdir(exist_ok=True)
    path=dest/("CP2_FOV_v3_geometry_"+run.name+".zip")
    with zipfile.ZipFile(path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in files:z.write(run/name,name)
    return path

def run_geometry(root,src,rows,run):
    ids=[r["id_code"] for r in rows];ref=known85(rows)
    require(len(rows)==len(set(ids))==527 and all(re.fullmatch("[0-9a-f]{12}",k) for k in ids),"Wrong validation audit")
    manifest=root/"environment"/"data_audit"/"preprocessing_v1"/"preprocessing_manifest_v1.csv"
    frozen={p:sha(p.read_bytes()) for p in (src/"summary.json",src/"mask_audit_v2.csv",manifest)}
    rgb_hash={r["id_code"]:r["output_sha256"] for r in csvrows(manifest) if r["split"]=="validation" and r["status"]=="ok"}
    require(set(rgb_hash)==set(ids),"Different validation image set")
    (run/"candidate_masks").mkdir()
    protocol=dict(version="CP2_FOV_GEOMETRY_v3.0",canvas="384x384; saved v2 masks; unchanged processed RGB",
                  data="fixed 527 APTOS validation IDs only; 85 seen in development, 442 outside the review packet",
                  corner_rule="4 in-frame 16x16 patches (inset 3); at least 512/1024 pixels within v2; "
                     "each local RGB camera median <=32/channel, patch p95 deviation <=4; inter-corner range <=8/channel",
                  proposal="connected camera-coloured components must reach the frame and >=80% of each seed patch; "
                     "per-channel distance tolerances derived from patch variation; isolated interior dark regions do not seed removal",
                  protections="no expansion; centre retained, central half-rectangle protected, at most 15% removed; "
                     "hold when detached pixels are unlike the camera, or edge contrast support <75% or <32 eligible edge pixels",
                  no_class_labels_used_by_geometry=True, reference85_sha256=REFERENCE85_SIG,code_sha256=sha(Path(__file__).read_bytes()),
                  caution="development-geometry audit; fallback copies v2 and can retain background. Not anatomical ground truth.")
    json_new(run/"protocol.json",protocol)
    tiles={};sigs=[];actions={"reviewed85":Counter(),"other442":Counter()};reasons=Counter()
    changed=[];held=[]
    with (run/"per_image.csv").open("x",encoding="utf-8",newline="") as out:
        writer=None
        for i,r in enumerate(rows,1):
            k=r["id_code"];g="reviewed85" if k in ref else "other442"
            rb=(root/"processed_images_384_v1"/(k+".png")).read_bytes();rh=sha(rb)
            require(rh==rgb_hash[k],k+": processed RGB hash mismatch")
            with Image.open(io.BytesIO(rb)) as im:
                require(im.mode=="RGB" and im.size==(384,384),k+": unexpected RGB");a=np.array(im)
            with Image.open(src/"candidate_masks"/(k+".png")) as im:
                require(im.mode=="L" and im.size==(384,384),k+": unexpected v2 mask");arr=np.array(im)
            m=arr==255;require(((arr==0)|m).all() and int(m.sum())==int(r["candidate_pixels"]),k+": v2 area or values mismatch")
            n,note=refine(a,m);require(n.any() and not (n & ~m).any(),k+": invalid v3 mask")
            if k in ref:sigs.append(sigline(k,rh,m,n,note))
            actions[g][note["action"]]+=1;reasons[note["reason"]]+=1
            if note["action"]=="refined":changed.append(k)
            if note["action"]=="held":held.append(k)
            rec=dict(id_code=k,group=g,v2_action=r["action"],v2_automatically_flagged=int(r["review"]),
                     v2_pixels=int(m.sum()),v3_pixels=int(n.sum()),v3_raw_sha256=sha(n.tobytes()),**note)
            if writer is None:writer=csv.DictWriter(out,fieldnames=list(rec));writer.writeheader()
            writer.writerow(rec)
            with (run/"candidate_masks"/(k+".png")).open("xb") as f:Image.fromarray(n.astype(np.uint8)*255).save(f,format="PNG")
            tiles[k]=(preview_tile(a,m,n),f"{note['action']} | removed {100*note['removed_fraction_of_v2']:.2f}% of v2")
            if i==1 or i%50==0 or i==len(rows):print(f"Geometry {i}/{len(rows)}",flush=True)
    require(canonical(sigs)==REFERENCE85_SIG,"Results for the 85 native examples differ; no success bundle published")
    require(all(sha(p.read_bytes())==v for p,v in frozen.items()),"Upstream metadata changed during preview")
    others=[k for k in ids if k not in ref];changed_new=[k for k in changed if k not in ref]
    require(len(others)==442,"Incorrect 442-image group")
    # All 442 are included, not only favourable or automatically flagged examples.
    sheets=pages(run,"other442_all",others,tiles)+pages(run,"other442_changed",changed_new,tiles)
    if held:sheets+=pages(run,"auto_held",held,tiles)
    summary=dict(status="GEOMETRY_COMPLETE_REVIEW_REQUIRED",images=527,meaning="No classification or lesion evaluation",
                 seen_85_replay="PASS",counts_by_group={g:dict(v) for g,v in actions.items()},
                 reasons=dict(reasons),known_broad_retained_cases=KNOWN_BROAD_HOLDS,other442_changed_ids=changed_new,
                 held_ids=held,all_442_previews_in_zip=True,model_forwards=0,optimizer_steps=0,
                 checkpoint_reads=0,aptos_test_images_read=0,idrid_images_read=0)
    json_new(run/"summary.json",summary)
    return review_zip(run,["summary.json","per_image.csv","protocol.json",*sheets])

def main():
    root=Path.cwd().resolve();src,rows=find_v2(root);run=start_run(root)
    print("New geometry-only run:",run,flush=True)
    try:packet=run_geometry(root,src,rows,run)
    except Exception as exc:
        json_new(run/"failure.json",dict(error=type(exc).__name__,detail=str(exc)))
        raise
    print("GEOMETRY COMPLETE | REVIEW REQUIRED | No A3 inference.\nReturn:",packet)

if __name__=="__main__":
    main()
