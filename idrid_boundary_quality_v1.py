"""CP2 section 3.10.2: fixed SLIC versus boundary-SLIC lesion-boundary recall.

Run in C:\\CP2: python idrid_boundary_quality_v1.py
Continue:       python idrid_boundary_quality_v1.py --resume

All 81 accepted IDRiD segmentation images. Existing CPU superpixel recipe,
target K=256. No classifier construction, checkpoint load, training or tuning.
Requires the existing CP2 scripts, including idrid_xai_v1.py, beside this file.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading IDRiD boundary-quality libraries...', flush=True)
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import torch
import idrid_xai_v1 as ix
import superpixels_v1 as sp
import boundary_map_v1 as bm

pp, ie, ev, audit = ix.pp, ix.ie, ix.ev, ix.audit
require, canonical, read_json, safe_path = ix.require, ix.canonical, ix.read_json, ix.safe_path
VERSION = 'idrid_boundary_quality_v1_20261108'
ROOT = Path(__file__).resolve().parent
METHODS = ('slic', 'boundary_slic')
LESIONS = ('MA', 'HE', 'EX', 'SE')
TARGETS = (*LESIONS, 'ALL4_COMPLETE')
DOMAINS = ('canvas_384', 'native_gt_projected_to_384')
RADII = (1, 2, 3)
PRIMARY_RADIUS = 2
PREVIEWS = ix.PREVIEWS
SOURCE_SHA_LF = {
    'superpixels_v1.py': '2d7e3dcd0631688bf88c4773714ae2a0b6c66dabecf2f372370a39f881707979',
    'boundary_map_v1.py': '0812cf16475865ca345627a96bdc2f78e868091040a793856ad0a9a55a0b263d',
    'idrid_preprocess_v1.py': '7b249e325e0e6bc3f11b7848f422aac03125ee1878e7ff6bf2dfba137b421a97',
    'idrid_xai_v1.py': 'dbe453bfc690ccf8e18b1a5240fc12329e32a5f88d2ada34eae3cab85564b9e4',
}
PROTOCOL = {
    'proposal': 'CP1 section3.10.2, Superpixel Boundary Quality; separate from classification and final-model XAI localisation',
    'cohort': 'All 81 accepted IDRiD segmentation images; source training54/testing27 retained for reporting; no parameter selection on IDRiD',
    'algorithm': 'Existing shared SLIC-style routine used in CP2; boundary multiplier gamma0 versus gamma2; no substitute segmentation library',
    'parameters': {'initial_regions':256, 'compactness':10., 'iterations':10, 'minimum_area_fraction':.25},
    'execution': 'CPU float32 Lab/boundary map, as in formal input preparation; deterministic ordered connectivity; up to four CPU threads',
    'actual_regions': 'Report final connected region counts; target K=256 does not guarantee exactly256; no retuning to equalise counts',
    'annotation_independence': 'Both segmentations completed from RGB before lesion masks are decoded; no annotations, DR labels or model predictions enter segmentation',
    'canvas': 'Full384x384 including original borders and synthetic padding, without FOV filtering',
    'superpixel_boundary': 'Both pixels of every differing horizontal/vertical label pair; no automatic outer-image-frame boundary',
    'lesion_boundary': 'Foreground pixels sharing a horizontal/vertical edge with observed background; no background fabricated beyond image frame',
    'tolerance': 'Chebyshev distance <=r pixels, square (2r+1)x(2r+1) neighbourhood on384canvas, clipped at frame; r2 primary, r1/r3 fixed sensitivity',
    'boundary_recall': 'Number of reference boundary pixels with a superpixel boundary in tolerance / reference boundary pixels; many-to-one matching allowed',
    'primary_domain': 'Boundary extracted from accepted NEAREST-resized384 binary masks',
    'native_auxiliary': 'Extract original-mask inner boundary; map in-crop pixel centres to384 using floor((coordinate-crop_origin+pad+0.5)*384/square_side); count each original boundary pixel. Outside-crop reference points always missed. Same384 tolerance, NOT native-pixel distance or a full-resolution segmentation.',
    'native_limit': 'Preserves original boundary denominator including crop exclusions and lesions lost at384; coordinate quantisation remains; no recovery of image detail',
    'missing': 'Missing masks unavailable, not zero negatives. ALL4_COMPLETE is the boundary of the union only when all4 masks exist; not the union of their individual boundary maps.',
    'empty': 'Empty reference or no observed foreground/background boundary yields an excluded status, not a perfect recall',
    'optic_disc': 'OD excluded; annotations are incomplete for retinal anatomy, so unmatched superpixel boundaries are not labelled false positives',
    'aggregation': 'Equal-image mean and median per type/domain/tolerance/source stratum; paired guided-minus-baseline difference, wins/ties/losses and pooled boundary counts; types kept separate',
    'context': 'Report actual K, superpixel boundary density and tolerance-neighbourhood canvas coverage; increased boundary density can increase recall. No claim of identical final region count or causal accuracy improvement.',
    'timing': 'Common Lab/gradient preparation timed separately; segmentation order alternates by sorted image index. Segmentation timers exclude mask scoring, validation and writes; descriptive timings, no repeated benchmark.',
    'previews': 'Same seven fixed cases as preceding IDRiD XAI stage, not selected by scores',
    'resume': 'Per-image receipt committed last, hashes checked on reuse; only unfinished image repeats; Windows-safe unique-temp atomic writes and OS-owned run lock',
    'result_policy': 'Report all fixed tolerances and both methods, including null/negative differences. No clinical pass threshold, hypothesis-test claim, model training or test-driven tuning.',
}
SOURCES = {
    'slic_and_boundary_recall': 'https://doi.org/10.1109/TPAMI.2012.120',
    'slic_paper_read': 'https://www.cs.jhu.edu/~ayuille/JHUcourses/VisionAsBayesianInference2020/4/s/Achanta_SLIC_PAMI2012.pdf',
    'boundary_recall_and_contour_density': 'https://davidstutz.github.io/superpixel-benchmark/md_docs_BENCHMARK.html',
    'benchmark_paper': 'https://doi.org/10.1016/j.cviu.2017.03.007',
    'IDRiD': 'https://doi.org/10.3390/data3030025',
    'accepted_superpixels': sp.SOURCES, 'accepted_gradient': bm.SOURCES,
    'accessed': '2026-11-08',
    'implementation': 'Existing CP2 segmentation APIs reused unchanged. New NumPy boundary extraction, square-neighbourhood matching, geometry projection and reporting are project code; no external benchmark source code copied.',
    'operational_choices': 'Inner/dual-sided4-neighbour boundaries, Chebyshev tolerance, r2 primary with r1/r3 sensitivity, complete-mask union and native projected diagnostic are declared project choices, not all mandated by CP1 or the cited papers.',
}


class Progress(ix.Progress):
    def show(self):
        n = self.done-self.initial
        eta = ev.clock((time.perf_counter()-self.phase_started)/n*(self.total-self.done)) if n else '--'
        if self.total and self.done == self.total: eta = '00:00:00'
        print(f'IDRiD boundaries | {self.stage} | Saved {self.done}/{self.total} | '
              f'Elapsed {ev.clock(time.perf_counter()-STARTED)} | Phase ETA {eta}', flush=True)


def region_boundary(labels):
    labels = np.asarray(labels)
    require(labels.ndim == 2 and min(labels.shape)>0, 'Expected a nonempty2D label image.')
    out = np.zeros(labels.shape, bool)
    d = labels[1:] != labels[:-1]; out[1:] |= d; out[:-1] |= d
    d = labels[:,1:] != labels[:,:-1]; out[:,1:] |= d; out[:,:-1] |= d
    return out


def lesion_boundary(mask):
    mask = np.asarray(mask, dtype=bool)
    return region_boundary(mask) & mask


def neighbourhood(boundary, radius):
    """Exact clipped Chebyshev dilation; no SciPy/OpenCV dependency."""
    require(isinstance(radius, int) and radius>=0, 'Invalid pixel tolerance.')
    boundary = np.asarray(boundary, dtype=bool)
    h,w = boundary.shape; padded = np.pad(boundary, radius)
    out = np.zeros_like(boundary)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1): out |= padded[dy:dy+h,dx:dx+w]
    return out


def reference_points(mask, row, native=False):
    """Retain original point multiplicity; absent crop support cannot be matched."""
    boundary = lesion_boundary(mask); y,x = np.nonzero(boundary)
    inside = np.ones(len(y), bool)
    if native:
        g = ix.geom_of(row)
        inside = ((x>=g['crop_left']) & (x<g['crop_right']) &
                  (y>=g['crop_top']) & (y<g['crop_bottom']))
        x = np.floor((x[inside]-g['crop_left']+g['pad_left']+.5)*384/g['square_side']).astype(np.int64)
        y = np.floor((y[inside]-g['crop_top']+g['pad_top']+.5)*384/g['square_side']).astype(np.int64)
        require(np.all((x>=0)&(x<384)&(y>=0)&(y<384)), 'Projection falls outside model canvas.')
    return dict(gt_pixels=int(np.count_nonzero(mask)), gt_boundary_pixels=len(inside),
                gt_boundary_inside_crop=int(inside.sum()), gt_boundary_outside_crop=int((~inside).sum()),
                y=y, x=x)


def load_references(ext, row):
    refs = {d:{} for d in DOMAINS}; small_masks = {}; availability = {}
    all_present = all(ext['masks'][row['sample_key'],t]['availability']=='present' for t in LESIONS)
    unions = {DOMAINS[0]:np.zeros((384,384),bool),
              DOMAINS[1]:np.zeros((int(row['height']),int(row['width'])),bool)} if all_present else {}
    for kind in LESIONS:
        m = ext['masks'][row['sample_key'],kind]; availability[kind] = m['availability']
        if m['availability']!='present': continue
        p = safe_path(ext['root'],m['processed_mask_path'])
        require(audit.sha_file(p)==m['processed_mask_sha256'], 'Processed annotation changed.')
        with Image.open(p) as im:
            im.load(); a = np.asarray(im)
            require(im.format=='PNG' and im.mode=='L' and im.size==(384,384) and
                    set(np.unique(a).tolist())<={0,255}, 'Invalid processed annotation.')
            small = a>0
        p = safe_path(ext['raw'],m['original_mask_path'])
        require(audit.sha_file(p)==m['original_mask_sha256'], 'Original annotation changed.')
        with Image.open(p) as im:
            im.load(); original = audit.binary_mask_array(im)[0].astype(bool)
        require(original.shape==(int(row['height']),int(row['width'])) and
                pp.pixel_hash(original,'IDRID_BINARY_SUPPORT_V1')==m['original_binary_support_sha256'],
                'Original annotation support differs.')
        require(int(original.sum())==int(m['positive_pixels_original']) and int(small.sum())==int(m['positive_pixels_384']),
                'Annotation positive counts differ.')
        g = ix.geom_of(row)
        outside = int(original.sum())-int(original[g['crop_top']:g['crop_bottom'],g['crop_left']:g['crop_right']].sum())
        require(outside==int(m['positive_pixels_outside_crop']), 'Crop exclusions changed.')
        small_masks[kind] = small
        for domain, mask in ((DOMAINS[0],small),(DOMAINS[1],original)):
            refs[domain][kind] = reference_points(mask,row,domain==DOMAINS[1])
            if all_present: unions[domain] |= mask
    if all_present:
        for domain,mask in unions.items(): refs[domain]['ALL4_COMPLETE'] = reference_points(mask,row,domain==DOMAINS[1])
    return refs, small_masks, availability


def generate_segmentations(rgb, index, progress):
    """Same CPU calls/default settings as train_v1.prepare_microbatch."""
    require(rgb.dtype==np.uint8 and rgb.shape==(384,384,3), 'Expected RGB384 uint8.')
    tensor = torch.from_numpy(rgb.copy()).permute(2,0,1).float().div_(255)
    before = tensor.clone(); start = time.perf_counter()
    with torch.no_grad(): details = bm.boundary_map(tensor,return_details=True)
    common_seconds = time.perf_counter()-start
    result = {}; order = METHODS if index%2==0 else METHODS[::-1]
    for method in order:
        def tick(i,n): progress.stage = f'{method} iteration {i}/{n}'
        with torch.no_grad():
            got = sp.segment_superpixels(details['lab'],details['boundary'][0],
                                         boundary_weight=2. if method=='boundary_slic' else 0.,progress=tick)
        sp.validate_regions(got)
        result[method] = got
    require(torch.equal(tensor,before), 'Segmentation modified the input RGB.')
    return result, details['boundary'][0].cpu().numpy(), common_seconds, list(order)


def score_maps(results, refs, row):
    scores = []; diagnostics = {}; hits_for_preview = {}
    for method in METHODS:
        labels = results[method]['labels']; edge = region_boundary(labels)
        diagnostics[method] = dict(actual_regions=int(np.unique(labels).size), boundary_pixels=int(edge.sum()),
            boundary_fraction=float(edge.mean()), segmentation_seconds=float(results[method]['stats']['elapsed_seconds']),
            generation_stats=results[method]['stats'])
        for radius in RADII:
            near = neighbourhood(edge,radius)
            if radius==PRIMARY_RADIUS: hits_for_preview[method] = near
            for domain in DOMAINS:
                for target in TARGETS:
                    ref = refs[domain].get(target)
                    status = ('incomplete_union_masks' if target=='ALL4_COMPLETE' else 'missing_mask') if ref is None else (
                        'empty_reference_in_domain' if ref['gt_pixels']==0 else (
                        'no_observed_boundary' if ref['gt_boundary_pixels']==0 else 'scored'))
                    match = int(near[ref['y'],ref['x']].sum()) if ref is not None else None
                    n = ref['gt_boundary_pixels'] if ref is not None else None
                    scores.append(dict(sample_key=row['sample_key'],official_source_partition=row['official_split'],
                        method=method,domain=domain,target=target,tolerance_pixels=radius,primary_tolerance=radius==PRIMARY_RADIUS,
                        status=status,target_regions=256,actual_regions=diagnostics[method]['actual_regions'],
                        superpixel_boundary_pixels=int(edge.sum()),superpixel_boundary_fraction=float(edge.mean()),
                        tolerance_neighbourhood_canvas_fraction=float(near.mean()),
                        gt_pixels=ref['gt_pixels'] if ref else None,gt_boundary_pixels=n,
                        gt_boundary_inside_crop=ref['gt_boundary_inside_crop'] if ref else None,
                        gt_boundary_outside_crop=ref['gt_boundary_outside_crop'] if ref else None,
                        matched_boundary_pixels=match,missed_boundary_pixels=n-match if n is not None else None,
                        boundary_recall=match/n if status=='scored' else None))
    return scores, diagnostics, hits_for_preview


def save_preview(path, rgb, results, masks, near, scores, row):
    font = ImageFont.load_default(size=17); smallfont = ImageFont.load_default(size=15)
    width,height = 384,424; canvas = Image.new('RGB',(width*4,height*3+70),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((8,6),row['sample_key']+' | fixed K=256 | boundary recall, tolerance r=2 (square5x5)',fill='black',font=font)
    draw.text((8,30),'Green: matched annotation boundary | Magenta: missed | Grey/white: superpixel boundary',fill='black',font=smallfont)
    def panel(col,line,arr,title,subtitle=''):
        x,y=col*width,70+line*height
        draw.text((x+5,y),title,fill='black',font=font);draw.text((x+5,y+21),subtitle,fill='black',font=smallfont)
        canvas.paste(Image.fromarray(arr),(x,y+40))
    panel(0,0,rgb,'Processed RGB','Full384canvas, including padding')
    annotated=rgb.copy()
    for kind,color in zip(LESIONS,((0,210,230),(255,50,50),(255,225,0),(60,255,80))):
        if kind in masks: annotated[masks[kind]]=color
    panel(1,0,annotated,'Available annotation support','MA cyan | HE red | EX yellow | SE green')
    for i,method in enumerate(METHODS):
        pic=rgb.copy();pic[region_boundary(results[method]['labels'])]=255
        panel(i+2,0,pic,method,f"Final regions: {results[method]['stats']['final_regions']}")
    for i,kind in enumerate(LESIONS):
        for j,method in enumerate(METHODS):
            col=(i%2)*2+j;line=i//2+1
            if kind not in masks:
                panel(col,line,np.full_like(rgb,238),kind+' | '+method,'Mask unavailable; not scored');continue
            pic=(rgb.astype(np.float32)*.8).astype(np.uint8)
            pic[region_boundary(results[method]['labels'])]=(150,150,150)
            edge=lesion_boundary(masks[kind]);pic[edge & ~near[method]]=(255,0,190);pic[edge & near[method]]=(0,255,90)
            r=next(s for s in scores if s['method']==method and s['domain']==DOMAINS[0] and s['target']==kind and s['primary_tolerance'])
            text=f"Recall: {r['boundary_recall']:.2%}" if r['boundary_recall'] is not None else r['status']
            panel(col,line,pic,kind+' | '+method,text)
    stream=io.BytesIO();canvas.save(stream,format='JPEG',quality=94);audit.atomic_bytes(path,stream.getvalue())


def receipt_path(output,row):
    return safe_path(output,'images/'+row['sample_key']+'/receipt_v1.json')


def read_receipt(output,row,signature):
    path=receipt_path(output,row)
    if not path.exists(): return None
    obj=read_json(path);digest=obj.pop('payload_sha256')
    require(canonical(obj)==digest and obj['signature']==signature and obj['input_row']==row, 'Saved receipt identity changed.')
    expected={(m,d,t,r) for m in METHODS for d in DOMAINS for t in TARGETS for r in RADII}
    require(len(obj['scores'])==len(expected) and {(s['method'],s['domain'],s['target'],s['tolerance_pixels']) for s in obj['scores']}==expected,
            'Incomplete saved image scores.')
    for rel,h in obj['files'].items(): require(audit.sha_file(safe_path(output,rel))==h, 'Saved evidence changed: '+rel)
    return obj


def process_one(output,ext,row,index,signature,progress):
    begun=time.perf_counter();progress.stage='read RGB '+row['sample_key']
    blob=safe_path(ext['root'],row['processed_rgb_path']).read_bytes()
    require(hashlib.sha256(blob).hexdigest()==row['processed_rgb_sha256'], 'RGB changed before decoding.')
    with Image.open(io.BytesIO(blob)) as im:
        im.load();require(im.format=='PNG' and im.mode=='RGB' and im.size==(384,384), 'Invalid RGB file.')
        rgb=np.asarray(im).copy()
    # Annotation decoding happens strictly after both methods have generated labels.
    result,boundary,common_seconds,order=generate_segmentations(rgb,index,progress)
    progress.stage='score annotations '+row['sample_key']
    refs,masks,availability=load_references(ext,row)
    scores,diagnostics,near=score_maps(result,refs,row)
    folder=receipt_path(output,row).parent;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'superpixel_evidence_v1.npz';stream=io.BytesIO()
    np.savez_compressed(stream,slic_labels=result['slic']['labels'],boundary_slic_labels=result['boundary_slic']['labels'],
                        boundary_strength=boundary)
    audit.atomic_bytes(path,stream.getvalue());files={path.relative_to(output).as_posix():audit.sha_file(path)}
    if row['sample_key'] in PREVIEWS:
        path=output/f"idrid_boundary_preview_{row['official_split']}_{row['canonical_id']}.jpg"
        save_preview(path,rgb,result,masks,near,scores,row);files[path.name]=audit.sha_file(path)
    obj=dict(version=VERSION,signature=signature,input_row=row,scores=scores,diagnostics=diagnostics,
        mask_availability=availability,masks_decoded_each_domain=len(masks),files=files,method_order=order,
        common_Lab_boundary_seconds=common_seconds,elapsed_seconds=time.perf_counter()-begun,model_forwards=0,optimizer_steps=0)
    audit.save_json(receipt_path(output,row),{**obj,'payload_sha256':canonical(obj)})
    return obj


def aggregate(records):
    rows=[s for rec in records for s in rec['scores']]
    paired=[];lookup={(s['sample_key'],s['method'],s['domain'],s['target'],s['tolerance_pixels']):s for s in rows}
    for rec in records:
        key=rec['input_row']['sample_key']
        for domain in DOMAINS:
            for target in TARGETS:
                for radius in RADII:
                    a,b=[lookup[key,m,domain,target,radius] for m in METHODS]
                    require(all(a[n]==b[n] for n in ('status','gt_pixels','gt_boundary_pixels','gt_boundary_outside_crop')), 'Paired reference mismatch.')
                    paired.append(dict(sample_key=key,official_source_partition=a['official_source_partition'],
                        domain=domain,target=target,tolerance_pixels=radius,primary_tolerance=radius==PRIMARY_RADIUS,status=a['status'],
                        slic_recall=a['boundary_recall'],boundary_slic_recall=b['boundary_recall'],
                        guided_minus_slic=b['boundary_recall']-a['boundary_recall'] if a['status']=='scored' else None,
                        gt_boundary_pixels=a['gt_boundary_pixels'],gt_boundary_outside_crop=a['gt_boundary_outside_crop'],
                        slic_matched_boundary_pixels=a['matched_boundary_pixels'],boundary_slic_matched_boundary_pixels=b['matched_boundary_pixels'],
                        slic_regions=a['actual_regions'],boundary_slic_regions=b['actual_regions'],
                        slic_boundary_fraction=a['superpixel_boundary_fraction'],boundary_slic_boundary_fraction=b['superpixel_boundary_fraction'],
                        slic_neighbourhood_fraction=a['tolerance_neighbourhood_canvas_fraction'],
                        boundary_slic_neighbourhood_fraction=b['tolerance_neighbourhood_canvas_fraction']))
    comparison=[]
    def mean(values): return float(np.mean(values)) if values else None
    def median(values): return float(np.median(values)) if values else None
    for domain in DOMAINS:
        for target in TARGETS:
            for radius in RADII:
                for part in ('all','training','testing'):
                    group=[r for r in paired if r['domain']==domain and r['target']==target and r['tolerance_pixels']==radius and
                           (part=='all' or r['official_source_partition']==part)]
                    good=[r for r in group if r['status']=='scored'];deltas=[r['guided_minus_slic'] for r in good]
                    denominator=sum(r['gt_boundary_pixels'] for r in good)
                    result=dict(domain=domain,target=target,tolerance_pixels=radius,primary_tolerance=radius==PRIMARY_RADIUS,
                        official_source_partition=part,total_images=len(group),paired_scored_images=len(good),
                        excluded_status_counts=ix.json.dumps(dict(Counter(r['status'] for r in group if r['status']!='scored')),sort_keys=True),
                        mean_paired_difference=mean(deltas),median_paired_difference=median(deltas),
                        guided_wins=sum(v>1e-12 for v in deltas),ties=sum(abs(v)<=1e-12 for v in deltas),guided_losses=sum(v< -1e-12 for v in deltas),
                        total_gt_boundary_pixels=denominator,total_gt_boundary_outside_crop=sum(r['gt_boundary_outside_crop'] for r in good))
                    for m in METHODS:
                        result.update({m+'_macro_mean_recall':mean([r[m+'_recall'] for r in good]),
                            m+'_median_recall':median([r[m+'_recall'] for r in good]),
                            m+'_pooled_recall':sum(r[m+'_matched_boundary_pixels'] for r in good)/denominator if denominator else None,
                            m+'_mean_regions':mean([r[m+'_regions'] for r in good]),
                            m+'_mean_boundary_fraction':mean([r[m+'_boundary_fraction'] for r in good]),
                            m+'_mean_neighbourhood_fraction':mean([r[m+'_neighbourhood_fraction'] for r in good])})
                    comparison.append(result)
    return rows,paired,comparison


def publish(output,records,config,decoded,reused):
    rows,paired,comparison=aggregate(records);artifacts={}
    for name,data in (('idrid_boundary_per_image_v1.csv',rows),('idrid_boundary_paired_v1.csv',paired),
                      ('idrid_boundary_comparison_v1.csv',comparison)):
        artifacts[name]=ix.write_table(output,name,data)
    diagnostics=[dict(sample_key=r['input_row']['sample_key'],method=m,**{k:v for k,v in r['diagnostics'][m].items() if k!='generation_stats'},
        common_Lab_boundary_seconds=r['common_Lab_boundary_seconds'],run_order=r['method_order'].index(m)+1) for r in records for m in METHODS]
    artifacts['idrid_boundary_region_counts_v1.csv']=ix.write_table(output,'idrid_boundary_region_counts_v1.csv',diagnostics)
    inventory={receipt_path(output,r['input_row']).relative_to(output).as_posix():audit.sha_file(receipt_path(output,r['input_row'])) for r in records}
    for r in records:inventory.update(r['files'])
    for name,obj in (('idrid_boundary_files_v1.json',inventory),('method_sources_v1.json',SOURCES)):
        audit.save_json(output/name,obj);artifacts[name]=audit.sha_file(output/name)
    counts={m:dict(min=min(r['diagnostics'][m]['actual_regions'] for r in records),max=max(r['diagnostics'][m]['actual_regions'] for r in records),
                  mean=float(np.mean([r['diagnostics'][m]['actual_regions'] for r in records]))) for m in METHODS}
    summary=dict(status='IDRID_BOUNDARY_QUALITY_COMPLETE',version=VERSION,signature=config['signature'],images=len(records),
        source_partitions=dict(Counter(r['input_row']['official_split'] for r in records)),protocol=PROTOCOL,target_regions=256,actual_regions=counts,
        primary_tolerance_results=[r for r in comparison if r['primary_tolerance'] and r['official_source_partition']=='all'],
        image_commits_reused_this_process=reused,RGB_decoded_this_process=decoded,segmentations_this_process=2*decoded,
        logical_segmentations=2*len(records),lesion_masks_decoded_each_domain_in_committed_images=sum(r['masks_decoded_each_domain'] for r in records),
        absent_lesion_entries_not_imputed=sum(v!='present' for r in records for v in r['mask_availability'].values()),
        per_image_rows=len(rows),paired_rows=len(paired),comparison_rows=len(comparison),
        classifier_constructed=False,checkpoints_loaded=0,model_forwards=0,optimizer_steps=0,APTOS_RGB_decoded=0,IDRiD_grading_RGB_decoded=0,
        optic_disc_masks_scored=0,upstream_files_unchanged=True,elapsed_seconds_this_process=time.perf_counter()-STARTED,
        committed_image_seconds=sum(r['elapsed_seconds'] for r in records),runtime=config['identity']['runtime'],
        output_sha256=artifacts,saved=str(output),conclusion='Execution complete; interpret the paired scores, actual region counts, density and resolution limits. No quality pass threshold asserted.')
    audit.save_json(output/'idrid_boundary_summary_v1.json',summary)
    audit.save_json(output/'idrid_boundary_state_v1.json',dict(status='COMPLETE',signature=config['signature'],
                    summary_sha256=audit.sha_file(output/'idrid_boundary_summary_v1.json')))
    return summary


def execute(root,output,progress):
    audit.load_libraries();torch.set_num_threads(min(4,torch.get_num_threads()))
    modules=(sp,bm,pp,ix)
    for module in modules:
        path=Path(module.__file__).resolve()
        require(pp.code_hash(path)==SOURCE_SHA_LF[path.name], 'Accepted source differs: '+path.name)
    require(pp.function_hash(audit.binary_mask_array)==pp.MASK_DECODER_SHA, 'Reviewed mask decoder differs.')
    ext=ix.load_inputs(root,progress)
    sources=dict(ext['sources'])
    for module in (*modules,ie,audit): sources[Path(module.__file__).resolve()]=audit.sha_file(Path(module.__file__))
    sources[Path(__file__).resolve()]=audit.sha_file(Path(__file__))
    identity=dict(version=VERSION,protocol=PROTOCOL,superpixel_settings=sp.SETTINGS,boundary_settings=bm.SETTINGS,
        sources={str(p):h for p,h in sources.items()},output=str(output),
        runtime={**ev.runtime_versions(),'pillow':PIL.__version__,'device':'cpu','torch_threads':torch.get_num_threads()})
    config=dict(identity=identity,signature=canonical(identity),created_utc=datetime.now(timezone.utc).isoformat())
    # Keep every earlier audit/training/evaluation directory protected, including XAI.
    audit_root=root/'environment/data_audit'
    protected=[root/'IDRiD',root/'train_images',root/'processed_images_384_v1']
    if audit_root.exists():protected.extend(p for p in audit_root.iterdir() if p.name!='idrid_boundary_quality_v1')
    pp.protect_output(output,protected);output.mkdir(parents=True,exist_ok=True)
    with ev.RunLock(output):
        path=output/'idrid_boundary_config_v1.json'
        if path.exists():
            saved=read_json(path)
            require(saved['signature']==canonical(saved['identity'])==config['signature'], 'Run identity changed; do not mix results.')
            config=saved
        else:
            require(not any(p.name!='evaluation.lock' for p in output.iterdir()), 'Unrecognised output folder contents.')
            audit.save_json(path,config)
        records=[];progress.reset('verify saved image commits',len(ext['rows']))
        for row in ext['rows']:records.append(read_receipt(output,row,config['signature']));progress.done+=1
        reused=sum(r is not None for r in records);statepath=output/'idrid_boundary_state_v1.json'
        if statepath.exists() and read_json(statepath)['status']=='COMPLETE':
            state=read_json(statepath)
            require(reused==len(records) and state['signature']==config['signature'] and
                    audit.sha_file(output/'idrid_boundary_summary_v1.json')==state['summary_sha256'], 'Completed run changed.')
            summary=read_json(output/'idrid_boundary_summary_v1.json')
            for name,h in summary['output_sha256'].items(): require(audit.sha_file(output/name)==h, 'Completed table changed: '+name)
            print('Already complete: verified committed outputs; no new segmentation.',flush=True)
            return summary
        audit.save_json(statepath,dict(status='RUNNING',signature=config['signature']))
        decoded=0;progress.reset('compare two segmentations per image',len(records),reused)
        for index,row in enumerate(ext['rows']):
            if records[index] is not None:continue
            records[index]=process_one(output,ext,row,index,config['signature'],progress);decoded+=1;progress.done+=1
            a,b=[records[index]['diagnostics'][m]['actual_regions'] for m in METHODS]
            print(f"{row['sample_key']} | saved {progress.done}/{len(records)} | SLIC regions={a} | boundary-SLIC regions={b}",flush=True)
        ie.verify_unchanged(sources,progress)
        progress.reset('save paired comparisons and summary')
        return publish(output,records,config,decoded,reused)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--resume',action='store_true',help='Reuse verified image commits; default is also restart-safe.')
    args=parser.parse_args();root=args.project_root.resolve()
    output=(args.output_dir or root/'environment/data_audit/idrid_boundary_quality_v1/comparison_K256_v1').resolve()
    try:
        with Progress() as progress:summary=execute(root,output,progress)
    except (Exception,KeyboardInterrupt) as exc:
        print(f'\nIDRID BOUNDARY QUALITY STOPPED | {type(exc).__name__}: {exc}')
        print('Committed images retained. Resolve the issue, then: python idrid_boundary_quality_v1.py --resume')
        return 1
    print('\nIDRID BOUNDARY QUALITY COMPLETE | 81 images | 162 segmentations | target K=256')
    print('CPU | same accepted segmentation implementation | primary tolerance: 2 pixels (square5x5)')
    for r in summary['primary_tolerance_results']:
        if r['domain']!=DOMAINS[0]:continue
        fmt=lambda v:'--' if v is None else f'{v:.5f}'
        print(f"{r['target']} | n={r['paired_scored_images']}/81 | SLIC recall={fmt(r['slic_macro_mean_recall'])} | boundary-SLIC recall={fmt(r['boundary_slic_macro_mean_recall'])} | paired difference={fmt(r['mean_paired_difference'])}")
    print('Classifier forwards: 0 | Optimizer steps: 0 | Checkpoints loaded: 0 | Upstream unchanged: PASS')
    print(f'Elapsed this process: {ev.clock(time.perf_counter()-STARTED)}\nSaved: {output}')
    print('Return idrid_boundary_summary_v1.json, idrid_boundary_comparison_v1.csv, idrid_boundary_paired_v1.csv and idrid_boundary_preview_*.jpg.')
    print('After result review: remember to commit idrid_boundary_quality_v1.py; keep images/results out of Git.')
    return 0


if __name__=='__main__':sys.exit(main())
