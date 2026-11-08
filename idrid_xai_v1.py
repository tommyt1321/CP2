"""CP2: frozen A3 Attention Rollout localisation on all 81 IDRiD segmentation images.

Run in C:\\CP2: python idrid_xai_v1.py
Continue:       python idrid_xai_v1.py --resume

Reuses the accepted model and rollout APIs. No training or threshold fitting.
Outputs are committed per image; default execution is also restart-safe.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading IDRiD XAI-localisation libraries...', flush=True)
import numpy as np
import PIL
from PIL import Image
import torch
import attention_rollout_v1 as ar
import idrid_evaluate_v1 as ie
import idrid_preprocess_v1 as pp

ev, tr, d, audit, require = ar.ev, ar.tr, ar.d, ie.audit, ar.require
read_json, canonical, safe_path, checked_file = ie.read_json, ie.canonical, ie.safe_path, ie.checked_file
VERSION = 'idrid_xai_v1_20261107'
ROOT = Path(__file__).resolve().parent
EXTERNAL_SHA = 'f108b459aa565dd043d034a5b91a7971c2af0fbf7e442fab2e854ae7f561d9e7'
ROLLOUT_SHA_LF = 'abf98a5f174d8e7730f52873f630d80a548549e99c4e3701f9d9e5aa882ae98e'
BUDGETS = (1, 5, 10, 20)
PRIMARY_BUDGET = 10
RANDOM_DRAWS = 20
LESIONS = ('MA', 'HE', 'EX', 'SE')
TARGETS = (*LESIONS, 'ALL4_COMPLETE')
DOMAINS = ('canvas_384', 'native_original')
PREVIEWS = (*pp.PREVIEW_KEYS, 'segmentation/testing/IDRiD_61')
METRICS = ('dice', 'iou', 'component_hit_rate', 'pixel_recall')
GEOMETRY = ('crop_left', 'crop_top', 'crop_right', 'crop_bottom', 'pad_left',
            'pad_top', 'pad_right', 'pad_bottom', 'square_side')
PROTOCOL = {
    'model': 'A3 validation-selected epoch21; frozen eval; existing rollout API; no optimiser or tuning',
    'explanation': 'head mean, residual identity, row normalisation, forward-ordered attention rollout; CLS-to-region mass',
    'interpretation': 'class-agnostic region relevance, not a class-specific lesion prediction or clinical explanation',
    'cohort': 'all 81 frozen segmentation records; official source training54/testing27 retained as reporting strata',
    'budgets_percent_regions': list(BUDGETS), 'primary_budget_percent_regions': PRIMARY_BUDGET,
    'selection': 'ceil(budget*R/100) whole superpixels; descending raw rollout mass, ascending region ID breaks ties; never select using annotations',
    'coverage': 'report actual pixel coverage; a percentage of regions is not a percentage of pixels',
    'primary_domain': 'entire 384x384 canvas including black image borders and synthetic padding; no FOV filtering',
    'native_auxiliary': 'NEAREST lift selected 384 mask to recorded square; remove padding; paste into original crop; zero outside crop; score all original GT pixels including IDRiD_61 HE stripe',
    'native_limit': 'synthetic padding has no native coordinate; native scores cannot replace full-canvas scores or recover lost visual detail',
    'missing': 'absent masks are unavailable, never all-zero negatives; ALL4_COMPLETE union only when all four lesion masks exist',
    'per_type': 'each provided type scored separately against its own annotation; overlap with another type is not a TP for this type',
    'optic_disc': 'OD excluded from lesion union and scores',
    'empty': 'zero-positive reference in a domain excluded from positive-case Dice/IoU/hit averages; status/denominator retained',
    'zero_flow': 'no ranked localisation scores if rollout has zero regional mass; do not invent an informative ranking',
    'dice': '2*TP/(selected_pixels+GT_pixels)', 'iou': 'TP/(selected_pixels+GT_pixels-TP)',
    'component_hit_rate': '8-connected GT components hit by at least one selected pixel / all GT components; proxy lesion instances, not clinician-counted lesions',
    'hit_limit': 'one-pixel overlap is a lenient hit; components can merge/split/disappear after resizing; read with pixel recall, Dice and IoU',
    'random_control': f'{RANDOM_DRAWS} fixed seeded uniform region permutations/image, same k, nested budgets; canvas384 only; not area-matched; no model reruns',
    'aggregation': 'equal-image macro mean and median per domain/type/budget/source stratum; pooled hit counts are secondary; random draws averaged within image before paired differences',
    'reporting': 'descriptive evidence; no significance/causal/clinical validity claim and no best-budget selection',
    'previews': 'six previously fixed preprocessing cases plus documented IDRiD_61 crop case; not selected for localisation success',
    'resume': 'atomic unique temporary files with Windows retry; hashed per-image receipt saved last; unfinished image may repeat; no interrupt checkpoint proliferation',
    'proposal': 'CP1 section3.10.4 clinical localisation; superpixel lesion-boundary recall in3.10.2 remains a separate stage',
}
SOURCES = {
    'existing_rollout_method_sources': ar.SOURCES,
    'rollout_paper': 'https://aclanthology.org/2020.acl-main.385/',
    'overlap_metric_guidance': 'https://doi.org/10.1038/s41592-023-02151-z',
    'metric_pitfalls': 'https://doi.org/10.1038/s41592-023-02150-0',
    'IDRiD_dataset': 'https://doi.org/10.3390/data3030025',
    'accessed': '2026-11-07',
    'implementation': 'Project-specific NumPy run-length/union-find 8-connectivity, overlap arithmetic, inverse geometry and reporting; no external metric implementation copied.',
    'project_choices': 'Budgets, primary10%, complete-mask union, 8-connected component hit definition, empty-case handling, inverse projection and random-control policy are explicit project operational choices, not mandated by cited papers.',
}


class Progress(ar.Progress):
    def refresh(self):
        while not self.stop.wait(5):
            self.show()

    def show(self):
        n = self.done - self.initial
        eta = ev.clock((time.perf_counter()-self.phase_started)/n*(self.total-self.done)) if n else '--'
        if self.total and self.done == self.total:
            eta = '00:00:00'
        print(f'IDRiD XAI | {self.stage} | Saved {self.done}/{self.total} | '
              f'Elapsed {ev.clock(time.perf_counter()-STARTED)} | Phase ETA {eta}', flush=True)


def geom_of(row):
    geom = {k: int(row[k]) for k in GEOMETRY}
    size = (int(row['width']), int(row['height']))
    expected = pp.geometry(size, tuple(geom[k] for k in GEOMETRY[:4]))
    require(all(geom[k] == expected[k] for k in GEOMETRY), 'Invalid saved crop/pad geometry.')
    require(math.isclose(float(row['scale_to_384']), expected['scale_to_384'], abs_tol=1e-12), 'Invalid saved scale.')
    return expected


def load_inputs(root, progress):
    """Verify only the segmentation allowlist, related masks, and frozen metadata."""
    logs = root/'environment/data_audit/idrid_preprocessing_v1/aligned_384_v1'
    final = root/'environment/data_audit/idrid_final_cohort_v1/final_aptos_idrid_v1'
    images, raw, sources = root/'IDRiD/processed_384_v1', root/'IDRiD/raw', {}
    summary = read_json(checked_file(logs/'idrid_preprocessing_summary_v1.json', ie.PREPROCESS_SHA, sources))
    require(summary['segmentation'] == 81 and summary['processed_existing_masks'] == 363
            and summary['same_APTOS_RGB_function'] is True, 'Wrong accepted preprocessing summary.')
    statepath = logs/'preprocessing_state_v1.json'
    state = read_json(statepath)
    require(state['status'] == 'COMPLETE' and state['summary_sha256'] == ie.PREPROCESS_SHA
            and state['signature'] == summary['signature'], 'Preprocessing is not complete.')
    sources[statepath] = audit.sha_file(statepath)
    names = ('idrid_segmentation_384_v1.csv', 'idrid_mask_manifest_384_v1.csv',
             'idrid_preprocessing_config_v1.json', 'idrid_preprocessing_files_v1.json',
             'idrid_preprocessing_review_v1.csv')
    for name in names:
        checked_file(logs/name, summary['output_sha256'][name], sources)
    cfg = read_json(logs/names[2])
    require(canonical({k:v for k,v in cfg.items() if k != 'signature'}) == cfg['signature'] == summary['signature'],
            'Preprocessing identity changed.')
    require(cfg['image_root'] == str(images) and cfg['pillow'] == PIL.__version__, 'Image root/Pillow differs from accepted preprocessing.')
    fs = read_json(checked_file(final/'idrid_final_cohort_summary_v1.json', ie.FINAL_COHORT_SHA, sources))
    name = 'idrid_segmentation_primary_v1.csv'
    original = d.read_csv(checked_file(final/name, fs['output_sha256'][name], sources))
    rows = d.read_csv(logs/names[0]); masks = d.read_csv(logs/names[1])
    by_original = {r['sample_key']:r for r in original}
    require(len(rows) == len(original) == len(by_original) == 81 and
            {r['sample_key'] for r in rows} == set(by_original), 'Segmentation cohort changed.')
    require(Counter(r['official_split'] for r in rows) == {'training':54, 'testing':27}, 'Source partitions changed.')
    require(sum(r['dr_grade'] == '' for r in rows) == 78 and sum(bool(r['grading_sample_key']) for r in rows) == 3,
            'Known/unknown DR links changed; do not infer grades from IDs.')
    lookup = {(r['sample_key'], r['mask_type']):r for r in masks}
    require(len(masks) == len(lookup) == 405 and set(lookup) == {(k,t) for k in by_original for t in (*LESIONS,'OD')},
            'Mask manifest membership changed.')
    require(Counter(r['mask_type'] for r in masks if r['availability'] == 'present') ==
            {'MA':81, 'HE':80, 'EX':81, 'SE':40, 'OD':81}, 'Present-mask counts changed.')
    require(Counter(r['availability'] for r in masks) == {'present':363,'absent_not_inferred':42}, 'Mask availability changed.')
    issues = d.read_csv(logs/names[4])
    require(issues == [{'sample_key':'segmentation/testing/IDRiD_61', 'mask_type':'HE',
                       'issue':'MASK_FOREGROUND_OUTSIDE_RGB_CROP', 'details':'48416'}], 'Unreviewed preprocessing issue.')
    inventory = read_json(logs/names[3])
    require(inventory['image_root'] == str(images), 'Image inventory root changed.')
    progress.reset('verify segmentation RGB and lesion mask hashes (no RGB decoding)', 81+2*282)
    for r in rows:
        key = r['sample_key']
        require(key == f"segmentation/{r['official_split']}/{r['canonical_id']}" and
                all(r.get(k) == v for k,v in by_original[key].items()), 'Segmentation provenance changed.')
        require(r['segmentation_primary'] == 'True' and r['primary_classification'] == r['sensitivity_classification'] == 'False',
                'Wrong cohort flags.')
        geom_of(r)
        require(r['processed_rgb_path'] == 'rgb/'+key+'.png' and r['output_width'] == r['output_height'] == '384',
                'Unexpected RGB path/resolution.')
        require(inventory['image_files'][r['processed_rgb_path']] == r['processed_rgb_sha256'], 'RGB inventory mismatch.')
        checked_file(safe_path(images,r['processed_rgb_path']),r['processed_rgb_sha256'],sources); progress.done += 1
        for kind in (*LESIONS, 'OD'):
            m = lookup[key,kind]
            require(m['role'] == ('anatomy' if kind == 'OD' else 'lesion') and
                    m['availability'] == r[kind+'_availability'] and m['original_mask_path'] == r[kind+'_mask_path'],
                    'Mask role/availability/link changed.')
            if m['availability'] != 'present':
                require(not m['original_mask_path'] and not m['processed_mask_path'], 'Missing mask was imputed.')
                continue
            require(all(int(m[k]) == int(r[k]) for k in GEOMETRY), 'RGB/mask geometry differs.')
            require(m['processed_mask_path'] == 'masks/'+key+'_'+kind+'.png' and
                    inventory['image_files'][m['processed_mask_path']] == m['processed_mask_sha256'], 'Mask inventory mismatch.')
            if kind != 'OD':
                checked_file(safe_path(images,m['processed_mask_path']),m['processed_mask_sha256'],sources)
                checked_file(safe_path(raw,m['original_mask_path']),m['original_mask_sha256'],sources)
                progress.done += 2
    outside = [m for m in masks if m['availability']=='present' and int(m['positive_pixels_outside_crop'])>0]
    require(len(outside)==1 and outside[0]['sample_key']=='segmentation/testing/IDRiD_61'
            and outside[0]['mask_type']=='HE' and int(outside[0]['positive_pixels_outside_crop'])==48416,
            'Crop-excluded annotation coverage changed.')
    path = root/'environment/data_audit/idrid_evaluation_v1/external_A0_A3_v1/idrid_evaluation_summary_v1.json'
    external = read_json(checked_file(path, EXTERNAL_SHA, sources))
    require(external['checkpoints']['A3'] == {'sha256':ie.CHECKPOINT_SHA['A3'], 'epoch':21}
            and external['optimizer_steps']==0 and external['parameters_and_BN_buffers_unchanged'] is True,
            'External evaluation/checkpoint anchor differs.')
    return dict(root=images, raw=raw, rows=sorted(rows,key=lambda r:r['sample_key']), masks=lookup,
                sources=sources, summary=summary, external=external)


class SegmentationDataset(ie.ExternalDataset):
    def __getitem__(self, index):
        row = self.rows[index]
        require(row['sample_key'].startswith('segmentation/'), 'Only frozen segmentation images may be read.')
        blob = safe_path(self.root,row['processed_rgb_path']).read_bytes()
        require(hashlib.sha256(blob).hexdigest()==row['processed_rgb_sha256'], 'RGB changed before decoding.')
        with Image.open(io.BytesIO(blob)) as im:
            im.load()
            require(im.format=='PNG' and im.mode=='RGB' and im.size==(384,384), 'Expected RGB384 PNG.')
            rgb = self.to_rgb(im).contiguous()
        self.decoded += 1
        return dict(rgb=rgb,cnn=self.normalise(rgb).contiguous(),
                    label=int(row['dr_grade']) if row['dr_grade'] else -1,
                    id_code=row['sample_key'],split='external_segmentation')


def components(mask):
    """Run-length 8-connected components; memory proportional to foreground runs."""
    mask = np.asarray(mask, dtype=bool)
    require(mask.ndim==2, 'Expected a 2D binary reference.')
    ys, starts, ends, parents, ranks = [], [], [], [], []
    previous = []
    def find(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]; a = parents[a]
        return a
    def union(a,b):
        a,b = find(a),find(b)
        if a == b: return
        if ranks[a] < ranks[b]: a,b = b,a
        parents[b] = a
        if ranks[a] == ranks[b]: ranks[a] += 1
    for y,line in enumerate(mask):
        change = np.flatnonzero(np.diff(np.pad(line.astype(np.int8),(1,1))))
        current = []; j = 0
        for s,e in zip(change[::2],change[1::2]):
            s,e = int(s),int(e); lab = len(parents)
            parents.append(lab); ranks.append(0); ys.append(y); starts.append(s); ends.append(e)
            while j < len(previous) and previous[j][1] < s: j += 1
            k = j
            while k < len(previous) and previous[k][0] <= e:
                union(lab,previous[k][2]); k += 1
            current.append((s,e,lab))
        previous = current
    roots = np.asarray([find(i) for i in range(len(parents))],dtype=np.int64)
    unique,ids = np.unique(roots,return_inverse=True)
    return dict(shape=mask.shape,y=np.asarray(ys,dtype=np.int32),start=np.asarray(starts,dtype=np.int32),
                end=np.asarray(ends,dtype=np.int32),ids=ids,count=len(unique),pixels=int(mask.sum()))


def measure(selected, refs):
    """Return counts/metrics against one or more run-length references."""
    selected = np.asarray(selected,dtype=bool)
    n = int(selected.sum())
    prefix = np.zeros((selected.shape[0],selected.shape[1]+1),dtype=np.int32)
    np.cumsum(selected,axis=1,dtype=np.int32,out=prefix[:,1:])
    result = {}
    for kind,ref in refs.items():
        require(selected.shape == ref['shape'], 'Prediction/reference grid mismatch.')
        p = ref['pixels']
        overlaps = prefix[ref['y'],ref['end']] - prefix[ref['y'],ref['start']]
        tp = int(overlaps.sum())
        hits = int(np.count_nonzero(np.bincount(ref['ids'],weights=overlaps,minlength=ref['count'])))
        result[kind] = dict(status='scored' if p else 'empty_reference_in_domain', selected_pixels=n,
            gt_pixels=p,tp=tp,fp=n-tp,fn=p-tp,gt_components=ref['count'],hit_components=hits,
            dice=2*tp/(n+p) if p else None,iou=tp/(n+p-tp) if p else None,
            component_hit_rate=hits/ref['count'] if ref['count'] else None,pixel_recall=tp/p if p else None)
    return result


def lift_to_native(selected, row):
    """Map an explanation back; outside the observed crop remains prediction zero."""
    g = geom_of(row)
    require(selected.shape == (384,384), 'Expected 384x384 selected-region mask.')
    square = np.asarray(Image.fromarray(selected.astype(np.uint8)*255).resize(
        (g['square_side'],g['square_side']),Image.Resampling.NEAREST)) > 0
    w,h = g['crop_right']-g['crop_left'],g['crop_bottom']-g['crop_top']
    result = np.zeros((int(row['height']),int(row['width'])),dtype=bool)
    result[g['crop_top']:g['crop_bottom'],g['crop_left']:g['crop_right']] = square[
        g['pad_top']:g['pad_top']+h,g['pad_left']:g['pad_left']+w]
    return result


def read_references(ext,row):
    refs = {domain:{} for domain in DOMAINS}; displays = {}; availability = []
    unions = {'canvas_384':np.zeros((384,384),bool),
              'native_original':np.zeros((int(row['height']),int(row['width'])),bool)}
    all_present = all(ext['masks'][row['sample_key'],t]['availability']=='present' for t in LESIONS)
    for kind in LESIONS:
        m = ext['masks'][row['sample_key'],kind]
        availability.append(dict(mask_type=kind,availability=m['availability']))
        if m['availability'] != 'present': continue
        p = safe_path(ext['root'],m['processed_mask_path'])
        require(audit.sha_file(p)==m['processed_mask_sha256'], 'Processed mask changed before scoring.')
        with Image.open(p) as im:
            im.load(); arr = np.asarray(im)
            require(im.format=='PNG' and im.mode=='L' and im.size==(384,384) and
                    set(np.unique(arr).tolist()) <= {0,255}, 'Invalid processed binary mask.')
            small = arr>0
        p = safe_path(ext['raw'],m['original_mask_path'])
        require(audit.sha_file(p)==m['original_mask_sha256'], 'Original mask changed before scoring.')
        with Image.open(p) as im:
            im.load(); original,_,_ = audit.binary_mask_array(im)
            original = original.astype(bool)
        require(original.shape==(int(row['height']),int(row['width'])) and
                pp.pixel_hash(original,'IDRID_BINARY_SUPPORT_V1')==m['original_binary_support_sha256'], 'Original support differs.')
        require(int(original.sum())==int(m['positive_pixels_original']) and
                int(small.sum())==int(m['positive_pixels_384']), 'Ground-truth positive counts differ.')
        g = geom_of(row)
        inside = int(original[g['crop_top']:g['crop_bottom'],g['crop_left']:g['crop_right']].sum())
        require(int(original.sum())-inside==int(m['positive_pixels_outside_crop']), 'Original crop exclusion changed.')
        displays[kind] = small
        for domain,arr in (('canvas_384',small),('native_original',original)):
            refs[domain][kind] = components(arr)
            if all_present: unions[domain] |= arr
    if all_present:
        for domain in DOMAINS: refs[domain]['ALL4_COMPLETE'] = components(unions[domain])
    return refs,displays,availability


def selection_orders(evidence,key):
    scores = np.asarray(evidence['region_raw'])
    require(scores.ndim==1 and np.isfinite(scores).all() and (scores>=0).all(), 'Invalid region scores.')
    n = len(scores)
    require(np.array_equal(np.unique(evidence['labels']),np.arange(n)), 'Superpixel IDs are not aligned.')
    ranked = np.lexsort((np.arange(n),-scores))
    seed = int.from_bytes(hashlib.sha256(('IDRID_XAI_REGION_CONTROL_V1\0'+key).encode()).digest()[:8],'little')
    rng = np.random.Generator(np.random.PCG64(seed))
    random = np.stack([rng.permutation(n) for _ in range(RANDOM_DRAWS)])
    return ranked,random,seed


def score_explanation(evidence,row,refs):
    labels = evidence['labels']; n = len(evidence['region_raw'])
    ranked,random,seed = selection_orders(evidence,row['sample_key'])
    scores,controls = [],[]
    padding = ~np.asarray(pp.transform_mask(np.ones((int(row['height']),int(row['width'])),bool),geom_of(row))[0],dtype=bool)
    # This geometry-only padding indicator is reported, never used to filter scores.
    for budget in BUDGETS:
        k = math.ceil(n*budget/100)
        chosen = np.isin(labels,ranked[:k]) if not evidence['zero_region_flow'] else np.zeros_like(labels,dtype=bool)
        base = dict(sample_key=row['sample_key'],official_split=row['official_split'],budget_percent_regions=budget,
                    requested_regions=k,selected_regions=k if not evidence['zero_region_flow'] else 0,
                    total_regions=n,canvas_selected_pixels=int(chosen.sum()),
                    canvas_selected_fraction=float(chosen.mean()),selected_synthetic_padding_pixels=int((chosen&padding).sum()))
        for domain in DOMAINS:
            metrics = measure(chosen if domain=='canvas_384' else lift_to_native(chosen,row),refs[domain])
            for kind in TARGETS:
                reason = 'missing_mask' if kind in LESIONS else 'incomplete_four_type_annotation'
                m = metrics.get(kind,{'status':reason})
                if evidence['zero_region_flow']:
                    m = {**m,'status':'zero_rollout_region_flow',**{name:None for name in METRICS}}
                scores.append({**base,'domain':domain,'target':kind,**m})
        for draw,order in enumerate(random):
            selected = np.isin(labels,order[:k])
            for kind,m in measure(selected,refs['canvas_384']).items():
                controls.append(dict(sample_key=row['sample_key'],official_split=row['official_split'],
                    budget_percent_regions=budget,selected_regions=k,draw=draw,seed=str(seed),target=kind,
                    canvas_selected_fraction=float(selected.mean()),**m))
    return scores,controls,ranked,random


def png_bytes(arr):
    out = io.BytesIO(); Image.fromarray(arr).save(out,format='PNG'); return out.getvalue()


def save_preview(path,evidence,row,displays,ranked):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rgb = evidence['rgb_uint8']; labels = evidence['labels']; k = math.ceil(len(ranked)*PRIMARY_BUDGET/100)
    selected = np.isin(labels,ranked[:k]) if not evidence['zero_region_flow'] else np.zeros_like(labels,dtype=bool)
    fig,axes = plt.subplots(2,4,figsize=(14,8),layout='constrained')
    for ax in axes.flat: ax.set_axis_off()
    axes[0,0].imshow(rgb); axes[0,0].set_title('Processed RGB | full canvas')
    axes[0,1].imshow(rgb); axes[0,1].imshow(evidence['pixel_display'],cmap='magma',vmin=0,vmax=1,alpha=.55)
    axes[0,1].set_title('Attention Rollout | relative relevance')
    axes[0,2].imshow(selected,cmap='gray',vmin=0,vmax=1)
    axes[0,2].set_title(f'Top {PRIMARY_BUDGET}% regions: {k}/{len(ranked)}\nPixel coverage: {selected.mean():.1%}')
    if evidence['zero_region_flow']: axes[0,2].set_title('Zero regional flow\nNo ranked localisation scored')
    axes[0,3].imshow(rgb)
    colours = {'MA':(0,1,1),'HE':(1,0,0),'EX':(1,1,0),'SE':(0,1,0)}
    for kind,gt in displays.items():
        colour = np.zeros((*gt.shape,4)); colour[gt] = (*colours[kind],.8)
        axes[0,3].imshow(colour)
    axes[0,3].set_title('Available annotations (GT)\nMA cyan | HE red | EX yellow | SE green')
    for ax,kind in zip(axes[1],LESIONS):
        if kind not in displays:
            ax.set_facecolor('#eeeeee'); ax.text(.5,.5,'ABSENT MASK\nNot imputed; not scored',ha='center',va='center')
            ax.set_title(kind+' | unavailable'); continue
        gt = displays[kind]; colour = np.zeros((*gt.shape,4))
        colour[selected & ~gt] = (0,.8,1,.75); colour[gt & ~selected] = (1,0,.65,.85); colour[selected & gt] = (0,1,0,.85)
        ax.imshow(rgb); ax.imshow(colour); ax.set_title(kind+' | top10% comparison')
    known = row['dr_grade'] or 'unknown'
    fig.suptitle(f"{row['sample_key']} | A3 predicted {int(np.argmax(evidence['logits']))} | known grade: {known}\n"
                 'Class-agnostic explanation. Green: overlap | Cyan: selected outside this mask | Magenta: missed annotation',fontsize=12)
    out = io.BytesIO(); fig.savefig(out,format='jpg',dpi=125); plt.close(fig); audit.atomic_bytes(path,out.getvalue())


def receipt_path(output,row):
    return safe_path(output,'images/'+row['sample_key']+'/receipt_v1.json')


def read_receipt(output,row,signature):
    path = receipt_path(output,row)
    if not path.exists(): return None
    obj = read_json(path); digest = obj.pop('payload_sha256')
    require(canonical(obj)==digest and obj['signature']==signature and obj['input_row']==row,
            'Saved image receipt changed or belongs to a different run.')
    require(len(obj['scores'])==len(BUDGETS)*len(DOMAINS)*len(TARGETS), 'Incomplete per-image scoring receipt.')
    for rel,h in obj['files'].items():
        require(audit.sha_file(safe_path(output,rel))==h,'Saved evidence changed: '+rel)
    return obj


def process_one(output,ext,row,index,dataset,model,source_state,signature,progress):
    begun = time.perf_counter(); progress.stage = 'A3 rollout '+row['sample_key']
    sample = dataset[index]
    ev.sync(next(model.parameters()).device); start_forward = time.perf_counter()
    evidence,record = ar.explain_sample(model,sample)
    ev.sync(next(model.parameters()).device); forward_seconds = time.perf_counter()-start_forward
    tr.assert_tree_equal(model.state_dict(),source_state)
    require(all(p.grad is None for p in model.parameters()), 'Unexpected model gradients.')
    progress.stage = 'score masks '+row['sample_key']
    refs,displays,availability = read_references(ext,row)
    scores,controls,ranked,random = score_explanation(evidence,row,refs)
    folder = receipt_path(output,row).parent; folder.mkdir(parents=True,exist_ok=True)
    out = io.BytesIO(); np.savez_compressed(out,**evidence,ranked_region_ids=ranked,random_region_permutations=random)
    path = folder/'rollout_evidence_v1.npz'; audit.atomic_bytes(path,out.getvalue())
    files = {path.relative_to(output).as_posix():audit.sha_file(path)}
    for budget in BUDGETS:
        path = folder/f'selected_regions_{budget:02d}pct_v1.png'
        selected = (np.isin(evidence['labels'],ranked[:math.ceil(len(ranked)*budget/100)])
                    if not evidence['zero_region_flow'] else np.zeros_like(evidence['labels'],dtype=bool))
        audit.atomic_bytes(path,png_bytes(selected.astype(np.uint8)*255))
        files[path.relative_to(output).as_posix()] = audit.sha_file(path)
    if row['sample_key'] in PREVIEWS:
        path = output/f"idrid_xai_preview_{row['official_split']}_{row['canonical_id']}.jpg"
        save_preview(path,evidence,row,displays,ranked); files[path.name] = audit.sha_file(path)
    obj = dict(version=VERSION,signature=signature,input_row=row,record=record,
        known_dr_grade=int(row['dr_grade']) if row['dr_grade'] else None,mask_availability=availability,
        scores=scores,random_controls=controls,files=files,forward_seconds=forward_seconds,
        elapsed_seconds=time.perf_counter()-begun,model_forwards=2,original_lesion_masks_read=len(displays),
        processed_lesion_masks_read=len(displays),parameters_and_BN_buffers_unchanged=True,
        cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(next(model.parameters()).device)/2**20 if next(model.parameters()).device.type=='cuda' else 0.,
        cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(next(model.parameters()).device)/2**20 if next(model.parameters()).device.type=='cuda' else 0.)
    audit.save_json(receipt_path(output,row),{**obj,'payload_sha256':canonical(obj)})
    return obj


def write_table(output,name,rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    require(bool(fields),'Cannot publish empty table '+name)
    audit.save_csv(output/name,rows,fields)
    return audit.sha_file(output/name)


def aggregate(records):
    rows = [r for obj in records for r in obj['scores']]
    randoms = [r for obj in records for r in obj['random_controls']]
    by_random = defaultdict(list)
    for r in randoms: by_random[r['sample_key'],r['budget_percent_regions'],r['target']].append(r)
    pairs = []
    for r in rows:
        if r['domain']!='canvas_384' or r['status']!='scored': continue
        rnd = by_random[r['sample_key'],r['budget_percent_regions'],r['target']]
        require(len(rnd)==RANDOM_DRAWS and all(x['status']=='scored' for x in rnd), 'Incomplete random control.')
        pairs.append({k:r[k] for k in ('sample_key','official_split','budget_percent_regions','target')} |
            {'draws':len(rnd),'ranked_pixel_fraction':r['canvas_selected_fraction'],
             'random_mean_pixel_fraction':float(np.mean([x['canvas_selected_fraction'] for x in rnd]))} |
            {prefix+name:value for name in METRICS for prefix,value in
             (('ranked_',r[name]),('random_mean_',float(np.mean([x[name] for x in rnd]))),
              ('ranked_minus_random_',r[name]-float(np.mean([x[name] for x in rnd]))))})
    summary = []
    for domain in DOMAINS:
        for kind in TARGETS:
            for budget in BUDGETS:
                for stratum in ('all','training','testing'):
                    group = [r for r in rows if r['domain']==domain and r['target']==kind and r['budget_percent_regions']==budget
                             and (stratum=='all' or r['official_split']==stratum)]
                    valid = [r for r in group if r['status']=='scored']; statuses = Counter(r['status'] for r in group)
                    n = len(valid); components_n = sum(r['gt_components'] for r in valid)
                    paired = [r for r in pairs if r['target']==kind and r['budget_percent_regions']==budget and
                              (stratum=='all' or r['official_split']==stratum)] if domain=='canvas_384' else []
                    summary.append(dict(domain=domain,target=kind,budget_percent_regions=budget,
                        primary_budget=budget==PRIMARY_BUDGET,official_source_partition=stratum,total_images=len(group),
                        scored_positive_images=n,excluded_status_counts=json.dumps({k:v for k,v in statuses.items() if k!='scored'},sort_keys=True),
                        **{stat+'_'+m:(float(fun([r[m] for r in valid])) if n else None)
                           for m in METRICS for stat,fun in (('macro_mean',np.mean),('median',np.median))},
                        total_gt_components=components_n,total_hit_components=sum(r['hit_components'] for r in valid),
                        pooled_component_hit_rate=sum(r['hit_components'] for r in valid)/components_n if components_n else None,
                        mean_canvas_selected_fraction=float(np.mean([r['canvas_selected_fraction'] for r in valid])) if n else None,
                        paired_images=len(paired),**{prefix+m:float(np.mean([r[source+m] for r in paired])) if paired else None
                            for m in METRICS for prefix,source in (('random_macro_mean_','random_mean_'),
                                                                ('paired_mean_difference_','ranked_minus_random_'))}))
    return rows,randoms,pairs,summary


def publish(output,records,config,progress,decoded,reused):
    rows,randoms,pairs,comparison = aggregate(records)
    artifacts = {}
    for name,data in (('idrid_xai_per_image_v1.csv',rows),('idrid_xai_random_controls_v1.csv',randoms),
                      ('idrid_xai_paired_random_v1.csv',pairs),('idrid_xai_comparison_v1.csv',comparison)):
        # If every explanation has zero flow, the paired table has a header but no invented results.
        if not data:
            audit.save_csv(output/name,[],['sample_key','status']); artifacts[name]=audit.sha_file(output/name)
        else: artifacts[name]=write_table(output,name,data)
    predictions = [dict(sample_key=r['input_row']['sample_key'],official_split=r['input_row']['official_split'],
                        known_dr_grade=r['known_dr_grade'],predicted_grade=r['record']['prediction'],
                        **{f'probability_{i}':p for i,p in enumerate(r['record']['probabilities'])},
                        regions=r['record']['regions'],zero_region_flow=r['record']['zero_region_flow'],
                        attention_return_max_logit_difference=r['record']['attention_return_max_logit_difference']) for r in records]
    artifacts['idrid_xai_predictions_v1.csv'] = write_table(output,'idrid_xai_predictions_v1.csv',predictions)
    inventory = {receipt_path(output,r['input_row']).relative_to(output).as_posix():audit.sha_file(receipt_path(output,r['input_row'])) for r in records}
    for r in records: inventory.update(r['files'])
    audit.save_json(output/'idrid_xai_files_v1.json',inventory)
    artifacts['idrid_xai_files_v1.json']=audit.sha_file(output/'idrid_xai_files_v1.json')
    audit.save_json(output/'method_sources_v1.json',SOURCES); artifacts['method_sources_v1.json']=audit.sha_file(output/'method_sources_v1.json')
    summary = dict(status='IDRID_XAI_LOCALISATION_COMPLETE',version=VERSION,signature=config['signature'],
        images=len(records),source_partitions=dict(Counter(r['input_row']['official_split'] for r in records)),
        known_dr_grades=sum(r['known_dr_grade'] is not None for r in records),unknown_dr_grades=sum(r['known_dr_grade'] is None for r in records),
        variant='A3',checkpoint_sha256=ie.CHECKPOINT_SHA['A3'],best_epoch=21,protocol=PROTOCOL,
        preprocessing_review={**ie.PREPROCESS_REVIEW,'localisation_disposition':'All original lesion annotations retained in native-domain denominator, including the 48416-pixel IDRiD_61 HE stripe.'},
        primary_budget_results=[r for r in comparison if r['official_source_partition']=='all' and r['primary_budget']],
        image_commits_reused_this_process=reused,processed_RGB_decoded_this_process=decoded,
        model_forwards_this_process=2*decoded,logical_model_forwards_in_committed_images=sum(r['model_forwards'] for r in records),
        original_lesion_masks_in_committed_images=sum(r['original_lesion_masks_read'] for r in records),
        processed_lesion_masks_in_committed_images=sum(r['processed_lesion_masks_read'] for r in records),
        absent_lesion_masks_not_imputed=sum(x['availability']!='present' for r in records for x in r['mask_availability']),
        zero_region_flow_images=sum(r['record']['zero_region_flow'] for r in records),
        per_image_scoring_rows=len(rows),random_control_rows=len(randoms),
        optimizer_steps=0,APTOS_RGB_decoded=0,IDRiD_grading_RGB_decoded=0,optic_disc_masks_scored=0,
        parameters_and_BN_buffers_unchanged=True,upstream_files_unchanged=True,
        elapsed_seconds_this_process=time.perf_counter()-STARTED,
        committed_image_seconds=sum(r['elapsed_seconds'] for r in records),
        cuda_peak_allocated_mib=max(r['cuda_peak_allocated_mib'] for r in records),
        cuda_peak_reserved_mib=max(r['cuda_peak_reserved_mib'] for r in records),
        timing_note='Wall time includes SLIC, two forwards/image, original-mask scoring and evidence writes; interrupted unfinished work not included in committed-image sum.',
        runtime=config['identity']['runtime'],output_sha256=artifacts,saved=str(output),
        conclusion='Execution complete; localisation results still require interpretation. No localisation quality pass threshold is asserted.')
    audit.save_json(output/'idrid_xai_summary_v1.json',summary)
    audit.save_json(output/'idrid_xai_state_v1.json',dict(status='COMPLETE',signature=config['signature'],
                    summary_sha256=audit.sha_file(output/'idrid_xai_summary_v1.json')))
    return summary


def execute(root,output,device,progress):
    audit.load_libraries()
    require(pp.code_hash(Path(ar.__file__))==ROLLOUT_SHA_LF,'Use the accepted attention_rollout_v1.py.')
    require(pp.function_hash(audit.binary_mask_array)==pp.MASK_DECODER_SHA,'Use the reviewed r3 IDRiD binary-mask decoder.')
    ext = load_inputs(root,progress)
    project,info,files,codes = ar.locked_context(root,'A3',progress)
    require(info['best_sha256']==ie.CHECKPOINT_SHA['A3'] and info['summary']['best_epoch']==21,'Frozen A3 differs.')
    sources = {**ext['sources'],**{Path(p):h for p,h in files.items()},**{Path(p):h for p,h in codes.items()}}
    sources.update({safe_path(root,p):h for p,h in project['source_sha256'].items()})
    for module in (pp,audit,ie): sources[Path(module.__file__).resolve()] = audit.sha_file(Path(module.__file__))
    sources[Path(__file__).resolve()] = audit.sha_file(Path(__file__))
    identity = dict(version=VERSION,protocol=PROTOCOL,sources={str(p):h for p,h in sources.items()},
                    runtime={**ev.runtime_versions(),'pillow':PIL.__version__},device_type=device.type,
                    data_sha256=project['source_sha256'],output=str(output))
    config = dict(identity=identity,signature=canonical(identity),created_at=datetime.now(timezone.utc).isoformat())
    pp.protect_output(output,[root/'IDRiD',root/'train_images',root/'processed_images_384_v1',
        *[root/'environment/data_audit'/n for n in ('training_v1','evaluation_v1','xai_test_v1','split_v1',
          'idrid_audit_v1','idrid_cohort_v1','idrid_final_cohort_v1','idrid_preprocessing_v1','idrid_evaluation_v1')]])
    output.mkdir(parents=True,exist_ok=True)
    with ev.RunLock(output):
        configpath = output/'idrid_xai_config_v1.json'
        if configpath.exists():
            saved = read_json(configpath)
            require(saved['signature']==canonical(saved['identity'])==config['signature'],'Run identity changed; do not mix results.')
            config = saved
        else:
            require(not any(p.name!='evaluation.lock' for p in output.iterdir()),'Unrecognised files in new output folder.')
            audit.save_json(configpath,config)
        records = []; progress.reset('verify saved image commits',len(ext['rows']))
        for row in ext['rows']:
            records.append(read_receipt(output,row,config['signature'])); progress.done += 1
        reused = sum(x is not None for x in records)
        statepath = output/'idrid_xai_state_v1.json'
        if statepath.exists() and read_json(statepath)['status']=='COMPLETE':
            state = read_json(statepath)
            require(reused==len(records) and state['signature']==config['signature'] and
                    audit.sha_file(output/'idrid_xai_summary_v1.json')==state['summary_sha256'],'Completed run changed.')
            summary = read_json(output/'idrid_xai_summary_v1.json')
            for name,h in summary['output_sha256'].items(): require(audit.sha_file(output/name)==h,'Completed output changed: '+name)
            print('Already complete: all image commits/output hashes verified; no new model forward.',flush=True)
            return summary
        audit.save_json(statepath,dict(status='RUNNING',signature=config['signature']))
        dataset = SegmentationDataset(ext)
        if reused < len(records):
            progress.reset('load frozen A3',len(records),reused)
            tr.seed_runtime(info['config']['settings']['seed'])
            model,source_state = ev.load_best_model(info,device)
            require(not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters()),'Model is not frozen.')
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            try:
                for index,row in enumerate(ext['rows']):
                    if records[index] is not None: continue
                    records[index] = process_one(output,ext,row,index,dataset,model,source_state,config['signature'],progress)
                    progress.done += 1
                    print(f"{row['sample_key']} | committed {progress.done}/{len(records)} | masks={records[index]['original_lesion_masks_read']}/4",flush=True)
                tr.assert_tree_equal(model.state_dict(),source_state)
            finally:
                del model,source_state
                if device.type=='cuda': torch.cuda.empty_cache()
        ie.verify_unchanged(sources,progress)
        progress.reset('save localisation tables and summary')
        return publish(output,records,config,progress,dataset.decoded,reused)


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--resume',action='store_true',help='Continue verified image commits; default is also restart-safe.')
    args = parser.parse_args(); root = args.project_root.resolve()
    output = (args.output_dir or root/'environment/data_audit/idrid_xai_v1/localisation_A3_v1').resolve()
    try:
        with Progress() as progress:
            summary = execute(root,output,torch.device('cuda' if torch.cuda.is_available() else 'cpu'),progress)
    except (Exception,KeyboardInterrupt) as exc:
        print(f'\nIDRID XAI STOPPED | {type(exc).__name__}: {exc}')
        print('Verified image commits are retained. Resolve the issue, then: python idrid_xai_v1.py --resume')
        return 1
    print('\nIDRID XAI LOCALISATION COMPLETE | A3 | 81 segmentation images')
    print('Rollout: class-agnostic | Primary budget: top10% regions | Additional budgets: 1/5/20%')
    for r in summary['primary_budget_results']:
        if r['domain']!='canvas_384': continue
        fmt = lambda v: '--' if v is None else f'{v:.5f}'
        print(f"{r['target']} | n={r['scored_positive_images']}/81 | Dice={fmt(r['macro_mean_dice'])} | IoU={fmt(r['macro_mean_iou'])} | component hit={fmt(r['macro_mean_component_hit_rate'])}")
    print('Original-mask auxiliary scores retained; IDRiD_61 cropped-out HE positives included.')
    print('Optimizer steps: 0 | APTOS/grading RGB read: 0 | Weights/upstream files unchanged: PASS')
    print(f'Elapsed this process: {ev.clock(time.perf_counter()-STARTED)}\nSaved: {output}')
    print('Return idrid_xai_summary_v1.json, idrid_xai_comparison_v1.csv, idrid_xai_per_image_v1.csv and idrid_xai_preview_*.jpg.')
    print('After result review: remember to commit idrid_xai_v1.py; keep images/checkpoints/results out of Git.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
