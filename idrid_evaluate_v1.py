"""CP2: frozen A0-A3 external DR classification on the accepted IDRiD cohorts.

In C:\\CP2: python idrid_evaluate_v1.py
Continue:  python idrid_evaluate_v1.py --resume
Optional input/checkpoint check without model forward: --check-only

Predict the 507-image union once per model; score primary502 and sensitivity507
from the same saved logits. Reads processed grading RGB only; no segmentation
images, masks, training, threshold tuning, or checkpoint selection. All existing
model modules and preprocessing files remain unchanged.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import time

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading IDRiD external-evaluation libraries...', flush=True)
import numpy as np
import PIL
from PIL import Image
import torch
import evaluate_v1 as ev
import idrid_audit_v1 as audit

tr, d, require = ev.tr, ev.d, ev.require
VERSION = 'idrid_evaluation_v1_20261107'
VARIANTS = ('A0', 'A1', 'A2', 'A3')
BATCH_SIZE = 2
ROOT = Path(__file__).resolve().parent
PREPROCESS_SHA = '040331c6256eb83db6d75df38bf2a6e1b9647e6be5a72af3eba44dfc3a58c117'
FINAL_COHORT_SHA = '14dcbcae99974ed8ce4fa8263328a5ad751215d065bb18d24a1eeea142d53763'
INTERNAL_SUMMARY_SHA = '59ed3ae8fbbc10d27aedd90b39335b71d47f2e933bb01ba8e726672df1588cac'
INTERNAL_SIGNATURE = '4798db9cc4dda8c843eb15104b2fdc88a70a91715af446979b83712500f5aad0'
CHECKPOINT_SHA = {
    'A0':'8537f7b73906595829f55ef837fb268897108cee59b61b5a473f7a725d6bdc77',
    'A1':'a62a48e2f0d3731560d550489c035998c6736e79c0dc6c34eb98a366803e8c4c',
    'A2':'db90c0e5253ec188e07fe43b9f9195c6a06b0fec7d05f56d7df053b32a84500f',
    'A3':'921e0368a183287939d0142b4d2a1c2b0c3da5db5a3b97791ee7adff4a947219',
}
PRIMARY_COUNTS = [164,23,163,91,61]
SENSITIVITY_COUNTS = [167,23,165,91,61]
PRECAUTION_KEYS = {f'grading/training/IDRiD_{n}' for n in (216,228,244,260,318)}
# This records the completed review; it does not rewrite the historical summary.
PREPROCESS_REVIEW = {
    'status':'ACCEPTED_WITH_DOCUMENTED_LIMITATIONS', 'review_date':'2026-11-06',
    'decision_file':'idrid_preprocessing_review_decision_v1.json',
    'decision_sha256':'3f4daa4f44ea20f2d3acc3191442448b70ae48fa12be7e237c755b641e231068',
    'resolved_finding':'segmentation/testing/IDRiD_61 HE: all 48416 crop-excluded annotation pixels form a 17x2848 far-right stripe over RGB 0..4; source-mask edge anomaly documented.',
    'scope':'Retain accepted preprocessing and cohorts. This stage does not read or score lesion masks; original annotations/coverage and 384px limitations remain relevant to later localisation.',
    'source_audit_status':'FAIL', 'source_overlap_status':'REVIEW_REQUIRED',
    'historical_source_statuses_modified':False,
}
PROTOCOL = {
    'variants':list(VARIANTS), 'class_order':list(range(5)), 'batch_size':BATCH_SIZE,
    'cohorts':{'primary':502,'sensitivity':507}, 'inference_union':507,
    'prediction':'argmax of five logits; softmax probabilities; no TTA/calibration/threshold fitting',
    'model':'existing validation-selected checkpoints; eval mode, requires_grad False, torch.no_grad',
    'input':'existing aligned 384x384 RGB; same torchvision evaluation transforms and DenseNet normalization as APTOS',
    'segmentation':'existing train_v1.prepare_microbatch; CPU boundary/SLIC; weight0 A0/A2, weight2 A1/A3; no lesion masks',
    'metrics':'existing five-class QWK/macro-F1/accuracy/macro precision/macro recall; zero_division0; per-class measures and confusion matrices',
    'weighted_CE':'diagnostic only; original APTOS train-only class weights; never fit external class weights',
    'comparison':'curated IDRiD primary/sensitivity versus existing APTOS test results; descriptive unpaired dataset differences, not causal or significance claims',
    'official_partitions':'retained as source strata; both are external to CP2 training; not a claim to match the untouched official 516/103-image benchmark',
    'selection':'no checkpoint, hyperparameter or cohort changes based on external predictions',
    'resume':'same folder; atomic cache after each batch; unfinished batch may repeat; no interrupt checkpoints',
}
SOURCES = {
    'proposal':'Accepted CP1 proposal section3.10.3, printed49 / PDF55; metrics in3.10.1; computational scope3.10.4.',
    'implementation':'Project-specific adapter/runner. Directly reuses CP2 evaluate_v1, train_v1, classifier_v1 and dataset_v1 APIs; model architecture is not reimplemented.',
    'checkpoint_and_eval':'https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html',
    'gradient_disabled':'https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html',
    'qwk':'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html',
    'macro_and_class_measures':'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html',
    'definitions_accessed':'2026-11-07',
    'metric_code':'CP2 evaluate_v1.score_rows and metrics_from_confusion, implemented previously; sklearn is not a runtime dependency.',
    'IDRiD_dataset_DOI':'10.3390/data3030025',
    'cohort_rules':'Previously frozen project audit/overlap decisions; not claimed as official IDRiD benchmark membership.',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def safe_path(base, relative):
    r = PurePosixPath(relative)
    require(relative and not r.is_absolute() and '..' not in r.parts and '\\' not in relative
            and ':' not in relative, 'Invalid relative file path: '+relative)
    p = base.joinpath(*r.parts)
    require(not p.is_symlink() and p.resolve().is_relative_to(base.resolve()), 'File escapes expected folder: '+str(p))
    return p


def checked_file(path, expected, sources):
    require(path.is_file() and audit.sha_file(path) == expected, 'Input file missing/changed: '+str(path))
    sources[path] = expected
    return path


class Progress(ev.Progress):
    def show(self):
        processed = self.done-self.initial
        eta = (ev.clock((time.perf_counter()-self.phase_started)/processed*(self.total-self.done))
               if processed > 0 else ('00:00:00' if self.total and self.done == self.total else '--'))
        print(f'IDRiD evaluation | {self.stage} | {self.done}/{self.total} | '
              f'Elapsed {ev.clock(time.perf_counter()-STARTED)} | Phase ETA {eta}', flush=True)


def validate_cohorts(primary, union):
    require(len(primary) == 502 and len(union) == 507, 'IDRiD cohort counts changed.')
    keys = [r['sample_key'] for r in union]
    pk = {r['sample_key'] for r in primary}
    require(len(set(keys)) == 507 and len(pk) == 502 and set(keys)-pk == PRECAUTION_KEYS,
            'Primary/sensitivity membership differs from frozen policy.')
    by_key = {r['sample_key']:r for r in union}
    require(all(r == by_key[r['sample_key']] for r in primary), 'Primary row differs from the union row.')
    for r in union:
        key = r['sample_key']
        require(r['official_split'] in ('training','testing') and
                key == f"grading/{r['official_split']}/{r['canonical_id']}", 'Grading identity/partition mismatch.')
        require(r['dr_grade'] in ('0','1','2','3','4'), 'Missing/invalid DR label.')
        require(r['processed_rgb_path'] == 'rgb/'+key+'.png' and
                (int(r['output_width']),int(r['output_height'])) == (384,384), 'Unexpected image path/resolution.')
        require(r['sensitivity_classification'] == 'True' and r['segmentation_primary'] == 'False'
                and r['primary_classification'] == str(key in pk), 'Cohort flags changed.')
        require(not r['crop_warning'], 'Unreviewed classification crop warning.')
    for rows, counts, partitions in ((primary,PRIMARY_COUNTS,{'training':400,'testing':102}),
                                      (union,SENSITIVITY_COUNTS,{'training':405,'testing':102})):
        require([Counter(int(r['dr_grade']) for r in rows)[i] for i in range(5)] == counts, 'DR counts changed.')
        require(Counter(r['official_split'] for r in rows) == partitions, 'Official source partitions changed.')
    require(len({r['image_path'] for r in union}) == len({r['rgb_sha256'] for r in union}) == 507,
            'Repeated selected source path/RGB digest.')
    return pk


def load_external(root, progress):
    logs = root/'environment/data_audit/idrid_preprocessing_v1/aligned_384_v1'
    images = root/'IDRiD/processed_384_v1'
    final = root/'environment/data_audit/idrid_final_cohort_v1/final_aptos_idrid_v1'
    sources = {}
    summary_path = checked_file(logs/'idrid_preprocessing_summary_v1.json', PREPROCESS_SHA, sources)
    summary = read_json(summary_path)
    require(summary['status'] == 'PREPROCESSING_FILE_CHECK_PASS' and summary['classification_primary'] == 502
            and summary['classification_sensitivity'] == 507 and summary['same_APTOS_RGB_function'] is True
            and summary['automatic_review_counts'] == {'MASK_FOREGROUND_OUTSIDE_RGB_CROP':1},
            'Unexpected preprocessing result/review scope.')
    state_path = logs/'preprocessing_state_v1.json'
    state = read_json(state_path)
    require(state['status'] == 'COMPLETE' and state['summary_sha256'] == PREPROCESS_SHA
            and state['signature'] == summary['signature'], 'Preprocessing is incomplete or changed.')
    sources[state_path] = audit.sha_file(state_path)
    final_summary = read_json(checked_file(final/'idrid_final_cohort_summary_v1.json', FINAL_COHORT_SHA, sources))
    require(summary['source_final_cohort_summary_sha256'] == FINAL_COHORT_SHA, 'Wrong final cohort source.')
    names = ('idrid_classification_primary_384_v1.csv','idrid_classification_sensitivity_384_v1.csv',
             'idrid_preprocessing_config_v1.json','idrid_preprocessing_files_v1.json',
             'idrid_preprocessing_review_v1.csv')
    for name in names:
        checked_file(logs/name,summary['output_sha256'][name],sources)
    config = read_json(logs/names[2]); body = {k:v for k,v in config.items() if k != 'signature'}
    require(canonical(body) == config['signature'] == summary['signature'], 'Preprocessing configuration fingerprint differs.')
    require(config['image_root'] == str(images) and config['cohort_summary_sha256'] == FINAL_COHORT_SHA,
            'Preprocessing paths/cohort differ.')
    require(config['pillow'] == PIL.__version__, 'Pillow changed since accepted preprocessing; return the error before changing packages.')
    primary = d.read_csv(logs/names[0]); union = d.read_csv(logs/names[1])
    primary_keys = validate_cohorts(primary, union)
    inventory = read_json(logs/names[3])
    require(inventory['image_root'] == str(images), 'Output inventory image root differs.')
    # Match processed rows back to the frozen original cohort rows. No raw image is decoded.
    for name, processed in (('idrid_classification_primary_v1.csv',primary),
                            ('idrid_classification_sensitivity_v1.csv',union)):
        p = checked_file(final/name,final_summary['output_sha256'][name],sources)
        original = d.read_csv(p); lookup = {r['sample_key']:r for r in original}
        require(len(original) == len(lookup) == len(processed) and set(lookup) == {r['sample_key'] for r in processed},
                'Processed cohort differs from frozen original cohort.')
        for r in processed:
            require(all(r.get(k) == v for k,v in lookup[r['sample_key']].items()), 'Original label/provenance fields changed.')
    issues = d.read_csv(logs/'idrid_preprocessing_review_v1.csv')
    require(issues == [{'sample_key':'segmentation/testing/IDRiD_61','mask_type':'HE',
                       'issue':'MASK_FOREGROUND_OUTSIDE_RGB_CROP','details':'48416'}], 'An unreviewed preprocessing finding remains.')
    union = sorted(union,key=lambda r:r['sample_key'])
    progress.reset('verify 507 external grading RGB file hashes (no decoding)',507)
    for r in union:
        expected = r['processed_rgb_sha256']
        require(inventory['image_files'].get(r['processed_rgb_path']) == expected, 'RGB inventory/row hash mismatch.')
        checked_file(safe_path(images,r['processed_rgb_path']),expected,sources)
        progress.done += 1
    return {'root':images,'rows':union,'primary_keys':primary_keys,'sources':sources,'summary':summary}


def training_context(root, progress):
    runs = {v:root/'environment/data_audit/training_v1'/ev.RUNS[v] for v in VARIANTS}
    project, weights, infos, hashes = ev.preflight(root,runs,progress)
    sources = {Path(p):h for p,h in hashes.items()}
    sources.update({root/p:h for p,h in project['source_sha256'].items()})
    folder = root/'environment/data_audit/evaluation_v1/test_A0_A3_v1'
    baseline = read_json(checked_file(folder/'evaluation_summary_v1.json',INTERNAL_SUMMARY_SHA,sources))
    config_path = folder/'evaluation_config_v1.json'; state_path = folder/'evaluation_status_v1.json'
    config, state = read_json(config_path), read_json(state_path)
    require(tr.fingerprint(config['identity']) == config['signature'] == baseline['signature']
            == state['signature'] == INTERNAL_SIGNATURE and state['status'] == 'COMPLETE',
            'Original internal evaluation config/status differs.')
    for path in (config_path,state_path): sources[path] = audit.sha_file(path)
    require(baseline['status'] == 'TEST_EVALUATION_COMPLETE', 'Original internal evaluation is incomplete.')
    require(config['identity']['data_sha256'] == project['source_sha256'] and
            config['identity']['versions'] == ev.runtime_versions(), 'Model/data environment differs from internal evaluation.')
    for path,expected in config['identity']['code_sha256'].items():
        checked_file(Path(path),expected,sources)
    for v,info in infos.items():
        ref = baseline['results'][v]
        require(info['best_sha256'] == CHECKPOINT_SHA[v] == ref['checkpoint_sha256']
                == config['identity']['checkpoints'][v]['sha256'] and
                info['summary']['best_epoch'] == ref['best_epoch'] == ev.BEST_EPOCHS[v],
                v+': checkpoint differs from the accepted validation-selected model.')
    require(np.asarray(weights).shape == (5,) and np.isfinite(weights).all() and (weights>0).all(), 'Invalid train-only weights.')
    return weights,infos,sources,baseline


class ExternalDataset:
    """Only allowlisted grading PNGs. Original DR labels are never model inputs."""
    def __init__(self, external):
        self.root, self.rows = external['root'], external['rows']
        self.decoded = 0
        self.to_rgb = d.v2.Compose([d.v2.ToImage(),d.v2.ToDtype(torch.float32,scale=True),d.v2.ToPureTensor()])
        self.normalise = d.v2.Normalize(d.MEAN,d.STD,inplace=False)

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        require(row['sample_key'].startswith('grading/'), 'Only grading samples are permitted.')
        blob = safe_path(self.root,row['processed_rgb_path']).read_bytes()
        require(hashlib.sha256(blob).hexdigest() == row['processed_rgb_sha256'], 'Processed RGB changed before decoding.')
        with Image.open(io.BytesIO(blob)) as im:
            im.load()
            require(im.format == 'PNG' and im.mode == 'RGB' and im.size == (384,384), 'Expected RGB 384x384 PNG.')
            rgb = self.to_rgb(im).contiguous()
        self.decoded += 1
        return {'rgb':rgb,'cnn':self.normalise(rgb).contiguous(),'label':int(row['dr_grade']),
                'id_code':row['sample_key'],'split':'external'}


def row_identity(row, primary_keys):
    return {'sample_key':row['sample_key'],'official_split':row['official_split'],
            'label':int(row['dr_grade']),'in_primary':row['sample_key'] in primary_keys,
            'processed_rgb_sha256':row['processed_rgb_sha256']}


def write_cache(path, cache):
    audit.save_json(path,{**cache,'payload_sha256':canonical(cache)})


def load_cache(path, variant, signature, external):
    if not path.exists():
        return {'version':VERSION,'variant':variant,'signature':signature,'rows':[],
                'preparation_seconds':0.,'forward_seconds':0.,'peak_cuda_allocated_mib':0.,'peak_cuda_reserved_mib':0.}
    cache = read_json(path); digest = cache.pop('payload_sha256')
    require(canonical(cache) == digest and cache['version'] == VERSION and cache['variant'] == variant
            and cache['signature'] == signature,'Saved prediction cache changed or belongs to another run.')
    n = len(cache['rows'])
    require(n <= len(external['rows']) and (n == len(external['rows']) or n % BATCH_SIZE == 0), 'Incomplete cached batch.')
    for saved,expected in zip(cache['rows'],external['rows']):
        require({k:v for k,v in saved.items() if k!='logits'} == row_identity(expected,external['primary_keys']),
                'Cached ID/label/hash/order differs from external cohort.')
        logits = np.asarray(saved['logits'])
        require(logits.shape == (5,) and np.isfinite(logits).all(), 'Invalid cached logits.')
    for k in ('preparation_seconds','forward_seconds','peak_cuda_allocated_mib','peak_cuda_reserved_mib'):
        require(math.isfinite(cache[k]) and cache[k]>=0, 'Invalid cached timing/memory.')
    return cache


def predict_variant(info, external, device, cache, path, progress):
    dataset = ExternalDataset(external)
    initial = len(cache['rows']); total = len(dataset)
    progress.reset('EXTERNAL '+info['variant'],total,initial)
    if initial == total: return 0
    tr.seed_runtime(info['config']['settings']['seed'])
    model, source_state = ev.load_best_model(info,device)
    require(not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters()),
            'Model must be fully frozen and in eval mode.')
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.no_grad():
            for start in range(initial,total,BATCH_SIZE):
                indices = list(range(start,min(start+BATCH_SIZE,total)))
                ev.sync(device); begun = time.perf_counter()
                batch = tr.prepare_microbatch(dataset,indices,model.superpixel_method,device)
                ev.sync(device); prepared = time.perf_counter()
                refs = dataset.rows[start:start+len(indices)]
                require(batch['ids'] == [r['sample_key'] for r in refs]
                        and batch['labels'].tolist() == [int(r['dr_grade']) for r in refs], 'Batch order/labels changed.')
                result = model(batch['cnn'],batch['maps'],segmentation_method=model.superpixel_method)
                ev.sync(device); finished = time.perf_counter()
                logits = result['logits'].detach().cpu().numpy()
                require(logits.shape == (len(indices),5) and np.isfinite(logits).all(), 'Invalid external logits.')
                tr.assert_tree_equal(model.state_dict(),source_state)
                require(all(p.grad is None for p in model.parameters()), 'Unexpected inference gradients.')
                new_rows = [{**row_identity(r,external['primary_keys']),'logits':values.tolist()} for r,values in zip(refs,logits)]
                pending = {**cache,'rows':cache['rows']+new_rows,
                           'preparation_seconds':cache['preparation_seconds']+prepared-begun,
                           'forward_seconds':cache['forward_seconds']+finished-prepared}
                if device.type == 'cuda':
                    pending['peak_cuda_allocated_mib'] = max(cache['peak_cuda_allocated_mib'],torch.cuda.max_memory_allocated(device)/2**20)
                    pending['peak_cuda_reserved_mib'] = max(cache['peak_cuda_reserved_mib'],torch.cuda.max_memory_reserved(device)/2**20)
                write_cache(path,pending)  # Windows retry/fsync/replace; commit before reporting progress.
                cache.clear(); cache.update(pending)
                progress.done = len(cache['rows'])
                del batch,result,logits
        tr.assert_tree_equal(model.state_dict(),source_state)
    finally:
        del model,source_state
        if device.type == 'cuda': torch.cuda.empty_cache()
    return dataset.decoded


def scored_cohorts(cache, weights):
    result = {}
    for name in ('primary','sensitivity'):
        rows = [r for r in cache['rows'] if name=='sensitivity' or r['in_primary']]
        metrics,probabilities,predictions = ev.score_rows(rows,weights)
        counts = PRIMARY_COUNTS if name=='primary' else SENSITIVITY_COUNTS
        require(metrics['support'] == counts and metrics['samples'] == sum(counts), 'Incomplete external scoring cohort.')
        strata = {}
        for part in ('training','testing'):
            strata[part] = ev.score_rows([r for r in rows if r['official_split']==part],weights)[0]
        result[name] = {'metrics':metrics,'official_source_partition_metrics':strata}
    return result


def publish(output, caches, weights, infos, baseline, progress):
    results = {v:scored_cohorts(caches[v],weights) for v in VARIANTS}
    artifacts = {}
    def table(name, rows):
        audit.save_csv(output/name,rows,list(rows[0]))
        artifacts[name] = audit.sha_file(output/name)
    comparisons, generalisation, strata_rows = [],[],[]
    for v in VARIANTS:
        rows = caches[v]['rows']
        _,probs,preds = ev.score_rows(rows,weights)
        table(f'{v}_external_predictions_v1.csv',[
            {**{k:val for k,val in r.items() if k!='logits'},'predicted_grade':int(pred),
             **{f'probability_{c}':float(prob[c]) for c in range(5)},**{f'logit_{c}':r['logits'][c] for c in range(5)}}
            for r,prob,pred in zip(rows,probs,preds)])
        for name,detail in results[v].items():
            m = detail['metrics']
            comparisons.append({'variant':v,'cohort':name,'samples':m['samples'],'best_epoch':infos[v]['summary']['best_epoch'],
                                **{k:m[k] for k in (*ev.METRICS,'weighted_loss')}})
            internal = baseline['results'][v]['test_metrics']
            generalisation.append({'variant':v,'external_cohort':name,'APTOS_test_n':524,'IDRiD_n':m['samples'],
                **{prefix+k:val for k in ev.METRICS for prefix,val in
                   (('APTOS_',internal[k]),('IDRiD_',m[k]),('IDRiD_minus_APTOS_',m[k]-internal[k]))}})
            table(f'{v}_{name}_confusion_v1.csv',[
                {'true_grade':i,**{f'predicted_{j}':m['confusion_matrix'][i][j] for j in range(5)}} for i in range(5)])
            table(f'{v}_{name}_per_class_v1.csv',[
                {'grade':i,'support':m['support'][i],'precision':m['per_class_precision'][i],
                 'recall':m['per_class_recall'][i],'f1':m['per_class_f1'][i]} for i in range(5)])
            for part,sm in detail['official_source_partition_metrics'].items():
                strata_rows.append({'variant':v,'cohort':name,'official_source_partition':part,'samples':sm['samples'],
                                    **{k:sm[k] for k in ev.METRICS}})
    table('idrid_external_comparison_v1.csv',comparisons)
    table('idrid_internal_external_comparison_v1.csv',generalisation)
    table('idrid_official_partition_metrics_v1.csv',strata_rows)
    differences = []
    for name in ('primary','sensitivity'):
        for a,b,label in [('A1','A0','Boundary without GAT'),('A2','A0','GAT with SLIC'),
                          ('A3','A2','Boundary with GAT'),('A3','A1','GAT with boundary SLIC'),
                          ('A3','A0','Full versus baseline')]:
            differences.append({'cohort':name,'contrast':a+'-'+b,'interpretation':label,
                **{'delta_'+k:results[a][name]['metrics'][k]-results[b][name]['metrics'][k] for k in ev.METRICS}})
    table('idrid_ablation_differences_v1.csv',differences)
    import matplotlib.pyplot as plt
    for name in ('primary','sensitivity'):
        n = sum(PRIMARY_COUNTS if name=='primary' else SENSITIVITY_COUNTS)
        for normalized in (False,True):
            fig,axes = plt.subplots(2,2,figsize=(10,9),layout='constrained')
            for ax,v in zip(axes.flat,VARIANTS):
                cm = np.asarray(results[v][name]['metrics']['confusion_matrix'])
                values = np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros((5,5)),where=cm.sum(1,keepdims=True)>0) if normalized else cm
                limit = 1 if normalized else max(PRIMARY_COUNTS if name=='primary' else SENSITIVITY_COUNTS)
                im = ax.imshow(values,cmap='Blues',vmin=0,vmax=limit)
                for i in range(5):
                    for j in range(5):
                        label = f'{values[i,j]:.1%}' if normalized else str(cm[i,j])
                        ax.text(j,i,label,ha='center',va='center',color='white' if values[i,j]>limit*.5 else '#18202a')
                ax.set(xticks=range(5),yticks=range(5),xlabel='Predicted grade',ylabel='True grade',title=f'{v} | frozen epoch {infos[v]["summary"]["best_epoch"]}')
            fig.colorbar(im,ax=axes,shrink=.8,label='Fraction within true grade' if normalized else 'Images')
            fig.suptitle(f'IDRiD external {name} | {n} images per model')
            file = f'idrid_{name}_confusion_'+('normalized' if normalized else 'counts')+'_v1.png'
            stream = io.BytesIO(); fig.savefig(stream,format='png',dpi=160);plt.close(fig)
            audit.atomic_bytes(output/file,stream.getvalue());artifacts[file] = audit.sha_file(output/file)
    audit.save_json(output/'method_sources_v1.json',SOURCES)
    artifacts['method_sources_v1.json'] = audit.sha_file(output/'method_sources_v1.json')
    for v in VARIANTS: artifacts[f'{v}_prediction_cache_v1.json'] = audit.sha_file(output/f'{v}_prediction_cache_v1.json')
    return results,artifacts


def verify_unchanged(sources, progress):
    progress.reset('verify input files and checkpoints unchanged',len(sources))
    for path,expected in sources.items():
        require(audit.sha_file(path) == expected,'Input changed during external evaluation: '+str(path))
        progress.done += 1


def run(args):
    root = args.project_root.resolve()
    require(args.device!='cuda' or torch.cuda.is_available(),'CUDA is unavailable.')
    device = torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    output = root/'environment/data_audit/idrid_evaluation_v1/external_A0_A3_v1'
    require(not output.is_symlink() and output.resolve().is_relative_to(root),'Output path must remain in the project.')
    with Progress() as progress:
        weights,infos,sources,baseline = training_context(root,progress)
        external = load_external(root,progress); sources.update(external['sources'])
        # All old model modules are checked against original training/evaluation.
        # Record this new runner separately; normalize only its line endings.
        for module in (audit,): sources[Path(module.__file__).resolve()] = audit.sha_file(Path(module.__file__))
        if args.check_only:
            print('\nIDRID EXTERNAL INPUT CHECK: PASS | Checkpoints loaded | Model forwards: 0 | Images decoded: 0')
            print('Primary: 502 | Sensitivity/union: 507 | Ready: python idrid_evaluate_v1.py')
            return 0
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot  # Check dependency before the long run.
        runner = Path(__file__).resolve()
        runner_hash = hashlib.sha256(runner.read_text(encoding='utf-8-sig').encode()).hexdigest()
        identity = {'version':VERSION,'protocol':PROTOCOL,'review':PREPROCESS_REVIEW,
                    'runner_LF_sha256':runner_hash,'source_sha256':{str(p):h for p,h in sources.items()},
                    'ordered_samples':[row_identity(r,external['primary_keys']) for r in external['rows']],
                    'checkpoints':{v:{'sha256':infos[v]['best_sha256'],'epoch':infos[v]['summary']['best_epoch']} for v in VARIANTS},
                    'versions':{**ev.runtime_versions(),'pillow':PIL.__version__,'matplotlib':matplotlib.__version__},
                    'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                    'train_only_class_weights':weights.tolist()}
        signature = canonical(identity)
        output.mkdir(parents=True,exist_ok=True)
        with ev.RunLock(output):
            config_path = output/'idrid_evaluation_config_v1.json'
            status_path = output/'idrid_evaluation_status_v1.json'
            if config_path.exists():
                require(args.resume,'Run exists. Continue with: python idrid_evaluate_v1.py --resume')
                prior = read_json(config_path)
                require(prior['signature'] == canonical(prior['identity']) == signature and prior['identity'] == identity,
                        'Resume data, model, code, protocol or environment changed.')
            else:
                require(not list(output.glob('*_prediction_cache_v1.json')) and not status_path.exists(),
                        'Existing results have no matching configuration; do not overwrite.')
                audit.save_json(config_path,{'identity':identity,'signature':signature,'created_UTC':datetime.now(timezone.utc).isoformat()})
            caches = {v:load_cache(output/f'{v}_prediction_cache_v1.json',v,signature,external) for v in VARIANTS}
            if status_path.exists() and read_json(status_path).get('status') == 'COMPLETE':
                state = read_json(status_path)
                summary_path = output/'idrid_evaluation_summary_v1.json'
                require(state['signature'] == signature and audit.sha_file(summary_path) == state['summary_sha256'], 'Completed summary changed.')
                summary = read_json(summary_path)
                require(all(len(caches[v]['rows']) == 507 for v in VARIANTS), 'Completed run has incomplete predictions.')
                for rel,h in summary['output_sha256'].items():
                    require(audit.sha_file(safe_path(output,rel)) == h,'Completed result changed: '+rel)
                print('\nIDRID EXTERNAL EVALUATION ALREADY COMPLETE | No new image decoding or forward passes')
                print('Saved:',output)
                return 0
            active = False
            try:
                audit.save_json(status_path,{'status':'RUNNING','signature':signature}); active=True
                counts = {}; timings = {}
                for v in VARIANTS:
                    before = len(caches[v]['rows'])
                    decoded = predict_variant(infos[v],external,device,caches[v],output/f'{v}_prediction_cache_v1.json',progress)
                    counts[v] = {'resumed_predictions':before,'new_predictions':507-before,'images_decoded':decoded}
                    timings[v] = {k:caches[v][k] for k in ('preparation_seconds','forward_seconds','peak_cuda_allocated_mib','peak_cuda_reserved_mib')}
                    scores = scored_cohorts(caches[v],weights)
                    for cohort,details in scores.items():
                        m = details['metrics']
                        print(f'{v} | {cohort} n={m["samples"]} | QWK={m["qwk"]:.5f} | macro-F1={m["macro_f1"]:.5f} | accuracy={m["accuracy"]:.5f}',flush=True)
                verify_unchanged(sources,progress)
                require(hashlib.sha256(runner.read_text(encoding='utf-8-sig').encode()).hexdigest()==runner_hash,'Runner changed during execution.')
                progress.reset('save external comparisons and confusion matrices')
                results,artifacts = publish(output,caches,weights,infos,baseline,progress)
                artifacts[config_path.name] = audit.sha_file(config_path)
                summary = {'status':'IDRID_EXTERNAL_EVALUATION_COMPLETE','version':VERSION,'signature':signature,
                    'class_order':list(range(5)),'results':results,'unique_external_images':507,
                    'primary_images':502,'sensitivity_images':507,'predictions_total':2028,
                    'cohorts_share_predictions':True,'prediction_counts_this_process':counts,
                    'checkpoints':identity['checkpoints'],'train_only_class_weights':weights.tolist(),
                    'timing_and_memory':timings,'elapsed_seconds_this_process':time.perf_counter()-STARTED,
                    'timing_scope':'Accumulated committed-batch preparation and forward times; interrupted unsaved work, verification, plotting and checkpoint loading excluded. Sequential run throughput, not a controlled speed benchmark.',
                    'APTOS_images_decoded':0,'IDRiD_segmentation_images_decoded':0,'mask_files_read':0,
                    'optimizer_steps':0,'parameters_and_BN_buffers_unchanged':True,'upstream_files_unchanged':True,
                    'cohorts_unchanged':True,'preprocessing_review':PREPROCESS_REVIEW,
                    'source_preprocessing_summary_sha256':PREPROCESS_SHA,'source_final_cohort_summary_sha256':FINAL_COHORT_SHA,
                    'internal_evaluation_summary_sha256':INTERNAL_SUMMARY_SHA,'sources':SOURCES,
                    'limitations':['Single fixed training seed; descriptive comparisons without significance claims.',
                        'External cohorts were curated; not the unmodified official IDRiD 516/103 benchmark.',
                        'Six uncertain overlap pairs resulted in five precautionary grading exclusions from primary only; absence of confirmed overlap does not prove all possible overlaps absent.',
                        'Internal/external metric differences also reflect label prevalence, case mix and acquisition differences.',
                        'No lesion localisation or superpixel-boundary-quality metric is evaluated by this script.',
                        'CUDA scatter operations may prevent bitwise reproducibility.'],
                    'output_sha256':artifacts,'saved':str(output)}
                summary_path = output/'idrid_evaluation_summary_v1.json'
                audit.save_json(summary_path,summary)
                audit.save_json(status_path,{'status':'COMPLETE','signature':signature,'summary_sha256':audit.sha_file(summary_path)})
                active=False
            except (Exception,KeyboardInterrupt) as exc:
                if active:
                    try: audit.save_json(status_path,{'status':'STOPPED','signature':signature,'error':f'{type(exc).__name__}: {exc}'})
                    except (Exception,KeyboardInterrupt): pass
                raise
    print('\nIDRID EXTERNAL EVALUATION COMPLETE | A0-A3 | 507 unique images | 2028 predictions')
    print('Primary: 502 | Sensitivity: 507 | Shared predictions | Optimizer steps: 0')
    print('APTOS/segmentation images decoded: 0 | Masks read: 0 | Checkpoints/upstream files unchanged: PASS')
    print(f'Elapsed this process: {ev.clock(time.perf_counter()-STARTED)}\nSaved: {output}')
    print('Return idrid_evaluation_summary_v1.json, idrid_external_comparison_v1.csv and idrid_internal_external_comparison_v1.csv.')
    print('After review: remember to commit idrid_evaluate_v1.py; keep images/checkpoints/results out of Git.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    args = parser.parse_args()
    try:
        require(not (args.resume and args.check_only),'Use either --resume or --check-only.')
        return run(args)
    except (Exception,KeyboardInterrupt) as exc:
        print(f'\nIDRID EXTERNAL EVALUATION STOPPED | {type(exc).__name__}: {exc}',flush=True)
        print('Committed prediction batches remain saved. Resolve the error, then: python idrid_evaluate_v1.py --resume')
        return 130 if isinstance(exc,KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
