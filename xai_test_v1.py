r"""CP2: frozen A3 XAI evaluation on ALL 524 fixed APTOS test images.

    python xai_test_v1.py
    python xai_test_v1.py --resume

Place beside the original *_v1.py modules in C:\CP2. Requires the completed
A0-A3 test evaluation and its original A3 best checkpoint, not new training.
Imports the unchanged rollout, GAT-relation and faithfulness implementations.
No subset selection, protocol tuning, FOV mask, lesion-localisation claim or
automatic faithfulness PASS. Technical completion is separate from findings.

Each image has immutable original evidence and a prespecified intervention
plan. Each completed intervention is atomically committed to the SAME state
file. --resume skips committed work and preserves rankings/targets. A killed
process may repeat its current uncommitted operation. Existing output is never
silently restarted. Original model/data/results are read only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading formal test XAI libraries...', flush=True)
import numpy as np
import torch
import faithfulness_v1 as f

gr, ar, ev, tr, d = f.gr, f.ar, f.ev, f.tr, f.d
require = f.require
VERSION = 'xai_test_v1'
# Storage-only patch. The experiment protocol and original signatures stay fixed.
IO_REVISION = 'windows_atomic_save_retry_20261027'
SAVE_RETRY_SECONDS = 30.0
IO_STATS = {'retries': 0, 'recoveries': 0, 'timeouts': 0}
COMPATIBLE_ORIGINAL_RUNNER_SHA256 = frozenset({
    'ede0511d4bccaec4a200e26083a8e8fab9cdc1af222e8fc57bc691fadeeb3e9f',  # delivered LF
    'fe8e3e9d9177c07fc3dae6ec78027c4d8506d5cb62f534453e39a1ea39c31113',  # same file, CRLF
})
VARIANT = 'A3'
EXPECTED_COUNTS = [269, 51, 138, 26, 40]
FROZEN_CODE = {
    'attention_rollout_v1.py': 'abf98a5f174d8e7730f52873f630d80a548549e99c4e3701f9d9e5aa882ae98e',
    'gat_relations_v1.py': 'eb1a3383ad9b18d6022fcdfcab39a0ed4111bfb2e126fd97615d3c23f793bc07',
    'faithfulness_v1.py': 'a735db0def8d7ddd89d7e04b5d8e36bcde53a09674b0498dad961bffc0881ac0',
}
FAMILIES = (('edge_positive', 'none'), ('edge_negative', 'none'),
            ('region', 'mean'), ('region', 'black'))
SOURCES = {'rollout': ar.SOURCES, 'relations': gr.SOURCES, 'perturbation': f.SOURCES,
           'autograd': 'https://docs.pytorch.org/docs/2.12/notes/autograd.html',
           'cohort_summary': 'Project choice: equal image weights; separate budgets, signs, fills and controls. '
                             'Descriptive means/medians/IQR, not independent-trial inference or clinical validation.'}
LIMITS = [
    'Single model/seed and fixed internal test split; no new independent external cohort.',
    'All fixed test cases included, including original misclassifications. Target is original argmax, not true grade.',
    'Rollout is class-agnostic token relevance; GAT scores use the original predicted logit gradient.',
    'Finite interventions can disagree with local gradient signs; probability and logit responses may differ.',
    'Pixel replacement can introduce distribution shift; mean/black fills and random controls do not eliminate it.',
    'O/T/B is a border-connected near-black proxy, not a retinal or lesion mask. CNN features include spatial context.',
    'Uniform and matched controls answer different questions; matched controls condition on sign/proxy composition.',
    'Repeated random sets, overlap and failed area matches are reported. No valid matched draw means a missing paired effect.',
    'Statistics use one effect per image/setting. Budgets and 20 random draws are not independent images.',
    'Split groups are not patient identities. Descriptive summaries/IQR are not confidence intervals or significance tests.',
    'No test-driven changes to weights, preprocessing, rankings, budgets or classification thresholds.',
    'No lesion ground truth; behavioural faithfulness does not establish clinical correctness or lesion localisation.',
]


class OutputSaveError(OSError):
    """A bounded Windows sharing/access retry expired; old destination remains."""
    def __init__(self, path, action, winerror):
        self.output_path = Path(path)
        super().__init__(f'Windows kept blocking {action} for {SAVE_RETRY_SECONDS:g}s '
                         f'(WinError {winerror}): {path}. Previous committed results are retained. '
                         'Close any program holding this output file; if it persists, check its '
                         'read-only attribute and folder permissions, then use --resume.')


def retry_output(operation, path, action):
    """Retry only Windows access/sharing/lock errors, without deleting the target.

    A retry is for a transient file handle, not a permissions bypass. ENOSPC,
    invalid paths and other errors fail immediately. Ctrl+C remains responsive.
    """
    started = time.monotonic(); attempts = 0; next_notice = 0.; delay = .1
    while True:
        try:
            result = operation()
            if attempts:
                IO_STATS['recoveries'] += 1
                print(f'I/O recovered | {Path(path).name} | retries={attempts} | '
                      f'waited {time.monotonic()-started:.1f}s', flush=True)
            return result
        except OSError as exc:
            winerror = getattr(exc, 'winerror', None)
            if winerror not in (5, 32, 33):
                raise
            elapsed = time.monotonic()-started
            if elapsed >= SAVE_RETRY_SECONDS:
                IO_STATS['timeouts'] += 1
                raise OutputSaveError(path, action, winerror) from exc
            if elapsed >= next_notice:
                print(f'I/O retry | WinError {winerror} | {action}: {path} | '
                      f'{elapsed:.1f}/{SAVE_RETRY_SECONDS:g}s; waiting for file access', flush=True)
                next_notice = elapsed+5.
            attempts += 1; IO_STATS['retries'] += 1
            time.sleep(min(delay, SAVE_RETRY_SECONDS-elapsed))
            delay = min(delay*2, 2.)


def write_json(path, value):
    """Same JSON/schema and atomic replace as v1, with bounded Windows retries.

    On failure, leave the last committed destination intact. A leftover .tmp
    is never treated as committed evidence and is reused on the next save.
    """
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    encoded = json.dumps(value, indent=2, allow_nan=False)+'\n'
    retry_output(lambda: temporary.write_text(encoded, encoding='utf-8'), path, 'writing temporary output')
    retry_output(lambda: temporary.replace(path), path, 'replacing output')


def save_payload(path, payload):
    write_json(path, {'payload': payload, 'checksum': tr.fingerprint(payload)})


def resume_identity(config, current):
    """Allow the exact delivered runner's I/O-only upgrade, nothing else.

    Preserve the original experiment identity and signature so all committed
    image evidence remains valid. Actual execution hashes are audited separately.
    """
    old = config['identity']
    require(config['signature'] == tr.fingerprint(old), 'Saved config signature is invalid.')
    if old == current:
        return old, config['signature']
    key = str(Path(__file__).resolve())
    previous_codes, current_codes = old['code_sha256'], current['code_sha256']
    require(key in previous_codes and previous_codes.keys() == current_codes.keys() and
            previous_codes[key] in COMPATIBLE_ORIGINAL_RUNNER_SHA256,
            'Resume runner differs from the approved original or current version; prior results preserved.')
    candidate = {**current, 'code_sha256': {**current_codes, key: previous_codes[key]}}
    require(candidate == old,
            'Resume model/data/protocol/environment/device or upstream code changed; prior results preserved.')
    return old, config['signature']


def io_record(identity, signature, codes):
    key = str(Path(__file__).resolve())
    return {'revision': IO_REVISION, 'experiment_signature': signature,
            'original_runner_sha256': identity['code_sha256'][key],
            'executed_runner_sha256': codes[key],
            'approved_original_runner_compatibility_used': identity['code_sha256'][key] != codes[key],
            'change_scope': 'bounded Windows atomic output-save retries and failure reporting only; '
                            'model, data, explanation/intervention/aggregation protocol and saved signatures unchanged',
            'retryable_winerrors': [5, 32, 33], 'retry_timeout_seconds': SAVE_RETRY_SECONDS,
            'original_config_preserved': True}


def protocol():
    inherited = f.protocol()
    inherited['scope'] = 'all 524 fixed test images, original manifest order; no outcome-based exclusion'
    inherited['ranking'] = 'unchanged explainers on original test input; saved before any intervention; no reranking'
    return {'interventions': inherited,
            'rollout_method': {**ar.METHOD, 'selection': 'all fixed test IDs, original order'},
            'relation_method': gr.METHOD,
            'baseline_check': 'compare all five original logits to accepted A3 test cache, atol=rtol=5e-4; argmax must agree',
            'aggregation': 'one ranked-minus-mean-random effect per image/family/fill/budget/control/measure; '
                           'then equal-image mean, median, Q1/Q3, and descriptive proportion above zero',
            'strata': 'overall; true grade 0..4; original prediction correct/incorrect',
            'missing': 'retain every image; absent relation sign has no ranked comparison; failed area matches excluded '
                       'only from matched-control mean, with explicit counts; no valid draws means missing effect',
            'resume': 'immutable per-image evidence/plan; atomic intervention prefix; same folder/state files',
            'prediction': 'original classification metrics from accepted cache; perturbation outputs kept separately'}


def validate_cohort(project, refs):
    rows = project['rows']['test']; ids = [r['id_code'] for r in rows]
    require(len(rows) == sum(EXPECTED_COUNTS) and len(ids) == len(set(ids)), 'Wrong/incomplete test cohort.')
    require(Counter(int(r['diagnosis']) for r in rows) == dict(enumerate(EXPECTED_COUNTS)), 'Test class counts changed.')
    other = {r['id_code'] for s in ('train', 'validation') for r in project['rows'][s]}
    other_groups = {r['split_group_id'] for s in ('train', 'validation') for r in project['rows'][s]}
    require(not set(ids) & other and not {r['split_group_id'] for r in rows} & other_groups,
            'Test IDs/groups overlap training or validation.')
    require(len(refs) == len(rows), 'Original A3 evaluation cache is incomplete.')
    for row, ref in zip(rows, refs):
        require(row['split'] == 'test' and row['id_code'] == ref['id_code'] and
                int(row['diagnosis']) == ref['label'] and row['split_group_id'] == ref['split_group_id'],
                'Test/evaluation ID, order, split, label or group mismatch.')
        require(ref['output_sha256'] == project['processed'][row['id_code']]['output_sha256'], 'PNG identity changed.')
        values = np.asarray(ref['logits'])
        require(values.shape == (5,) and np.isfinite(values).all(), 'Invalid original prediction.')


def source_context(root, progress):
    for module in (ar, gr, f):
        path = Path(module.__file__).resolve()
        require(d.sha256(path) == FROZEN_CODE[path.name], f'Use the original unchanged {path.name}.')
    project, info, files, codes = ar.locked_context(root, VARIANT, progress)
    folder = root / 'environment/data_audit/evaluation_v1/test_A0_A3_v1'
    path = folder / f'{VARIANT}_cache_v1.json'
    require(path.is_file(), 'Original A3 test prediction cache is missing.')
    config = ev.read_json(folder / 'evaluation_config_v1.json')
    cache = ev.load_cache(path, VARIANT, ar.EVALUATION_SIGNATURE, project, config['identity']['batch_size'])
    validate_cohort(project, cache['rows']); files[str(path)] = d.sha256(path)
    for module in (gr, f):
        path = Path(module.__file__).resolve(); codes[str(path)] = d.sha256(path)
    codes[str(Path(__file__).resolve())] = d.sha256(Path(__file__))
    return project, info, files, codes, cache['rows']


def check_logits(actual, reference):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    require(a.shape == b.shape == (5,) and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid original logits.')
    np.testing.assert_allclose(a, b, atol=5e-4, rtol=5e-4)
    require(int(a.argmax()) == int(b.argmax()), 'Original predicted class differs from accepted test evaluation.')
    return float(np.max(np.abs(a-b)))


def make_snapshot(model, sample, row, reference):
    require(sample['id_code'] == row['id_code'] and sample['split'] == row['split'] == 'test' and
            int(sample['label']) == int(row['diagnosis']), 'Dataset returned the wrong test sample.')
    rolled, rollout_record = ar.explain_sample(model, sample)
    error = check_logits(rolled['logits'], reference['logits'])
    device = next(model.parameters()).device
    related, entry = gr.explain_relations(model, sample['cnn'].unsqueeze(0).to(device), rolled['labels'])
    error = max(error, check_logits(related['logits'], reference['logits']))
    entry.update(id_code=row['id_code'], label=int(row['diagnosis']), original_evaluation_max_logit_error=error)
    graph = tr.rag.build_region_graph(rolled['labels']); tr.rag.validate_graph(graph)
    spatial = f.spatial_profile(rolled['rgb_uint8'], rolled['labels'], graph)
    jobs = f.make_plan(row['id_code'], VARIANT, rolled, related, graph, spatial)
    return {'row': row, 'entry': entry, 'rollout_record': rollout_record, 'reference': reference,
            'rolled': rolled, 'related': related, 'graph': graph, 'spatial': spatial, 'jobs': jobs}


def save_snapshot(folder, signature, snapshot):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {prefix+'__'+key: np.asarray(value) for prefix in ('rolled', 'related', 'spatial')
              for key, value in snapshot[prefix].items()}
    retry_output(lambda: f.save_npz(folder / 'original_evidence_v1.npz', arrays),
                 folder / 'original_evidence_v1.npz', 'saving original arrays')
    plan = {'signature': signature, 'id_code': snapshot['row']['id_code'], 'jobs': snapshot['jobs']}
    save_payload(folder / 'intervention_plan_v1.json', plan)
    meta = {key: snapshot[key] for key in ('row', 'entry', 'rollout_record', 'reference')}
    meta.update(signature=signature, version=VERSION, plan_sha256=tr.fingerprint(snapshot['jobs']),
                files={name: d.sha256(folder / name) for name in ('original_evidence_v1.npz', 'intervention_plan_v1.json')})
    # Commit manifest last. An interrupted preparation without this manifest can be recomputed.
    save_payload(folder / 'original_manifest_v1.json', meta)
    return meta


def load_snapshot(folder, signature, row, reference):
    meta = f.load_payload(folder / 'original_manifest_v1.json')
    require(meta['signature'] == signature and meta['version'] == VERSION and meta['row'] == row and
            meta['reference'] == reference, 'Original evidence belongs to a different experiment/image.')
    require(set(meta['files']) == {'original_evidence_v1.npz', 'intervention_plan_v1.json'}, 'Invalid evidence file list.')
    for name, digest in meta['files'].items():
        require(d.sha256(folder / name) == digest, f'Saved original evidence changed: {row["id_code"]}/{name}')
    plan = f.load_payload(folder / 'intervention_plan_v1.json')
    require(plan['signature'] == signature and plan['id_code'] == row['id_code'] and
            tr.fingerprint(plan['jobs']) == meta['plan_sha256'], 'Saved intervention plan changed.')
    with np.load(folder / 'original_evidence_v1.npz', allow_pickle=False) as archive:
        parts = {p: {k.split('__', 1)[1]: archive[k].copy() for k in archive.files if k.startswith(p+'__')}
                 for p in ('rolled', 'related', 'spatial')}
    graph = tr.rag.build_region_graph(parts['rolled']['labels']); tr.rag.validate_graph(graph)
    related, rolled = parts['related'], parts['rolled']
    np.testing.assert_array_equal(rolled['labels'], related['labels'])
    np.testing.assert_array_equal(graph['undirected_edges'], related['pairs'])
    np.testing.assert_array_equal(graph['areas'], related['areas'])
    recalculated = gr.aggregate_relations(related['alpha'], related['alpha_gradient'], related['edge_index'], graph)
    np.testing.assert_allclose(recalculated['pair_signed'], related['pair_signed'], atol=1e-12, rtol=1e-12)
    check_logits(rolled['logits'], reference['logits']); check_logits(related['logits'], reference['logits'])
    require(int(related['target_class']) == meta['entry']['target_class'] == int(np.argmax(reference['logits'])),
            'Saved target differs from original prediction.')
    return {**parts, **{k: meta[k] for k in ('row', 'entry', 'rollout_record', 'reference')},
            'graph': graph, 'jobs': plan['jobs']}, meta


def verify_frozen(model, before):
    require(not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters()),
            'Model left frozen evaluation mode.')
    require(all(p.grad is None for p in model.parameters()), 'Parameter gradients unexpectedly accumulated.')
    tr.assert_tree_equal(before, tr.cpu_tree(model.state_dict()))


def execute_image(model, before, sample, snapshot, payload, path, progress, counters):
    rgb = np.rint(ar.as_numpy(sample['rgb']).transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)
    np.testing.assert_array_equal(rgb, snapshot['rolled']['rgb_uint8'])
    runner = f.FrozenRunner(model, sample, snapshot)
    check_logits(runner.baseline, snapshot['reference']['logits'])
    if payload['baseline_logits'] is None:
        payload.update(baseline_logits=runner.baseline.tolist(), baseline_checks=runner.checks)
    else:
        check_logits(runner.baseline, payload['baseline_logits'])
        runner.baseline = np.asarray(payload['baseline_logits'], dtype=np.float64)
    payload['status'] = 'RUNNING'; save_payload(path, payload)
    cache = {r['input_key']: {k: v for k, v in r.items() if k not in
             ('job_id', 'input_key', 'reused_identical_intervention')} for r in payload['results']}
    try:
        for job in snapshot['jobs'][len(payload['results']):]:
            progress.stage = f'{snapshot["row"]["id_code"]} | {job["family"]}/{job["baseline"]}/{job["control"]}'
            key = f.intervention_key(job); reused = key in cache
            if not reused:
                cache[key] = runner.execute(job)
            record = {'job_id': job['job_id'], 'input_key': key, **cache[key], 'reused_identical_intervention': reused}
            if len(payload['results'])+1 == len(snapshot['jobs']):
                # A complete committed prefix always includes this check, even
                # if the process is killed before setting status=COMPLETE.
                verify_frozen(model, before)
                payload['frozen_parameters_and_BN_verified'] = True
            payload['results'].append(record)
            try:
                save_payload(path, payload)
            except BaseException:
                payload['results'].pop(); raise
            counters['reused' if reused else 'computed'] += 1
            progress.job_done = len(payload['results']); progress.saved += 1
        verify_frozen(model, before)
        payload['status'] = 'COMPLETE'; save_payload(path, payload)
    finally:
        counters['perturbation_and_baseline_forwards'] += runner.calls


def describe(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    require(np.isfinite(values).all(), 'Nonfinite aggregate value.')
    if not len(values):
        return dict(n=0, mean=None, median=None, q25=None, q75=None, proportion_positive=None)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                proportion_positive=float((values > 0).mean()))


def image_tables(snapshot, payload):
    row, entry, spatial, graph = (snapshot[k] for k in ('row', 'entry', 'spatial', 'graph'))
    target = entry['target_class']; grade = int(row['diagnosis'])
    comparisons = f.comparisons(snapshot, payload)
    for comp in comparisons:
        comp.update(original_correct=(target == grade), split_group_id=row['split_group_id'])
    single = []
    for job, result in zip(snapshot['jobs'], payload['results']):
        if job['control'] == 'single':
            pair = job['selected'][0]
            single.append({'id_code': row['id_code'], 'true_grade': grade, 'target_class': target,
                           'family': job['family'], 'rank': job['rank'], 'pair_index': pair,
                           'region_ids': graph['undirected_edges'][pair].tolist(),
                           'proxy_group': f.GROUPS[int(spatial['pair_group'][pair])],
                           'signed_score': float(snapshot['related']['pair_signed'][pair]),
                           **{k: result[k] for k in ('logit_drop', 'probability_drop', 'prediction', 'prediction_changed')}})
    baseline = np.asarray(payload['baseline_logits'], dtype=np.float64)
    probability = f.output_metrics(baseline, baseline, target)['probabilities'][target]
    summary = {'id_code': row['id_code'], 'split_group_id': row['split_group_id'], 'true_grade': grade,
               'target_class': target, 'original_correct': target == grade, 'original_target_probability': probability,
               'original_logits': baseline.tolist(), 'evaluation_logits': snapshot['reference']['logits'],
               'original_evaluation_max_logit_error': max(entry['original_evaluation_max_logit_error'],
                                                         check_logits(baseline, snapshot['reference']['logits'])),
               'regions': graph['num_nodes'], 'physical_pairs': len(graph['undirected_edges']),
               'positive_pairs': entry['positive_pairs'], 'negative_pairs': entry['negative_pairs'],
               'dark_pixel_fraction': float(spatial['proxy'].mean()),
               'region_counts_O_T_B': np.bincount(spatial['region_group'], minlength=3).tolist(),
               'pair_counts_O_T_B': np.bincount(spatial['pair_group'], minlength=3).tolist(),
               'experiments': len(payload['results']),
               'unique_interventions': len({r['input_key'] for r in payload['results']}),
               'excluded_area_match_draws_across_fills': sum(not j['match_ok'] for j in snapshot['jobs']),
               'region_pixel_noop_experiments': sum(r['pixel_noop'] for j, r in zip(snapshot['jobs'], payload['results'])
                                                    if j['domain'] == 'region')}
    return summary, comparisons, single


def aggregate(comparisons, images):
    ids = {r['id_code'] for r in images}
    require(len(ids) == len(images), 'Duplicate image in aggregation.')
    keys = [(r['id_code'], r['family'], r['baseline'], r['fraction_requested']) for r in comparisons]
    require(len(keys) == len(set(keys)) and all(k[0] in ids for k in keys), 'Duplicated/unknown image-setting row.')
    strata = [('overall', 'all', ids)]
    strata += [('true_grade', str(g), {r['id_code'] for r in images if r['true_grade'] == g}) for g in range(5)]
    strata += [('original_correct', str(flag).lower(), {r['id_code'] for r in images if r['original_correct'] == flag})
               for flag in (True, False)]
    rows = []
    for group, value, subset in strata:
        for family, baseline in FAMILIES:
            for fraction in f.FRACTIONS:
                pool = [r for r in comparisons if r['id_code'] in subset and r['family'] == family and
                        r['baseline'] == baseline and r['fraction_requested'] == fraction]
                for control in ('random_all', 'random_matched'):
                    for measure in ('logit', 'probability'):
                        key = f'ranked_minus_{control}_oriented_{measure}_effect'
                        valid = [r for r in pool if r[key] is not None]
                        stats = describe([r[key] for r in valid])
                        mean = lambda values: describe(values)['mean']
                        rows.append({'stratum': group, 'stratum_value': value, 'family': family, 'fill': baseline,
                            'fraction': fraction, 'control': control, 'measure': measure,
                            'effect_definition': 'rise' if family == 'edge_negative' else 'drop',
                            'n_cohort_images': len(subset), 'n_ranked_images': len(pool), 'n_paired_images': stats['n'],
                            'n_missing_sign_ranking': len(subset)-len(pool), 'n_without_valid_random': len(pool)-len(valid),
                            'valid_random_draws': sum(r[control+'_n'] for r in pool),
                            'excluded_area_match_draws': sum(r[control+'_excluded_area_mismatches'] for r in pool),
                            'mean_ranked_effect_paired_images': mean([r[f'ranked_oriented_{measure}_effect'] for r in valid]),
                            'mean_random_effect_paired_images': mean([r[f'{control}_oriented_{measure}_effect_mean'] for r in valid]),
                            **{'paired_effect_'+k: v for k, v in stats.items() if k != 'n'},
                            'mean_unique_random_sets': mean([r[control+'_unique_interventions'] for r in valid]),
                            'mean_overlap_with_ranked_set': mean([r[control+'_ranked_set_overlap_mean'] for r in valid]),
                            'mean_ranked_T_share': mean([r['ranked_proxy_group_counts_O_T_B'][1]/r['count'] for r in pool]),
                            'mean_ranked_B_share': mean([r['ranked_proxy_group_counts_O_T_B'][2]/r['count'] for r in pool]),
                            'mean_ranked_selected_pixel_fraction': mean([r['ranked_selected_pixel_fraction'] for r in pool]),
                            'mean_ranked_changed_pixel_fraction': mean([r['ranked_changed_pixel_fraction'] for r in pool])})
    return rows


def save_figure(path, aggregated):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    titles = ['GAT positive: logit/probability drop', 'GAT negative: logit/probability rise',
              'Rollout regions: mean fill, drop', 'Rollout regions: black fill, drop']
    for col, (family, baseline) in enumerate(FAMILIES):
        for line, measure in enumerate(('logit', 'probability')):
            ax = axes[line, col]
            for control, colour, label in (('random_matched', '#2369a6', 'Matched random'),
                                           ('random_all', '#c06822', 'Uniform random')):
                data = sorted([r for r in aggregated if r['stratum'] == 'overall' and r['family'] == family and
                               r['fill'] == baseline and r['measure'] == measure and r['control'] == control],
                              key=lambda r: r['fraction'])
                x = np.asarray([r['fraction']*100 for r in data])
                val = lambda key: np.asarray([np.nan if r[key] is None else r[key] for r in data])
                ax.plot(x, val('paired_effect_median'), 'o-', color=colour, label=label, linewidth=1.5, markersize=4)
                if control == 'random_matched':
                    ax.fill_between(x, val('paired_effect_q25'), val('paired_effect_q75'), color=colour, alpha=.16,
                                    label='Matched: image IQR')
            ax.axhline(0, color='#666666', linewidth=.8, linestyle='--')
            ax.set_xticks([1, 5, 10, 20]); ax.grid(alpha=.2); ax.tick_params(labelsize=8)
            if line == 0:
                ax.set_title(titles[col], fontsize=10)
            else:
                ax.set_xlabel('Requested node/pair budget (%)', fontsize=9)
            ax.set_ylabel(f'Paired {measure} effect', fontsize=9)
    axes[0, 0].legend(fontsize=8, loc='best')
    fig.suptitle('A3 fixed test cohort: ranked effect minus mean random effect, per image', fontsize=13)
    fig.text(.5, .015, 'Lines: median across images. Band: interquartile range, not a confidence interval. '
             'Positive values favour ranked over that control.\nBudgets/signs/fills stay separate; see CSV for sample sizes, '
             'mean effects, exclusions and grade/correctness strata.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .95))
    temp = path.with_name(path.stem+'.tmp.png')
    try:
        retry_output(lambda: fig.savefig(temp, dpi=160, facecolor='white'), path, 'saving summary figure')
        retry_output(lambda: temp.replace(path), path, 'replacing summary figure')
    finally:
        plt.close(fig)
        # A failed replacement may leave .tmp for the next attempt. Do not let
        # a cleanup error hide the original save error or delete the destination.
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def technical_checks():
    checks = f.technical_checks()
    # Unequal random counts must not change the equal-image aggregate weight.
    images = [{'id_code': 'a', 'true_grade': 0, 'original_correct': True},
              {'id_code': 'b', 'true_grade': 1, 'original_correct': False}]
    rows = []
    for image_id, effect, n in (('a', 2., 20), ('b', -4., 1)):
        row = {'id_code': image_id, 'family': 'region', 'baseline': 'mean', 'fraction_requested': f.FRACTIONS[0],
               'count': 1, 'ranked_proxy_group_counts_O_T_B': [1, 0, 0],
               'ranked_selected_pixel_fraction': .1, 'ranked_changed_pixel_fraction': .1}
        for measure in ('logit', 'probability'):
            row[f'ranked_oriented_{measure}_effect'] = effect+1
            for control in ('random_all', 'random_matched'):
                row.update({control+'_n': n, control+'_excluded_area_mismatches': 20-n,
                            control+'_unique_interventions': n, control+'_ranked_set_overlap_mean': 0.,
                            f'{control}_oriented_{measure}_effect_mean': 1.,
                            f'ranked_minus_{control}_oriented_{measure}_effect': effect})
        rows.append(row)
    found = [r for r in aggregate(rows, images) if r['stratum'] == 'overall' and r['family'] == 'region' and
             r['fill'] == 'mean' and r['fraction'] == f.FRACTIONS[0] and r['control'] == 'random_matched'][0]
    require(found['n_paired_images'] == 2 and found['paired_effect_mean'] == -1. and
            found['paired_effect_median'] == -1. and found['paired_effect_proportion_positive'] == .5,
            'Image-level aggregation incorrectly weights random draws.')
    try:
        check_logits([1, 0, 0, 0, 0], [0, 1, 0, 0, 0])
    except (ValueError, AssertionError):
        pass
    else:
        raise ValueError('Baseline mismatch was not rejected.')
    return {**checks, 'equal_image_pairing_not_random_trial_weighting': True, 'baseline_mismatch_rejected': True}


class Progress(ar.Progress):
    def __init__(self):
        super().__init__()
        self.job_done = self.job_total = self.job_initial = self.saved = self.new_images = 0
        self.job_started = self.work_started = time.perf_counter()

    def show(self):
        progressed = self.job_done-self.job_initial
        eta = ev.clock((time.perf_counter()-self.job_started)/progressed*(self.job_total-self.job_done)) if progressed else '--'
        run_eta = ev.clock((time.perf_counter()-self.work_started)/self.new_images*(self.total-self.done)) if self.new_images else '--'
        print(f'Test XAI | {self.stage} | Images {self.done}/{self.total} | This image {self.job_done}/{self.job_total} | '
              f'Saved experiments {self.saved} | Elapsed {ev.clock(time.perf_counter()-STARTED)} | '
              f'Image ETA {eta} | Run ETA estimate {run_eta}', flush=True)


class OutputLock(ar.OutputLock):
    def __init__(self, output):
        self.path = output / 'xai_test.lock'


def run(args, progress):
    for key in IO_STATS:
        IO_STATS[key] = 0
    root = args.root.resolve()
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA is unavailable.')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads())); tr.seed_runtime(f.SEED)
    import matplotlib
    matplotlib.use('Agg')
    project, info, files, codes, refs = source_context(root, progress)
    checks = technical_checks(); rows = project['rows']['test']
    identity = {'version': VERSION, 'variant': VARIANT, 'split': 'test', 'ids': [r['id_code'] for r in rows],
                'checkpoint_sha256': info['best_sha256'], 'best_epoch': info['summary']['best_epoch'],
                'evaluation_signature': ar.EVALUATION_SIGNATURE, 'protocol': protocol(),
                'data_sha256': project['source_sha256'], 'input_file_sha256': files, 'code_sha256': codes,
                'versions': ev.runtime_versions(), 'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    signature = tr.fingerprint(identity)
    output = root / 'environment/data_audit/xai_test_v1/test_A3_v1'
    config_path = output / 'xai_test_config_v1.json'; state_path = output / 'xai_test_state_v1.json'
    # Recover an initial creation interrupted before config commit, but never
    # reuse a folder containing image evidence or an existing experiment.
    unfinished_creation = output.is_dir() and not config_path.exists() and all(
        p.name in {'xai_test.lock', 'xai_test_config_v1.json.tmp'} for p in output.iterdir())
    require(args.resume or not output.exists() or unfinished_creation,
            'Output exists. Continue with: python xai_test_v1.py --resume')
    require(not args.resume or config_path.is_file(), 'No formal test XAI run to resume.')
    output.mkdir(parents=True, exist_ok=True)
    with OutputLock(output):
        if args.resume:
            old = ev.read_json(config_path)
            identity, signature = resume_identity(old, identity)
        else:
            write_json(config_path, {'signature': signature, 'identity': identity, 'sources': SOURCES,
                                    'created_utc': datetime.now(timezone.utc).isoformat(), 'limits': LIMITS})
        io_path = output / 'xai_io_compatibility_v1.json'
        io_details = io_record(identity, signature, codes)
        if io_path.is_file():
            require(ev.read_json(io_path) == io_details, 'Saved I/O compatibility record differs.')
        else:
            write_json(io_path, io_details)
        if io_details['approved_original_runner_compatibility_used']:
            print('I/O fix active | original experiment identity and committed progress preserved', flush=True)
        progress.total = len(rows); progress.stage = 'checking saved test evidence'
        model = before = dataset = None
        counters = dict(decoded=0, prepared=0, resumed=0, computed=0, reused=0, perturbation_and_baseline_forwards=0)
        images, comparisons, singles, paths = [], [], [], [io_path]
        active = active_path = None
        state = {'signature': signature, 'status': 'RUNNING', 'completed_images': 0,
                 'saved_experiments': 0, 'current_id': None, 'error': None}
        write_json(state_path, state)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        try:
            for index, (row, reference) in enumerate(zip(rows, refs)):
                folder = output / 'images' / row['id_code']
                manifest_path = folder / 'original_manifest_v1.json'
                payload_path = folder / 'intervention_state_v1.json'
                active = active_path = None; sample = None
                progress.job_done = progress.job_initial = progress.job_total = 0
                progress.stage = f'{row["id_code"]} | checking saved original evidence'
                state['current_id'] = row['id_code']
                if manifest_path.is_file():
                    snapshot, meta = load_snapshot(folder, signature, row, reference)
                else:
                    require(not payload_path.exists(), 'Intervention state exists without its original evidence.')
                    if model is None:
                        model, before = ev.load_best_model(info, device)
                        dataset = d.AptosDataset(project, 'test', verify_file_hash=True)
                        require(dataset.augment is None and dataset.rows == rows, 'Unexpected test dataset/augmentation.')
                    progress.stage = f'{row["id_code"]} | original segmentation, rollout, GAT rankings and plan'
                    sample = dataset[index]; counters['decoded'] += 1
                    snapshot = make_snapshot(model, sample, row, reference)
                    verify_frozen(model, before)
                    meta = save_snapshot(folder, signature, snapshot); counters['prepared'] += 1
                payload = f.load_image_state(payload_path, signature, meta['plan_sha256'], snapshot)
                initial = len(payload['results']); counters['resumed'] += initial; progress.saved += initial
                if initial:
                    check_logits(payload['baseline_logits'], reference['logits'])
                if initial == len(snapshot['jobs']):
                    require(payload.get('frozen_parameters_and_BN_verified') is True,
                            'Complete image lacks the final frozen-model check.')
                progress.job_done = progress.job_initial = initial; progress.job_total = len(snapshot['jobs'])
                progress.job_started = time.perf_counter()
                if initial < len(snapshot['jobs']):
                    active, active_path = payload, payload_path
                    if model is None:
                        model, before = ev.load_best_model(info, device)
                        dataset = d.AptosDataset(project, 'test', verify_file_hash=True)
                        require(dataset.augment is None and dataset.rows == rows, 'Unexpected test dataset/augmentation.')
                    if sample is None:
                        sample = dataset[index]; counters['decoded'] += 1
                    execute_image(model, before, sample, snapshot, payload, payload_path, progress, counters)
                    progress.new_images += 1
                elif payload['status'] != 'COMPLETE':
                    # A kill after the last intervention can precede the completion flag.
                    payload['status'] = 'COMPLETE'; save_payload(payload_path, payload)
                image_row, comp, single = image_tables(snapshot, payload)
                images.append(image_row); comparisons.extend(comp); singles.extend(single)
                paths.extend([manifest_path, payload_path, folder / 'original_evidence_v1.npz', folder / 'intervention_plan_v1.json'])
                progress.done = len(images); active = active_path = None
                state.update(completed_images=len(images), saved_experiments=progress.saved, current_id=None)
                write_json(state_path, state)
                print(f'{row["id_code"]} | test {len(images)}/{len(rows)} | {len(payload["results"])} experiments | '
                      f'true={row["diagnosis"]}, original prediction={snapshot["entry"]["target_class"]} | saved', flush=True)
                del snapshot, payload, sample
            if model is not None:
                verify_frozen(model, before)
            progress.stage = 'verifying original files and aggregating once per image'
            ev.verify_sources(project, files, codes)
            for row in rows:
                require(d.sha256(root / 'processed_images_384_v1' / (row['id_code']+'.png')) ==
                        project['processed'][row['id_code']]['output_sha256'], 'Original test PNG changed.')
            require([r['id_code'] for r in images] == identity['ids'], 'Incomplete test coverage.')
            aggregated = aggregate(comparisons, images)
            tables = {'xai_images_v1.csv': images, 'xai_image_comparisons_v1.csv': comparisons,
                      'xai_cohort_summary_v1.csv': aggregated}
            if singles:
                tables['xai_single_pairs_v1.csv'] = singles
            for name, table in tables.items():
                retry_output(lambda: f.save_csv(output / name, table), output / name, 'saving summary table')
                paths.append(output / name)
            save_figure(output / 'xai_test_curves_v1.png', aggregated); paths.append(output / 'xai_test_curves_v1.png')
            cm = np.zeros((5, 5), dtype=np.int64)
            for ref in refs:
                cm[ref['label'], int(np.argmax(ref['logits']))] += 1
            summary = {'status': 'TEST_XAI_EVALUATION_COMPLETE', 'technical_status': 'PASS',
                'faithfulness_status': 'EMPIRICAL_RESULTS_REVIEW_REQUIRED', 'version': VERSION, 'signature': signature,
                'variant': VARIANT, 'best_epoch': info['summary']['best_epoch'], 'checkpoint_sha256': info['best_sha256'],
                'test_images': len(images), 'true_class_counts': EXPECTED_COUNTS,
                'original_classification_metrics_from_accepted_cache': ev.metrics_from_confusion(cm),
                'original_evaluation_all_argmax_preserved': True,
                'max_original_logit_difference': max(r['original_evaluation_max_logit_error'] for r in images),
                'total_experiments': sum(r['experiments'] for r in images),
                'resumed_experiments': counters['resumed'], 'new_experiments_this_process': counters['computed']+counters['reused'],
                'actual_interventions_this_process': counters['computed'], 'identical_interventions_reused_this_process': counters['reused'],
                'test_PNGs_decoded_this_process': counters['decoded'], 'original_explanations_created_this_process': counters['prepared'],
                'perturbation_and_baseline_prediction_forwards_this_process': counters['perturbation_and_baseline_forwards'],
                'forward_count_scope': 'Excludes original rollout/relation extraction forwards; includes frozen runner checks.',
                'train_PNGs_decoded': 0, 'validation_PNGs_decoded': 0, 'optimizer_steps': 0,
                'parameters_and_BN_buffers_unchanged': True, 'upstream_files_unchanged': True,
                'original_partitions_targets_and_topologies_preserved': True, 'technical_checks': checks,
                'io_compatibility': io_details, 'execution_code_sha256': codes,
                'io_events_this_process_before_summary_commit': dict(IO_STATS),
                'protocol': protocol(), 'sources': SOURCES, 'limits': LIMITS, 'images': images,
                'overall_comparisons': [r for r in aggregated if r['stratum'] == 'overall'],
                'all_grade_and_correctness_strata': 'xai_cohort_summary_v1.csv',
                'files_sha256': {str(p.relative_to(output)).replace('\\', '/'): d.sha256(p) for p in paths},
                'elapsed_seconds_this_process': time.perf_counter()-STARTED,
                'timing_scope': 'Current process only, including import/preflight/cache checks/saves/export; not accumulated run time.',
                'device': str(device), 'CUDA_peak_allocated_MiB': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None}
            write_json(output / 'xai_test_summary_v1.json', summary)
            state.update(status='COMPLETE', error=None, current_id=None); write_json(state_path, state)
        except (Exception, KeyboardInterrupt) as exc:
            # Saving a STOPPED flag must never mask the actual error. If this
            # exact destination just exhausted retries, keep its committed prefix.
            if active is not None and not (isinstance(exc, OutputSaveError) and exc.output_path == active_path):
                active['status'] = 'STOPPED'
                try:
                    save_payload(active_path, active)
                except (OSError, KeyboardInterrupt) as note:
                    print(f'Could not save STOPPED flag; last committed image state retained: {note}', flush=True)
            state.update(status='STOPPED', completed_images=progress.done, saved_experiments=progress.saved,
                         error=f'{type(exc).__name__}: {exc}')
            if not (isinstance(exc, OutputSaveError) and exc.output_path == state_path):
                try:
                    write_json(state_path, state)
                except (OSError, KeyboardInterrupt) as note:
                    print(f'Could not save run status; per-image committed results retained: {note}', flush=True)
            raise
    print(f'\nTEST XAI EVALUATION COMPLETE | A3 | {len(images)} test images | {progress.saved} experiments')
    print('Original predictions verified | Frozen weights/BN/source files unchanged | Optimizer steps: 0')
    print('TECHNICAL CHECK: PASS | Faithfulness findings: AWAITING EMPIRICAL REVIEW')
    print(f'Elapsed this process: {ev.clock(time.perf_counter()-STARTED)}\nSaved: {output}')
    print('Return xai_test_summary_v1.json, xai_cohort_summary_v1.csv and xai_test_curves_v1.png.')
    print('After review: remember to commit xai_test_v1.py.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        with Progress() as progress:
            run(args, progress)
        return 0
    except KeyboardInterrupt:
        print('\nSTOPPED. Continue: python xai_test_v1.py --resume', flush=True)
        return 130
    except Exception as exc:
        print(f'\nTEST XAI STOPPED | {type(exc).__name__}: {exc}', flush=True)
        config = args.root.resolve() / 'environment/data_audit/xai_test_v1/test_A3_v1/xai_test_config_v1.json'
        command = 'python xai_test_v1.py' + (' --resume' if config.is_file() else '')
        print(f'Committed work is retained. Resolve the reported issue, then: {command}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
