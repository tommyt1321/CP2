r"""CP2 3.9.3: frozen-model perturbation diagnostics on five fixed TRAIN cases.

    python faithfulness_v1.py
    python faithfulness_v1.py --resume

Requires the completed attention_rollout_v1 and gat_relations_v1 previews.
Default A3; A2 is supported after both of its previews exist. No retraining,
test/validation-image use, new segmentation, or lesion-localisation claim.

Edges: suppress both message directions/all heads AFTER softmax, no
renormalisation, keeping self-loops. Analyse positive and negative separately.
Regions: replace whole superpixels by original-image channel means or black;
recompute neural features with the ORIGINAL partition and graph topology.
Compare prespecified budgets with 20 uniform and matched random controls.
The near-black border proxy is descriptive, NOT a retinal/FOV/lesion mask.

Each completed intervention is saved atomically. Ctrl+C then --resume reuses
completed work; no series of interrupt checkpoints is created. Technical PASS
never means that the explanations have passed faithfulness or clinical review.
"""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading faithfulness diagnostic libraries...', flush=True)
import numpy as np
import torch
import gat_relations_v1 as gr

ar, ev, tr, d = gr.ar, gr.ev, gr.tr, gr.d
require = ar.require
VERSION = 'faithfulness_v1'
GAT_CODE_SHA256 = 'eb1a3383ad9b18d6022fcdfcab39a0ed4111bfb2e126fd97615d3c23f793bc07'
FRACTIONS = (.01, .05, .10, .20)
REPEATS = 20
SEED = 42
MATCH_ATTEMPTS = 512
GROUPS = ('O_no_detected_dark_contact', 'T_dark_border_contact', 'B_mostly_dark_border')
SOURCES = {
    'proposal': 'CP1 section3.9.3, PDF53/printed47: fixed partition/topology; ranked vs random regions and connections.',
    'deletion_principle': 'Petsiuk, Das & Saenko (2018), RISE, BMVC. https://arxiv.org/abs/1806.07421',
    'perturbation_limit': 'Hooker et al. (2019), A Benchmark for Interpretability Methods in Deep Neural Networks, '
                          'NeurIPS. https://arxiv.org/abs/1806.10758',
    'scope': 'Uses the perturbation-evaluation principle, not the RISE explanation algorithm. '
             'This is frozen-model testing, not ROAR remove-and-retrain. Replacement artefacts/distribution '
             'shift remain possible. Budgets, controls, proxy and baselines are explicit project choices.',
    'relation_scores': gr.SOURCES,
}


def protocol():
    return {
        'scope': 'five fixed training cases; descriptive diagnosis, not a population estimate or lesion validation',
        'fractions_of_all_nodes_or_physical_pairs': list(FRACTIONS), 'random_repeats_per_budget': REPEATS,
        'seed': SEED, 'count': 'ceil(fraction * total), positive/negative edge count capped at available sign',
        'target': 'original argmax class fixed throughout; retain signed logit and probability drops and all logits',
        'ranking': 'original saved rankings; no reranking after perturbation; stable ID tie-breaking',
        'edges': 'pair gate zero after attention softmax, both directions/all heads, self loops unchanged, no renormalisation',
        'edge_random_all': 'uniform physical pairs, same count; same random sets reused across signs when counts match',
        'edge_random_matched': 'same count, same sign and same near-black-proxy category counts as ranked set',
        'regions': 'whole regions; fixed saved labels/topology; rerun CNN/GAT/ViT on altered RGB',
        'region_fills': ['original_image_per_channel_mean_including_background', 'black_RGB_zero'],
        'region_random_all': 'uniform regions, same count; actual selected/changed pixel areas recorded',
        'region_random_matched': 'same count and proxy categories; rejection-sample area within max(64px,5% ranked area); '
                                 '512 attempts then closest candidate FLAGGED and excluded from matched comparison',
        'proxy': 'max uint8 RGB <=7, 4-connected to canvas border; B region fraction>=0.95, T fraction>0, O fraction=0; '
                 'B pair has two B endpoints, T pair otherwise has any detected contact, O pair has none',
        'proxy_limit': 'descriptive near-black connectivity, NOT anatomical FOV; may miss nonblack background or include dark tissue',
        'single_pairs': 'also suppress each original top5 positive and top5 negative pair individually',
        'edge_cache': 'reuse original live CNN regional features within one image; compare cached tail against complete model '
                      'for all-ones and one suppressed pair; never reuse those features for region interventions',
        'precision': 'eval, frozen parameters, no_grad, float32; descriptive summaries float64',
        'random_band': '5th-95th percentiles of random interventions, not a confidence interval or significance test',
        'curve': 'separate per-image budget comparisons; no automatic faithfulness PASS and no full deletion-AUC claim',
    }


def rng_for(*parts):
    seed = int(hashlib.sha256(json.dumps([SEED, *parts], separators=(',', ':')).encode()).hexdigest()[:16], 16)
    return np.random.Generator(np.random.PCG64(seed))


def border_dark_proxy(rgb):
    """Only a descriptive pixel property; no morphology/retina segmentation."""
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, 'Expected uint8 RGB.')
    dark = rgb.max(axis=2) <= 7
    h, w = dark.shape; connected = np.zeros_like(dark)
    queue = deque()
    for y, x in [(y, x) for y in (0, h-1) for x in range(w)] + [(y, x) for y in range(h) for x in (0, w-1)]:
        if dark[y, x] and not connected[y, x]:
            connected[y, x] = True; queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and dark[yy, xx] and not connected[yy, xx]:
                connected[yy, xx] = True; queue.append((yy, xx))
    return connected


def spatial_profile(rgb, labels, graph):
    proxy = border_dark_proxy(rgb)
    areas = graph['areas']
    fraction = np.bincount(labels.ravel(), weights=proxy.ravel(), minlength=len(areas)) / areas
    categories = np.where(fraction >= .95, 2, np.where(fraction > 0, 1, 0)).astype(np.int64)
    endpoints = categories[graph['undirected_edges']]
    pair_group = np.where((endpoints == 2).all(axis=1), 2,
                          np.where((endpoints > 0).any(axis=1), 1, 0)).astype(np.int64)
    return {'proxy': proxy, 'dark_fraction': fraction, 'region_group': categories, 'pair_group': pair_group}


def sample_stratified(rng, selected, eligible, categories):
    chosen = []
    for group in range(3):
        count = int(np.count_nonzero(categories[selected] == group))
        pool = eligible[categories[eligible] == group]
        require(len(pool) >= count, 'Matched control pool is too small.')
        if count:
            chosen.extend(rng.choice(pool, count, replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def region_matched_draw(rng, selected, categories, areas):
    target = int(areas[selected].sum()); tolerance = max(64., .05 * target)
    best = None; best_error = math.inf
    for attempt in range(1, MATCH_ATTEMPTS + 1):
        candidate = sample_stratified(rng, selected, np.arange(len(areas)), categories)
        error = abs(int(areas[candidate].sum()) - target)
        if error < best_error:
            best, best_error = candidate, error
        if error <= tolerance:
            return candidate, {'match_ok': True, 'area_error_pixels': error, 'area_tolerance_pixels': tolerance,
                               'matching_attempts': attempt}
    return best, {'match_ok': False, 'area_error_pixels': best_error, 'area_tolerance_pixels': tolerance,
                  'matching_attempts': MATCH_ATTEMPTS}


def make_plan(image_id, variant, rolled, related, graph, spatial):
    """All choices depend on original evidence/geometry, never intervention outputs."""
    jobs = []; signed = related['pair_signed']; p = len(signed); n = graph['num_nodes']
    def add(domain, family, control, fraction, repeat, selected, *, baseline='none', direction=1, rank=0, **metadata):
        selected = sorted(int(i) for i in selected)
        require(len(selected) == len(set(selected)) and selected, 'An intervention needs distinct IDs.')
        require(min(selected) >= 0 and max(selected) < (p if domain == 'edge' else n), 'Selected ID out of range.')
        key = f'{family}_{baseline}_{control}_f{round(fraction*100):02d}_r{repeat:02d}_s{rank:02d}'
        groups = spatial['pair_group' if domain == 'edge' else 'region_group']
        jobs.append({'job_id': key, 'domain': domain, 'family': family, 'control': control, 'fraction': fraction,
                     'repeat': repeat, 'baseline': baseline, 'direction': direction, 'rank': rank,
                     'selected': selected, 'count': len(selected),
                     'actual_count_fraction': len(selected)/(p if domain == 'edge' else n),
                     'group_counts': np.bincount(groups[selected], minlength=3).tolist(),
                     'match_ok': True, **metadata})
    for sign, family in ((1, 'edge_positive'), (-1, 'edge_negative')):
        ranking = gr.ranked_indices(signed, sign)
        if not len(ranking):
            continue
        for fraction in FRACTIONS:
            k = min(len(ranking), max(1, math.ceil(p * fraction)))
            top = ranking[:k]
            add('edge', family, 'ranked', fraction, 0, top, direction=sign)
            for repeat in range(REPEATS):
                random_ids = rng_for(variant, image_id, 'edge_all', fraction, k, repeat).choice(p, k, replace=False)
                matched = sample_stratified(rng_for(variant, image_id, family, fraction, repeat),
                                             top, ranking, spatial['pair_group'])
                add('edge', family, 'random_all', fraction, repeat, random_ids, direction=sign)
                add('edge', family, 'random_matched', fraction, repeat, matched, direction=sign)
        for rank, pair in enumerate(ranking[:5], 1):
            add('edge', family, 'single', 0., 0, [pair], direction=sign, rank=rank)
    region_order = np.lexsort((np.arange(n), -rolled['region_raw']))
    for fraction in FRACTIONS:
        k = min(n, max(1, math.ceil(n * fraction))); top = region_order[:k]
        draws = []
        for repeat in range(REPEATS):
            random_ids = rng_for(variant, image_id, 'region_all', fraction, repeat).choice(n, k, replace=False)
            matched, metadata = region_matched_draw(rng_for(variant, image_id, 'region_match', fraction, repeat),
                                                    top, spatial['region_group'], graph['areas'])
            draws.append((random_ids, matched, metadata))
        for baseline in ('mean', 'black'):
            add('region', 'region', 'ranked', fraction, 0, top, baseline=baseline)
            for repeat, (random_ids, matched, metadata) in enumerate(draws):
                add('region', 'region', 'random_all', fraction, repeat, random_ids, baseline=baseline)
                add('region', 'region', 'random_matched', fraction, repeat, matched, baseline=baseline, **metadata)
    require(len({j['job_id'] for j in jobs}) == len(jobs), 'Duplicate experiment IDs.')
    return jobs


def replace_regions(rgb, labels, selected, baseline):
    require(rgb.dtype == np.float32 and rgb.shape[:2] == labels.shape and rgb.shape[2] == 3,
            'Expected float32 HWC RGB aligned with fixed labels.')
    require(baseline in ('mean', 'black'), 'Unknown replacement baseline.')
    mask = np.isin(labels, selected)
    fill = rgb.mean(axis=(0, 1), dtype=np.float64).astype(np.float32) if baseline == 'mean' else np.zeros(3, np.float32)
    result = rgb.copy(); result[mask] = fill
    changed = np.any(result != rgb, axis=2)
    return result, {'selected_pixels': int(mask.sum()), 'selected_pixel_fraction': float(mask.mean()),
                    'changed_pixels': int(changed.sum()), 'changed_pixel_fraction': float(changed.mean()),
                    'mean_absolute_RGB_change': float(np.abs(result-rgb).mean(dtype=np.float64)),
                    'fill_RGB_0_1': fill.tolist(), 'pixel_noop': bool(not changed.any())}


def normalise_rgb(rgb, device):
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
    mean = tensor.new_tensor(d.MEAN).view(1, 3, 1, 1); std = tensor.new_tensor(d.STD).view(1, 3, 1, 1)
    return ((tensor - mean) / std).contiguous()


def output_metrics(logits, baseline_logits, target):
    values = np.asarray(logits, dtype=np.float64); original = np.asarray(baseline_logits, dtype=np.float64)
    require(values.shape == original.shape == (5,) and np.isfinite(values).all() and np.isfinite(original).all(),
            'Invalid classification output.')
    def softmax(x):
        x = np.exp(x-x.max()); return x/x.sum()
    probabilities = softmax(values); before = softmax(original)
    return {'logits': values.tolist(), 'probabilities': probabilities.tolist(),
            'prediction': int(values.argmax()), 'target_class': int(target),
            'target_logit': float(values[target]), 'target_probability': float(probabilities[target]),
            'logit_drop': float(original[target]-values[target]),
            'probability_drop': float(before[target]-probabilities[target]),
            'prediction_changed': bool(int(values.argmax()) != int(original.argmax()))}


class FrozenRunner:
    """Validated cached GAT/ViT tail for edges; full neural rerun for RGB changes."""
    def __init__(self, model, sample, snapshot):
        self.model = model; self.snapshot = snapshot; self.calls = 0
        self.device = next(model.parameters()).device
        self.rgb = ar.as_numpy(sample['rgb']).transpose(1, 2, 0).copy()
        self.cnn = sample['cnn'].unsqueeze(0).to(self.device)
        self.labels = snapshot['rolled']['labels']; self.graph = snapshot['graph']
        self.target = snapshot['entry']['target_class']
        gr.require_frozen_gat(model, self.cnn)
        torch.testing.assert_close(normalise_rgb(self.rgb, self.device), self.cnn, atol=1e-6, rtol=0)
        with torch.no_grad():
            full = self.full_forward(self.rgb)
            self.regional = model.regional(self.cnn, [self.labels])
            self.packed = tr.rag.pack_graph_batch(self.regional, [self.graph])
            gates = torch.ones(len(self.graph['undirected_edges']), device=self.device)
            cached = self.edge_forward(gates)
            np.testing.assert_allclose(cached, full, atol=5e-4, rtol=5e-4)
            np.testing.assert_allclose(full, snapshot['related']['logits'], atol=5e-4, rtol=5e-4)
            require(int(np.argmax(full)) == self.target, 'Original prediction changed.')
            self.baseline = full
            probe = gates.clone(); probe[int(np.argmax(np.abs(snapshot['related']['pair_signed'])))] = 0
            reference = gr.forward_with_pair_gates(model, self.cnn, self.labels, probe)['logits'][0]
            self.calls += 1
            fast = self.edge_forward(probe)
            reference = ar.as_numpy(reference)
            np.testing.assert_allclose(fast, reference, atol=5e-4, rtol=5e-4)
            self.checks = {'cached_all_ones_max_logit_error': float(np.max(np.abs(cached-full))),
                           'cached_suppressed_pair_max_logit_error': float(np.max(np.abs(fast-reference))),
                           'baseline_saved_logit_max_error': float(np.max(np.abs(full-snapshot['related']['logits']))),
                           'baseline_prediction_preserved': True}

    def full_forward(self, rgb):
        output = self.model(normalise_rgb(rgb, self.device), [self.labels],
                            segmentation_method=self.model.superpixel_method, return_attention=True)
        self.calls += 1
        require(np.array_equal(ar.as_numpy(output['token_region_ids'][0]), np.arange(self.graph['num_nodes']+1)-1),
                'Region IDs changed during visual intervention.')
        np.testing.assert_allclose(ar.as_numpy(output['geometry'])[0], self.snapshot['related']['geometry'], atol=2e-6, rtol=0)
        return ar.as_numpy(output['logits'][0]).copy()

    def edge_forward(self, gates):
        with gr.AttentionProbe(self.model.graph_stage.conv, self.graph, gates) as probe:
            refined = self.model.graph_stage(self.packed, return_attention=True)
            probe.require_connected({'gat_attention': refined['attention']})
            current = {**self.regional, **tr.rag.unpack_graph_features(refined['x'], self.packed['ptr'])}
            encoded = self.model.encoder(self.model.preparer(current), return_attention=True)
            logits = self.model.classifier(encoded['cls'])[0]
        self.calls += 1
        return ar.as_numpy(logits).copy()

    def execute(self, job):
        with torch.no_grad():
            if job['domain'] == 'edge':
                gates = torch.ones(len(self.graph['undirected_edges']), device=self.device)
                gates[job['selected']] = 0
                logits = self.edge_forward(gates)
                details = {'selected_pixels': 0, 'selected_pixel_fraction': 0., 'changed_pixels': 0,
                           'changed_pixel_fraction': 0., 'mean_absolute_RGB_change': 0.,
                           'fill_RGB_0_1': [], 'pixel_noop': True}
            else:
                altered, details = replace_regions(self.rgb, self.labels, job['selected'], job['baseline'])
                logits = self.full_forward(altered)
        return {**output_metrics(logits, self.baseline, self.target), **details}


def code_names(values):
    names = {str(p).replace('\\', '/').rsplit('/', 1)[-1]: h for p, h in values.items()}
    require(len(names) == len(values), 'Ambiguous code basenames.')
    return names


def source_context(root, variant, progress):
    require(d.sha256(Path(gr.__file__)) == GAT_CODE_SHA256 and
            d.sha256(Path(ar.__file__)) == gr.ROLLOUT_CODE_SHA256, 'Use the delivered unchanged explanation modules.')
    project, info, files, codes = ar.locked_context(root, variant, progress)
    rollout_folder, chosen, old_entries, rollout_files, rollout_signature = gr.load_rollout_context(
        root, variant, project, info, codes)
    files.update(rollout_files)
    codes[str(Path(gr.__file__).resolve())] = d.sha256(Path(gr.__file__))
    folder = root / 'environment/data_audit/gat_relations_v1' / f'train_preview_{variant}_v1'
    paths = [folder / f'gat_relations_{part}_v1.json' for part in ('config', 'summary', 'state')]
    config, summary, state = (ev.read_json(path) for path in paths)
    identity = config['identity']; signature = tr.fingerprint(identity)
    require(config['signature'] == summary['signature'] == state['signature'] == signature and
            summary['status'] == 'GAT_RELATIONS_PREVIEW_COMPLETE' and state['status'] == 'COMPLETE',
            'A completed GAT relationship preview is required.')
    require(identity['variant'] == summary['variant'] == variant and identity['split'] == 'train' and
            identity['checkpoint_sha256'] == summary['checkpoint_sha256'] == info['best_sha256'] and
            identity['best_epoch'] == summary['best_epoch'] == info['summary']['best_epoch'] and
            identity['rollout_signature'] == summary['rollout_signature'] == rollout_signature,
            'Relationship preview belongs to a different model or rollout.')
    require(identity['data_sha256'] == project['source_sha256'] and identity['versions'] == ev.runtime_versions() and
            code_names(identity['code_sha256']) == code_names(codes) and identity['method'] == summary['method'] == gr.METHOD,
            'Relationship preview data/code/environment/method differs.')
    require(sorted(identity['input_file_sha256'].values()) == sorted(files.values()), 'Source evidence changed since relations.')
    require(state['entries'] == summary['results'] and identity['selected_ids'] == [r['id_code'] for r in chosen] and
            len(state['entries']) == len(chosen) == 5, 'Expected the same five fixed training cases.')
    require(summary['validation_images_read'] == summary['test_images_read'] == summary['optimizer_steps'] == 0 and
            summary['parameters_and_BN_buffers_unchanged'] and summary['upstream_files_unchanged'],
            'Unexpected source run changes or split usage.')
    files.update({str(path): d.sha256(path) for path in paths})
    snapshots = []
    for row, old, entry in zip(chosen, old_entries, state['entries']):
        require(row['id_code'] == old['id_code'] == entry['id_code'] and
                int(row['diagnosis']) == old['label'] == entry['label'] and
                old['prediction'] == entry['prediction'] == entry['target_class'], 'Source image/class mismatch.')
        rolled = ar.load_saved_evidence(rollout_folder, old); related = gr.load_evidence(folder, entry)
        for key in ('labels', 'rgb_uint8'):
            np.testing.assert_array_equal(rolled[key], related[key])
        require(str(related['variant']) == variant, 'Wrong cached variant.')
        np.testing.assert_allclose(related['logits'], rolled['logits'], atol=5e-4, rtol=5e-4)
        graph = tr.rag.build_region_graph(related['labels']); tr.rag.validate_graph(graph)
        require(graph['num_nodes'] == entry['regions'] and len(graph['undirected_edges']) == entry['physical_pairs'],
                'Saved graph counts differ.')
        for key, graph_key in (('pairs', 'undirected_edges'), ('areas', 'areas'), ('shared_border_lengths', 'boundary_lengths')):
            np.testing.assert_array_equal(related[key], graph[graph_key])
        recalculated = gr.aggregate_relations(related['alpha'], related['alpha_gradient'], related['edge_index'], graph)
        np.testing.assert_allclose(recalculated['pair_signed'], related['pair_signed'], atol=1e-12, rtol=1e-12)
        require(int(related['target_class']) == int(np.argmax(related['logits'])) == entry['target_class'], 'Wrong target.')
        require(np.isfinite(rolled['region_raw']).all() and (rolled['region_raw'] >= 0).all(), 'Invalid regional ranking.')
        for name, digest in entry['files'].items():
            files[str(folder / name)] = digest
        spatial = spatial_profile(rolled['rgb_uint8'], rolled['labels'], graph)
        jobs = make_plan(row['id_code'], variant, rolled, related, graph, spatial)
        snapshots.append({'row': row, 'entry': entry, 'rolled': rolled, 'related': related,
                          'graph': graph, 'spatial': spatial, 'jobs': jobs})
    return project, info, files, codes, snapshots, {'rollout': rollout_signature, 'relations': signature}


def technical_checks():
    # A dark island inside a bright image must not be called border-connected.
    rgb = np.full((8, 8, 3), 100, np.uint8); rgb[0] = 7; rgb[3:5, 3:5] = 0
    rgb[1, 0] = 8
    proxy = border_dark_proxy(rgb)
    require(proxy[0].all() and not proxy[3:5, 3:5].any() and not proxy[1, 0], 'Dark connectivity/threshold error.')
    labels = np.repeat(np.repeat(np.arange(4).reshape(2, 2), 4, axis=0), 4, axis=1).astype(np.int32)
    graph = tr.rag.build_region_graph(labels); spatial = spatial_profile(rgb, labels, graph)
    np.testing.assert_allclose(spatial['dark_fraction'], [4/16, 4/16, 0, 0], atol=0, rtol=0)
    pixels = rgb.astype(np.float32)/255; original = pixels.copy()
    for baseline in ('mean', 'black'):
        altered, details = replace_regions(pixels, labels, [1], baseline)
        mask = labels == 1
        np.testing.assert_array_equal(altered[~mask], pixels[~mask])
        expected = pixels.mean((0, 1), dtype=np.float64).astype(np.float32) if baseline == 'mean' else np.zeros(3)
        np.testing.assert_allclose(altered[mask], np.broadcast_to(expected, (16, 3)), atol=0, rtol=0)
        require(details['selected_pixels'] == 16, 'Whole-region replacement area error.')
    np.testing.assert_array_equal(pixels, original)
    dark_rgb = pixels.copy(); dark_rgb[labels == 1] = 0
    _, details = replace_regions(dark_rgb, labels, [1], 'black')
    require(details['pixel_noop'] and details['changed_pixels'] == 0, 'Black-on-black no-op was missed.')
    np.testing.assert_allclose(ar.as_numpy(normalise_rgb(pixels, torch.device('cpu')))[0],
                               (pixels.transpose(2, 0, 1)-np.array(d.MEAN)[:, None, None])/np.array(d.STD)[:, None, None],
                               atol=3e-7, rtol=1e-6)
    ranked = np.array([0, 2]); categories = np.array([0, 1, 0, 1, 2, 2]); areas = np.array([10, 11, 12, 13, 10, 12])
    chosen, meta = region_matched_draw(rng_for('test'), ranked, categories, areas)
    require(len(chosen) == 2 and len(set(chosen)) == 2 and (categories[chosen] == 0).all() and meta['match_ok'],
            'Matched random sampling failed.')
    signed = np.array([.2, -.4, .1, -.3])
    a = make_plan('synthetic', 'A3', {'region_raw': np.array([.4, .3, .2, .1])},
                  {'pair_signed': signed}, graph, spatial)
    b = make_plan('synthetic', 'A3', {'region_raw': np.array([.4, .3, .2, .1])},
                  {'pair_signed': signed}, graph, spatial)
    require(a == b, 'Random controls are not reproducible.')
    for job in a:
        if job['domain'] == 'edge' and job['control'] in ('ranked', 'random_matched', 'single'):
            require((signed[job['selected']] * job['direction'] > 0).all(), 'Positive/negative controls mixed signs.')
    result = output_metrics(np.array([0., 3., 0., 0., 0.]), np.array([2., 0., 0., 0., 0.]), 0)
    require(result['target_class'] == 0 and result['prediction'] == 1 and result['logit_drop'] == 2. and
            result['probability_drop'] > 0, 'Target followed perturbed argmax instead of original class.')
    return {'border_connected_dark_proxy': True, 'whole_region_fill_and_preserved_other_pixels': True,
            'black_on_black_noop_reported': True, 'normalisation_reference': True,
            'deterministic_count_area_category_controls': True, 'separate_sign_controls': True,
            'original_target_fixed_after_prediction_change': True}


class Progress(ar.Progress):
    def show(self):
        done = self.done-self.initial
        eta = ev.clock((time.perf_counter()-self.phase_started)/done*(self.total-self.done)) if done else '--'
        if self.total and self.done == self.total:
            eta = '00:00:00'
        print(f'Faithfulness | {self.stage} | Experiments {self.done}/{self.total} | '
              f'Elapsed {ev.clock(time.perf_counter()-STARTED)} | ETA {eta}', flush=True)


class OutputLock(ar.OutputLock):
    def __init__(self, output):
        self.path = output / 'faithfulness.lock'

    def __enter__(self):
        try:
            return super().__enter__()
        except ValueError as exc:
            raise ValueError('Another process is using this faithfulness output folder.') from exc


def save_payload(path, payload):
    ev.write_json(path, {'payload': payload, 'checksum': tr.fingerprint(payload)})


def load_payload(path):
    saved = ev.read_json(path)
    require(saved['checksum'] == tr.fingerprint(saved['payload']), 'Saved diagnostic state was changed or corrupted.')
    return saved['payload']


def intervention_key(job):
    return tr.fingerprint([job['domain'], job['baseline'], job['selected']])


def load_image_state(path, signature, plan_hash, snapshot):
    if not path.is_file():
        return {'signature': signature, 'plan_sha256': plan_hash, 'id_code': snapshot['row']['id_code'],
                'baseline_logits': None, 'baseline_checks': None, 'results': [], 'status': 'RUNNING'}
    payload = load_payload(path); jobs = snapshot['jobs']
    require(payload['signature'] == signature and payload['plan_sha256'] == plan_hash and
            payload['id_code'] == snapshot['row']['id_code'] and len(payload['results']) <= len(jobs),
            'Saved diagnostic state has a different experiment identity.')
    for job, result in zip(jobs, payload['results']):
        require(result['job_id'] == job['job_id'] and result['input_key'] == intervention_key(job), 'Saved experiment order changed.')
        calculated = output_metrics(result['logits'], payload['baseline_logits'], snapshot['entry']['target_class'])
        for key in ('target_class', 'prediction', 'prediction_changed'):
            require(result[key] == calculated[key], 'Saved target/prediction identity changed.')
        for key in ('probabilities', 'logit_drop', 'probability_drop'):
            np.testing.assert_allclose(result[key], calculated[key], atol=1e-12, rtol=1e-12)
    require(not payload['results'] or payload['baseline_checks']['baseline_prediction_preserved'], 'Missing baseline evidence.')
    return payload


def save_npz(path, arrays):
    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def save_csv(path, rows, fields=None):
    require(rows, 'No rows to write.')
    fields = fields or list(rows[0]); temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(row.get(k), separators=(',', ':')) if isinstance(row.get(k), (list, dict))
                                 else row.get(k) for k in fields})
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'mean': None, 'sd': None, 'p05': None, 'p95': None}
    return {'mean': float(values.mean()), 'sd': float(values.std()),
            'p05': float(np.quantile(values, .05)), 'p95': float(np.quantile(values, .95))}


def comparisons(snapshot, payload):
    jobs = snapshot['jobs']; results = payload['results']; rows = []
    for job, result in zip(jobs, results):
        if job['control'] != 'ranked':
            continue
        direction = job['direction']
        row = {'id_code': snapshot['row']['id_code'], 'true_grade': int(snapshot['row']['diagnosis']),
               'target_class': snapshot['entry']['target_class'], 'family': job['family'], 'baseline': job['baseline'],
               'oriented_effect_definition': 'probability/logit rise' if direction == -1 else 'probability/logit drop',
               'fraction_requested': job['fraction'], 'count': job['count'], 'actual_count_fraction': job['actual_count_fraction'],
               'ranked_proxy_group_counts_O_T_B': job['group_counts'],
               'ranked_logit_drop': result['logit_drop'], 'ranked_probability_drop': result['probability_drop'],
               'ranked_oriented_logit_effect': direction*result['logit_drop'],
               'ranked_oriented_probability_effect': direction*result['probability_drop'],
               'ranked_selected_pixel_fraction': result['selected_pixel_fraction'],
               'ranked_changed_pixel_fraction': result['changed_pixel_fraction'], 'ranked_pixel_noop': result['pixel_noop']}
        for control in ('random_all', 'random_matched'):
            pool = [(j, r) for j, r in zip(jobs, results) if j['family'] == job['family'] and
                    j['baseline'] == job['baseline'] and j['fraction'] == job['fraction'] and j['control'] == control]
            valid = [(j, r) for j, r in pool if j['match_ok']]
            row[control+'_n'] = len(valid); row[control+'_excluded_area_mismatches'] = len(pool)-len(valid)
            row[control+'_unique_interventions'] = len({r['input_key'] for _, r in valid})
            row[control+'_ranked_set_overlap_mean'] = float(np.mean([len(set(j['selected']) & set(job['selected']))/job['count']
                                                                    for j, _ in valid])) if valid else None
            for measure in ('logit_drop', 'probability_drop'):
                stats = statistics([direction*r[measure] for _, r in valid])
                effect_name = measure.replace('_drop', '_effect')
                for key, value in stats.items():
                    row[control+'_oriented_'+effect_name+'_'+key] = value
                row['ranked_minus_'+control+'_oriented_'+effect_name] = direction*result[measure]-stats['mean'] if valid else None
            row[control+'_selected_pixel_fraction_mean'] = float(np.mean([r['selected_pixel_fraction'] for _, r in valid])) if valid else None
            row[control+'_changed_pixel_fraction_mean'] = float(np.mean([r['changed_pixel_fraction'] for _, r in valid])) if valid else None
        rows.append(row)
    return rows


def save_figures(output, variant, snapshots, payloads, comparison_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    columns = [('edge_positive', 'none', 'Positive pairs: probability decrease'),
               ('edge_negative', 'none', 'Negative pairs: probability increase'),
               ('region', 'mean', 'Regions / image-mean fill: decrease'),
               ('region', 'black', 'Regions / black fill: decrease')]
    fig, axes = plt.subplots(len(snapshots), 4, figsize=(17, 3*len(snapshots)), squeeze=False)
    for index, snapshot in enumerate(snapshots):
        image_id = snapshot['row']['id_code']
        for col, (family, baseline, title) in enumerate(columns):
            ax = axes[index, col]
            rows = sorted([r for r in comparison_rows if r['id_code'] == image_id and
                           r['family'] == family and r['baseline'] == baseline], key=lambda r: r['fraction_requested'])
            ax.axhline(0, color='#64748b', linewidth=.7)
            if rows:
                x = np.array([r['actual_count_fraction']*100 for r in rows])
                ax.plot(x, [r['ranked_oriented_probability_effect'] for r in rows], 'o-', color='#111827', label='Ranked')
                for control, label, color in [('random_all', 'Random / count', '#0f766e'),
                                               ('random_matched', 'Random / matched', '#c2410c')]:
                    def values(suffix):
                        return [np.nan if r[control+'_oriented_probability_effect_'+suffix] is None
                                else r[control+'_oriented_probability_effect_'+suffix] for r in rows]
                    ax.plot(x, values('mean'), 'o--', color=color, label=label, markersize=3)
                    ax.fill_between(x, values('p05'), values('p95'), color=color, alpha=.13)
            else:
                ax.text(.5, .5, 'No pairs of this sign', transform=ax.transAxes, ha='center')
            ax.set_title(title if index == 0 else '')
            ax.set_xlabel('Selected pairs (%)' if family.startswith('edge') else 'Selected regions (%)')
            if col == 0:
                ax.set_ylabel(f'{image_id}\nTrue {snapshot["row"]["diagnosis"]} / target {snapshot["entry"]["target_class"]}\nProbability change')
            ax.grid(alpha=.16)
    axes[0, 0].legend(loc='best', fontsize=7)
    fig.suptitle(f'{variant} | Frozen-model faithfulness diagnostics | Five training cases', fontsize=14)
    fig.text(.5, .012, 'Bands: 5th-95th percentiles of random interventions, not confidence intervals. '
             'Negative values are retained.\n'
             'Y = target probability decrease for positive pairs/regions; increase for negative pairs. '
             'Local logit-gradient signs do not guarantee probability direction.\n'
             'Region x-axis counts superpixels, not pixels; actual pixel areas and failed matches are in the CSV. '
             'These plots do not establish lesion localisation.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .965))
    curve = output / 'faithfulness_curves_v1.png'
    temporary = curve.with_name(curve.name+'.tmp'); fig.savefig(temporary, format='png', dpi=130)
    temporary.replace(curve); plt.close(fig)

    fig, axes = plt.subplots(len(snapshots), 3, figsize=(14, 3.5*len(snapshots)), squeeze=False)
    colours = ['#64748b', '#e67e22', '#b52e98']
    for index, (snapshot, payload) in enumerate(zip(snapshots, payloads)):
        rgb = snapshot['rolled']['rgb_uint8']; labels = snapshot['rolled']['labels']; spatial = snapshot['spatial']
        axes[index, 0].imshow(rgb); axes[index, 0].axis('off')
        axes[index, 0].set_title(f'{snapshot["row"]["id_code"]} | true {snapshot["row"]["diagnosis"]} / '
                                f'target {snapshot["entry"]["target_class"]}')
        overlay = rgb.astype(np.float64)/255
        for category, colour in ((1, [1., .5, 0.]), (2, [.8, .05, .65])):
            mask = spatial['region_group'][labels] == category
            overlay[mask] = .60*overlay[mask] + .40*np.asarray(colour)
        axes[index, 1].imshow(overlay); axes[index, 1].axis('off')
        axes[index, 1].set_title('Region proxy: orange=T, magenta=B; O unchanged')
        singles = [(j, r) for j, r in zip(snapshot['jobs'], payload['results']) if j['control'] == 'single']
        ax = axes[index, 2]; ticks = []; effects = []; bar_colours = []
        for job, result in singles:
            pair_id = job['selected'][0]; u, v = snapshot['graph']['undirected_edges'][pair_id]
            group = int(spatial['pair_group'][pair_id]); mark = '+' if job['direction'] == 1 else '-'
            ticks.append(f'{mark}{job["rank"]}: {u}-{v} [{"OTB"[group]}]')
            effects.append(result['probability_drop']); bar_colours.append(colours[group])
        ax.barh(np.arange(len(singles)), effects, color=bar_colours)
        ax.set_yticks(np.arange(len(singles)), ticks, fontsize=7); ax.invert_yaxis(); ax.axvline(0, color='black', linewidth=.7)
        ax.set_title('Individual top-ranked pairs: actual suppression effect')
        ax.set_xlabel('Original target probability - perturbed probability')
        ax.grid(axis='x', alpha=.16)
    fig.suptitle(f'{variant} | Border-contact context and single-pair effects', fontsize=14)
    fig.text(.5, .012, 'Proxy: near-black pixels (max RGB <=7) connected to canvas border. B: region >=95% proxy; '
             'T: other detected contact; O: none detected.\n'
             'Pairs: B if both endpoints are B; T if otherwise any contact; O if neither. '
             'This is not a retinal or lesion mask.\n'
             'Pair + / - is the original local logit-sensitivity sign; a finite suppression need not follow it. '
             'Only the original top five of each sign are shown.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .965))
    spatial_path = output / 'faithfulness_spatial_v1.jpg'
    temporary = spatial_path.with_name(spatial_path.name+'.tmp'); fig.savefig(temporary, format='jpg', dpi=130)
    temporary.replace(spatial_path); plt.close(fig)
    return [curve, spatial_path]


def export_evidence(output, variant, snapshots, payloads):
    trials = []; pair_rows = []; region_rows = []; comparison_rows = []; image_summaries = []; paths = []
    for snapshot, payload in zip(snapshots, payloads):
        row = snapshot['row']; image_id = row['id_code']; graph = snapshot['graph']; spatial = snapshot['spatial']
        require(len(payload['results']) == len(snapshot['jobs']), 'Cannot export an incomplete image.')
        by_pair = {j['selected'][0]: (j, r) for j, r in zip(snapshot['jobs'], payload['results']) if j['control'] == 'single'}
        for job, result in zip(snapshot['jobs'], payload['results']):
            trials.append({'id_code': image_id, 'true_grade': int(row['diagnosis']), **job, **result})
        comparison_rows.extend(comparisons(snapshot, payload))
        for pair_id, (u, v) in enumerate(graph['undirected_edges']):
            single = by_pair.get(pair_id)
            pair_rows.append({'id_code': image_id, 'target_class': snapshot['entry']['target_class'],
                              'pair_id': pair_id, 'u': int(u), 'v': int(v),
                              'signed_local_logit_sensitivity': float(snapshot['related']['pair_signed'][pair_id]),
                              'proxy_category': GROUPS[int(spatial['pair_group'][pair_id])],
                              'u_dark_fraction': float(spatial['dark_fraction'][u]),
                              'v_dark_fraction': float(spatial['dark_fraction'][v]),
                              'shared_border_length': int(graph['boundary_lengths'][pair_id]),
                              'individual_probability_drop': single[1]['probability_drop'] if single else None,
                              'individual_logit_drop': single[1]['logit_drop'] if single else None,
                              'individual_rank_within_sign': single[0]['rank'] if single else None})
        order = np.lexsort((np.arange(graph['num_nodes']), -snapshot['rolled']['region_raw']))
        for rank, region_id in enumerate(order, 1):
            region_rows.append({'id_code': image_id, 'region_id': int(region_id), 'rollout_rank': rank,
                                'rollout_raw_score': float(snapshot['rolled']['region_raw'][region_id]),
                                'area_pixels': int(graph['areas'][region_id]),
                                'border_dark_fraction': float(spatial['dark_fraction'][region_id]),
                                'proxy_category': GROUPS[int(spatial['region_group'][region_id])]})
        spatial_path = output / f'{image_id}_spatial_v1.npz'
        save_npz(spatial_path, {'labels': snapshot['rolled']['labels'], 'border_dark_proxy': spatial['proxy'],
                               'region_dark_fraction': spatial['dark_fraction'], 'region_category': spatial['region_group'],
                               'pair_category': spatial['pair_group'], 'pairs': graph['undirected_edges']})
        paths.append(spatial_path)
        actual_regions = [(j, r) for j, r in zip(snapshot['jobs'], payload['results']) if j['domain'] == 'region']
        image_summaries.append({'id_code': image_id, 'true_grade': int(row['diagnosis']),
                               'fixed_target': snapshot['entry']['target_class'], 'regions': graph['num_nodes'],
                               'physical_pairs': len(graph['undirected_edges']), 'experiments': len(snapshot['jobs']),
                               'baseline_logits': payload['baseline_logits'], 'baseline_checks': payload['baseline_checks'],
                               'region_proxy_counts_O_T_B': np.bincount(spatial['region_group'], minlength=3).tolist(),
                               'pair_proxy_counts_O_T_B': np.bincount(spatial['pair_group'], minlength=3).tolist(),
                               'detected_border_dark_pixel_fraction': float(spatial['proxy'].mean()),
                               'region_pixel_noop_experiments': sum(r['pixel_noop'] for _, r in actual_regions),
                               'excluded_matched_area_experiments': sum(not j['match_ok'] for j in snapshot['jobs']),
                               'unique_interventions': len({r['input_key'] for r in payload['results']}),
                               'note': 'No-op counts count region interventions only; edges leave RGB unchanged by design.'})
    for filename, rows in [('faithfulness_trials_v1.csv', trials), ('faithfulness_pairs_v1.csv', pair_rows),
                           ('faithfulness_regions_v1.csv', region_rows), ('faithfulness_comparison_v1.csv', comparison_rows)]:
        path = output / filename
        save_csv(path, rows, list(dict.fromkeys(key for row in rows for key in row)))
        paths.append(path)
    paths.extend(save_figures(output, variant, snapshots, payloads, comparison_rows))
    return paths, image_summaries, comparison_rows


def run_diagnostic(args, progress):
    root = args.root.resolve(); variant = args.variant
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA is unavailable.')
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads())); tr.seed_runtime(SEED)
    progress.stage = 'verifying frozen model, original rankings and fixed training cases'
    project, info, files, codes, snapshots, source_signatures = source_context(root, variant, progress)
    codes[str(Path(__file__).resolve())] = d.sha256(Path(__file__))
    checks = technical_checks()
    plan_hashes = {s['row']['id_code']: tr.fingerprint(s['jobs']) for s in snapshots}
    identity = {'version': VERSION, 'variant': variant, 'split': 'train', 'selected_ids': list(plan_hashes),
                'checkpoint_sha256': info['best_sha256'], 'best_epoch': info['summary']['best_epoch'],
                'source_signatures': source_signatures, 'protocol': protocol(), 'plan_sha256': plan_hashes,
                'data_sha256': project['source_sha256'], 'input_file_sha256': files, 'code_sha256': codes,
                'versions': ev.runtime_versions(), 'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    signature = tr.fingerprint(identity)
    output = root / 'environment/data_audit/faithfulness_v1' / f'train_diagnostic_{variant}_v1'
    config_path = output / 'faithfulness_config_v1.json'; state_path = output / 'faithfulness_state_v1.json'
    require(args.resume or not output.exists(), f'Output exists. Continue: python faithfulness_v1.py --variant {variant} --resume')
    require(not args.resume or config_path.is_file(), 'No faithfulness diagnostic to resume.')
    output.mkdir(parents=True, exist_ok=True)
    with OutputLock(output):
        if args.resume:
            old = ev.read_json(config_path)
            require(old['signature'] == signature and old['identity'] == identity,
                    'Resume model/data/code/protocol/environment differs; existing results were not overwritten.')
        else:
            ev.write_json(config_path, {'identity': identity, 'signature': signature, 'sources': SOURCES,
                                       'created_utc': datetime.now(timezone.utc).isoformat()})
        payloads = []; payload_paths = []
        for snapshot in snapshots:
            image_id = snapshot['row']['id_code']
            plan_path = output / f'{image_id}_plan_v1.json'
            plan = {'signature': signature, 'id_code': image_id, 'plan_sha256': plan_hashes[image_id], 'jobs': snapshot['jobs']}
            if plan_path.exists():
                require(load_payload(plan_path) == plan, 'Saved intervention plan differs.')
            else:
                save_payload(plan_path, plan)
            payload_path = output / f'{image_id}_state_v1.json'; payload_paths.append(payload_path)
            payloads.append(load_image_state(payload_path, signature, plan_hashes[image_id], snapshot))
        initial = sum(len(p['results']) for p in payloads); total = sum(len(s['jobs']) for s in snapshots)
        model = before = None; read_count = forward_calls = actual_interventions = reused = 0
        state = {'signature': signature, 'status': 'RUNNING', 'completed_experiments': initial, 'total_experiments': total,
                 'image_state_files': [p.name for p in payload_paths], 'error': None}
        ev.write_json(state_path, state)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        progress.reset(f'{variant} fixed training perturbations', total=total, done=initial)
        active_payload = active_path = None
        try:
            if initial < total:
                progress.stage = f'loading trained {variant}, best epoch {info["summary"]["best_epoch"]}'
                model, before = ev.load_best_model(info, device)
                dataset = d.AptosDataset(project, 'train', verify_file_hash=True)
                dataset.rows = [s['row'] for s in snapshots]; dataset.augment = None
            for index, (snapshot, payload, payload_path) in enumerate(zip(snapshots, payloads, payload_paths)):
                if len(payload['results']) == len(snapshot['jobs']):
                    if payload['status'] != 'COMPLETE':
                        payload['status'] = 'COMPLETE'; save_payload(payload_path, payload)
                    continue
                active_payload, active_path = payload, payload_path; image_id = snapshot['row']['id_code']
                progress.stage = f'{variant} | {image_id} | validating original prediction and cached graph tail'
                sample = dataset[index]; read_count += 1
                require(sample['id_code'] == image_id and sample['split'] == 'train' and
                        int(sample['label']) == int(snapshot['row']['diagnosis']), 'Wrong image retrieved.')
                rgb = np.rint(ar.as_numpy(sample['rgb']).transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)
                np.testing.assert_array_equal(rgb, snapshot['rolled']['rgb_uint8'])
                runner = FrozenRunner(model, sample, snapshot)
                if payload['baseline_logits'] is None:
                    payload.update(baseline_logits=runner.baseline.tolist(), baseline_checks=runner.checks)
                else:
                    np.testing.assert_allclose(runner.baseline, payload['baseline_logits'], atol=5e-4, rtol=5e-4)
                    runner.baseline = np.asarray(payload['baseline_logits'], dtype=np.float64)
                payload['status'] = 'RUNNING'; save_payload(payload_path, payload)
                cache = {r['input_key']: {k: v for k, v in r.items() if k not in
                         ('job_id', 'input_key', 'reused_identical_intervention')} for r in payload['results']}
                for job in snapshot['jobs'][len(payload['results']):]:
                    progress.stage = f'{variant} | {image_id} | {job["family"]} / {job["baseline"]} / {job["control"]}'
                    key = intervention_key(job); was_reused = key in cache
                    if not was_reused:
                        cache[key] = runner.execute(job); actual_interventions += 1
                    else:
                        reused += 1
                    record = {'job_id': job['job_id'], 'input_key': key, **cache[key],
                              'reused_identical_intervention': was_reused}
                    payload['results'].append(record)
                    try:
                        save_payload(payload_path, payload)
                    except BaseException:
                        payload['results'].pop()
                        raise
                    progress.done += 1
                forward_calls += runner.calls
                payload['status'] = 'COMPLETE'; save_payload(payload_path, payload)
                state['completed_experiments'] = progress.done; ev.write_json(state_path, state)
                print(f'{image_id} | {len(payload["results"])}/{len(snapshot["jobs"])} experiments | '
                      f'original target={snapshot["entry"]["target_class"]} | saved', flush=True)
                active_payload = active_path = None
                del runner
            if model is not None:
                tr.assert_tree_equal(before, tr.cpu_tree(model.state_dict()))
                require(all(p.grad is None for p in model.parameters()), 'Unexpected parameter gradients.')
            progress.stage = 'verifying unchanged inputs and writing comparison figures'
            ev.verify_sources(project, files, codes)
            for snapshot in snapshots:
                image_id = snapshot['row']['id_code']
                require(d.sha256(root / 'processed_images_384_v1' / (image_id+'.png')) ==
                        project['processed'][image_id]['output_sha256'], 'Selected PNG changed.')
            paths, image_summaries, comparison_rows = export_evidence(output, variant, snapshots, payloads)
            paths += payload_paths + [output / f'{image_id}_plan_v1.json' for image_id in plan_hashes]
            summary = {'status': 'FAITHFULNESS_DIAGNOSTIC_COMPLETE', 'technical_status': 'PASS',
                       'faithfulness_status': 'EMPIRICAL_RESULTS_REVIEW_REQUIRED', 'version': VERSION, 'signature': signature,
                       'variant': variant, 'best_epoch': info['summary']['best_epoch'], 'checkpoint_sha256': info['best_sha256'],
                       'source_signatures': source_signatures, 'protocol': protocol(), 'sources': SOURCES,
                       'device': str(device), 'train_images': len(snapshots), 'train_PNGs_decoded_this_process': read_count,
                       'validation_images_read': 0, 'test_images_read': 0, 'optimizer_steps': 0,
                       'total_experiments': total, 'resumed_experiments': initial,
                       'new_experiments_this_process': total-initial, 'actual_interventions_this_process': actual_interventions,
                       'identical_interventions_reused_this_process': reused,
                       'neural_prediction_forwards_this_process_including_baseline_checks': forward_calls,
                       'parameters_and_BN_buffers_unchanged': True, 'parameter_gradients_accumulated': False,
                       'upstream_files_unchanged': True, 'original_partitions_and_graph_topologies_preserved': True,
                       'technical_checks': checks, 'images': image_summaries, 'comparisons': comparison_rows,
                       'files_sha256': {p.name: d.sha256(p) for p in paths},
                       'elapsed_seconds_this_process': time.perf_counter()-STARTED,
                       'CUDA_peak_allocated_MiB': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
                       'limits': ['Five selected training cases only; not independent generalisation evidence.',
                                  'Rankings fixed from the original model; original predicted class remains the target.',
                                  'Rollout is not class-conditional; evaluate its original score ranking without relabelling it.',
                                  'Finite gates can differ from local logit-gradient signs; probability can also change differently from logits. '
                                  'Retain signed raw logit/probability drops and separately labelled oriented comparisons.',
                                  'Pixel replacement can introduce out-of-distribution artefacts; two fills do not eliminate this.',
                                  'Matched controls condition on proxy categories; tiny pools/overlap reduce comparison diversity.',
                                  'Black-on-black no-op cannot establish that background information is unimportant.',
                                  'No lesion ground truth or clinical validation; faithful behaviour may still reflect background reliance.',
                                  'These budgets/random repetitions are project protocol choices, not paper-prescribed constants.']}
            ev.write_json(output / 'faithfulness_summary_v1.json', summary)
            state.update(status='COMPLETE', completed_experiments=total, error=None); ev.write_json(state_path, state)
        except (Exception, KeyboardInterrupt) as exc:
            if active_payload is not None:
                active_payload['status'] = 'STOPPED'; save_payload(active_path, active_payload)
            state.update(status='STOPPED', completed_experiments=sum(len(p['results']) for p in payloads),
                         error=f'{type(exc).__name__}: {exc}')
            ev.write_json(state_path, state)
            raise
    print(f'\nFAITHFULNESS DIAGNOSTIC COMPLETE | {variant} | {len(snapshots)} training cases | {total} experiments')
    print('TECHNICAL CHECK: PASS | Faithfulness result: AWAITING EMPIRICAL REVIEW')
    print('Original targets / partitions / graph topology preserved | Frozen parameters / BN / source files unchanged: PASS')
    print('Validation/test images read: 0 | Optimizer steps: 0 | No lesion-localisation claim')
    print(f'Total elapsed this process: {ev.clock(time.perf_counter()-STARTED)}\nSaved: {output}')
    print('Return faithfulness_summary_v1.json, faithfulness_curves_v1.png and faithfulness_spatial_v1.jpg.')
    print('After review: remember to commit faithfulness_v1.py.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--variant', choices=('A2', 'A3'), default='A3')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--self-test', action='store_true', help='Only synthetic numerical references; no project images/checkpoints.')
    args = parser.parse_args()
    try:
        with Progress() as progress:
            if args.self_test:
                print(technical_checks())
                print('FAITHFULNESS NUMERICAL CHECK: PASS | Project images/checkpoints read: 0')
            else:
                run_diagnostic(args, progress)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'\nFAITHFULNESS STOPPED | {type(exc).__name__}: {exc}', flush=True)
        print(f'If a diagnostic was started, continue: python faithfulness_v1.py --variant {args.variant} --resume')
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
