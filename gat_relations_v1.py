r"""CP2 proposal 3.9.2: predicted-class sensitivity of GAT neighbour pairs.

    python gat_relations_v1.py
    python gat_relations_v1.py --resume

Uses the same five TRAIN images and exact superpixel maps as the completed
attention_rollout_v1 preview. Default A3, validation-best epoch21. Supports A2
after its corresponding rollout preview exists. A0/A1 have no GAT to explain.

Score each directed edge/head as alpha * d(predicted-class logit)/d(alpha).
SUM heads and both directions of a physical pair, keeping the signed result.
Rank positive and negative pair scores separately; exclude self-loops from
physical-pair rankings. These are local sensitivities, not causal effects.

The derivative equals the sensitivity to a shared multiplicative message gate
on that pair at gate=1, AFTER attention softmax, WITHOUT renormalisation.
The reusable forward_with_pair_gates API provides this precise intervention
for later faithfulness work. It is different from deleting an edge before
softmax. No faithfulness benchmark, training or test-image use occurs here.

Existing model code/weights stay unchanged. A temporary PyG hook makes the
actual attention tensor differentiable while keeping all model weights frozen.
Progress/elapsed/ETA and atomic per-image saves support interruption/resume.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
from pathlib import Path
import time

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading GAT-relation explanation libraries...', flush=True)
import numpy as np
import torch
import attention_rollout_v1 as ar

ev, tr, d = ar.ev, ar.tr, ar.d
require = ar.require
VERSION = 'gat_relations_v1'
ROLLOUT_CODE_SHA256 = 'abf98a5f174d8e7730f52873f630d80a548549e99c4e3701f9d9e5aa882ae98e'
DISPLAY_PAIRS = 5
METHOD = {
    'target': 'argmax of original five-class logits, fixed before differentiation; true label is not the target',
    'scalar': 'predicted-class pre-softmax logit',
    'attention': 'actual eval-mode GAT coefficients, after neighbour+self softmax; dropout disabled',
    'directed_head_score': 'alpha[e,h] * partial(target_logit)/partial(alpha[e,h])',
    'pair_score': 'signed SUM over four heads and both physical edge directions; clip only AFTER summation',
    'positive_ranking': 'descending positive pair_score, stable region-ID tie breaking',
    'negative_ranking': 'descending magnitude of negative pair_score, stable region-ID tie breaking',
    'self_loops': 'retained in model and numerical export; excluded from physical-pair rankings and display',
    'gate_interpretation': 'same scalar gate for both directions/all heads of a pair, applied AFTER attention '
                           'softmax without renormalisation; pair_score equals d(logit)/d(gate) at one',
    'gate_limits': 'local derivative is not the measured effect of setting the gate to zero; no completeness claim',
    'topology': 'unchanged four-neighbour shared-pixel-side graph from the exact saved rollout segmentation',
    'geometry': 'display anchors are region pixels nearest each centroid; links denote adjacency, not vessels',
    'display': 'up to five positive and five negative pairs; endpoint regions, shared border and links highlighted',
    'display_scale': 'line strength normalised by largest magnitude within each sign/image; not cross-image scores',
    'background': 'full canvas retained; no FOV exclusion or background filtering',
    'precision': 'frozen float32 model and autograd; score aggregation in CPU float64',
    'scope': 'prediction-conditioned local GAT message sensitivity; not lesion detection or causal explanation',
}
SOURCES = {
    'proposal': 'CP1 3.9.2, printed46-47/PDF52-53; attention plus predicted-class sensitivity of physical neighbours.',
    'gat': 'Velickovic et al. (2018). Graph Attention Networks. ICLR. https://arxiv.org/abs/1710.10903',
    'gradient_times_input': 'Ancona et al. (2018). Towards better understanding of gradient-based attribution '
                           'methods for Deep Neural Networks. ICLR. https://arxiv.org/abs/1711.06104',
    'graph_explainability': 'Baldassarre & Azizpour (2019). Explainability Techniques for Graph Convolutional '
                           'Networks. ICML Workshop on Learning and Reasoning with Graph-Structured Data. '
                           'https://arxiv.org/abs/1905.13686',
    'pyg_source': 'https://pytorch-geometric.readthedocs.io/en/stable/_modules/torch_geometric/nn/conv/gat_conv.html',
    'pyg_hooks': 'https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.MessagePassing.html',
    'attribution_scope': 'This project adapts the signed gradient-times-activation principle to GAT attention. '
                         'The pair/head aggregation, gate definition and preview are explicit project choices. '
                         'This is not a reproduction of a named DR method, GNNExplainer, Grad-CAM or integrated gradients.',
}


def edge_pair_indices(edge_index, graph):
    """Map arbitrary message-edge order to physical pairs; self-loop index=-1."""
    edges = ar.as_numpy(edge_index)
    n = graph['num_nodes']; pairs = graph['undirected_edges']
    require(edges.ndim == 2 and edges.shape[0] == 2 and np.issubdtype(edges.dtype, np.integer),
            'Expected integer message edges [2,E].')
    require(np.all((edges >= 0) & (edges < n)), 'Message endpoint outside this image graph.')
    expected = set(map(tuple, graph['edge_index'].T.tolist())) | {(i, i) for i in range(n)}
    actual = list(map(tuple, edges.T.tolist()))
    require(len(actual) == len(expected) and set(actual) == expected,
            'Message graph differs from physical neighbours plus one self-loop per node.')
    lookup = {tuple(pair): i for i, pair in enumerate(pairs.tolist())}
    indices = np.array([-1 if u == v else lookup[tuple(sorted((u, v)))] for u, v in actual], dtype=np.int64)
    require(np.array_equal(np.bincount(indices[indices >= 0], minlength=len(pairs)), np.full(len(pairs), 2)),
            'Each physical pair must have exactly two message directions.')
    return indices


class AttentionProbe:
    """Temporary post-softmax probe; removed on success, error or interruption.

    capture mode exposes alpha as an independent differentiable activation.
    gate mode multiplies alpha by pair_gates, leaving self coefficients intact.
    Features, graph, softmax denominator and model weights remain fixed. Hooks
    require eager PyG execution; a missing hook invocation is an explicit error.
    """
    def __init__(self, conv, graph, pair_gates=None):
        require(conv is not None and not conv.training, 'An eval-mode GAT is required.')
        self.conv, self.graph, self.gates = conv, graph, pair_gates
        self.calls = 0; self.alpha = self.edges = self.pair_index = None

    def hook(self, module, inputs, alpha):
        self.calls += 1
        require(self.calls == 1, 'This probe expects one GAT call for one image.')
        edges = inputs[0]
        mapping = edge_pair_indices(edges, self.graph)
        require(alpha.shape == (len(mapping), 4) and bool(torch.isfinite(alpha).all()) and
                bool((alpha >= 0).all()), 'Expected finite four-head attention.')
        self.edges = edges.detach().clone(); self.pair_index = mapping
        if self.gates is None:
            self.alpha = alpha.detach().requires_grad_(True)
        else:
            gates = self.gates
            require(isinstance(gates, torch.Tensor) and gates.shape == (len(self.graph['undirected_edges']),) and
                    gates.dtype == alpha.dtype and gates.device == alpha.device and
                    bool(torch.isfinite(gates).all() and (gates >= 0).all()), 'Invalid pair gate vector.')
            # A final constant-one slot protects self loops, even when all pairs are masked.
            safe_indices = np.where(mapping < 0, len(gates), mapping)
            scales = torch.cat((gates, gates.new_ones(1)))[torch.as_tensor(safe_indices, device=alpha.device)]
            self.alpha = alpha * scales[:, None]
        return self.alpha

    def __enter__(self):
        self.handle = self.conv.register_edge_update_forward_hook(self.hook)
        return self

    def __exit__(self, *args):
        self.handle.remove()

    def require_connected(self, result):
        attention = result['gat_attention']
        require(self.calls == 1 and attention is not None and attention['alpha'] is self.alpha and
                torch.equal(attention['edge_index'], self.edges), 'The probed alpha is not the actual message coefficient.')


def aggregate_relations(alpha, gradients, edge_index, graph):
    """Signed head/direction sum equals the derivative of one shared pair gate."""
    a = ar.as_numpy(alpha).astype(np.float64)
    g = ar.as_numpy(gradients).astype(np.float64)
    edges = ar.as_numpy(edge_index)
    index = edge_pair_indices(edges, graph)
    require(a.shape == g.shape == (len(index), 4) and np.isfinite(a).all() and
            np.isfinite(g).all() and np.all(a >= 0), 'Invalid attention/gradient arrays.')
    # This function scores the unmodified baseline, never a gated perturbation.
    sums = np.zeros((graph['num_nodes'], 4)); np.add.at(sums, edges[1], a)
    require(np.allclose(sums, 1, atol=5e-5, rtol=0), 'Baseline incoming attention does not sum to one.')
    head_score = a * g
    directed = head_score.sum(axis=1)
    physical = index >= 0
    signed = np.bincount(index[physical], weights=directed[physical], minlength=len(graph['undirected_edges']))
    means = np.bincount(index[physical], weights=a[physical].mean(1), minlength=len(signed)) / 2
    self_score = np.zeros(graph['num_nodes']); self_score[edges[0, ~physical]] = directed[~physical]
    return {'alpha': a, 'alpha_gradient': g, 'directed_head_score': head_score,
            'directed_score': directed, 'edge_index': edges.copy(), 'edge_pair_index': index,
            'pair_signed': signed, 'pair_positive': np.maximum(signed, 0),
            'pair_negative_magnitude': np.maximum(-signed, 0), 'pair_mean_attention': means,
            'self_loop_signed': self_score, 'incoming_attention_max_error': np.asarray(np.abs(sums - 1).max())}


def ranked_indices(scores, sign):
    require(sign in (-1, 1), 'Ranking sign must be +1 or -1.')
    magnitude = sign * np.asarray(scores)
    valid = np.flatnonzero(magnitude > 0)
    # Graph pairs are already lexicographically sorted; index breaks exact ties.
    return valid[np.lexsort((valid, -magnitude[valid]))]


def require_frozen_gat(model, cnn):
    require(model.variant in ('A2', 'A3') and model.use_gat, 'A0/A1 have no GAT relations to explain.')
    require(not any(m.training for m in model.modules()), 'Use model.eval() for explanations.')
    require(not any(p.requires_grad for p in model.parameters()), 'Model parameters must be frozen for this API.')
    require(cnn.shape == (1, 3, 384, 384) and cnn.dtype == torch.float32,
            'Expected one normalised float32 CNN image [1,3,384,384].')
    require(not torch.is_inference_mode_enabled(), 'Use normal/no_grad context, not torch.inference_mode().')


def explain_relations(model, cnn, label_map, *, target_class=None):
    """Reusable live explanation; does not consume diagnosis/ground-truth label.

    Defaults to the original predicted class. target_class may be set explicitly
    for controlled class-sensitivity analysis, not for selecting nicer maps.
    """
    require_frozen_gat(model, cnn)
    graph = tr.rag.build_region_graph(label_map)
    tr.rag.validate_graph(graph)
    with torch.no_grad():
        baseline = model(cnn, [label_map], segmentation_method=model.superpixel_method, return_attention=True)
    logits = baseline['logits'].detach()
    predicted = int(logits.argmax(-1)[0])
    target = predicted if target_class is None else target_class
    require(isinstance(target, int) and target in range(5), 'Target class must be 0..4.')
    with torch.enable_grad(), AttentionProbe(model.graph_stage.conv, graph) as probe:
        explained = model(cnn, [label_map], segmentation_method=model.superpixel_method, return_attention=True)
        probe.require_connected(explained)
        torch.testing.assert_close(explained['logits'].detach(), logits, atol=5e-4, rtol=5e-4)
        require(int(explained['logits'].argmax(-1)[0]) == predicted, 'Explanation instrumentation changed prediction.')
        torch.testing.assert_close(probe.alpha.detach(), baseline['gat_attention']['alpha'], atol=5e-5, rtol=5e-4)
        derivative = torch.autograd.grad(explained['logits'][0, target], probe.alpha, create_graph=False)[0]
        require(bool(torch.isfinite(derivative).all()), 'Nonfinite attention gradient.')
        scores = aggregate_relations(probe.alpha, derivative, probe.edges, graph)
        logit_error = float((explained['logits'].detach() - logits).abs().max())
    require(all(p.grad is None for p in model.parameters()), 'An explanation accumulated parameter gradients.')
    require(np.array_equal(ar.as_numpy(baseline['token_region_ids'][0]), np.arange(graph['num_nodes'] + 1) - 1),
            'GAT node/ViT token region IDs differ.')
    geometry = ar.as_numpy(baseline['geometry'])[0, :graph['num_nodes']].copy()
    positive = ranked_indices(scores['pair_signed'], 1)
    negative = ranked_indices(scores['pair_signed'], -1)
    def top(indices):
        return [{'pair_index': int(i), 'regions': graph['undirected_edges'][i].tolist(),
                 'signed_score': float(scores['pair_signed'][i])} for i in indices[:DISPLAY_PAIRS]]
    record = {'prediction': predicted, 'target_class': target,
              'probabilities': ar.as_numpy(logits[0].double().softmax(-1)).tolist(),
              'target_logit': float(logits[0, target]), 'regions': graph['num_nodes'],
              'physical_pairs': len(graph['undirected_edges']), 'message_edges_including_self': len(scores['edge_pair_index']),
              'positive_pairs': len(positive), 'negative_pairs': len(negative),
              'zero_pairs': int(np.count_nonzero(scores['pair_signed'] == 0)),
              'max_abs_pair_score': float(np.max(np.abs(scores['pair_signed']), initial=0)),
              'all_relation_gradients_zero': bool(not np.any(scores['pair_signed'])),
              'instrumentation_max_logit_difference': logit_error,
              'incoming_attention_max_error': float(scores['incoming_attention_max_error']),
              'prediction_preserved': True, 'physical_pair_alignment': True,
              'top_positive': top(positive), 'top_negative': top(negative)}
    evidence = {**scores, 'pairs': graph['undirected_edges'], 'shared_border_lengths': graph['boundary_lengths'],
                'areas': graph['areas'], 'geometry': geometry, 'labels': np.asarray(label_map).copy(),
                'logits': ar.as_numpy(logits[0]).copy(), 'probabilities': np.array(record['probabilities']),
                'target_class': np.asarray(target, dtype=np.int64), 'prediction': np.asarray(predicted, dtype=np.int64),
                'positive_order': positive, 'negative_order': negative}
    return evidence, record


def forward_with_pair_gates(model, cnn, label_map, pair_gates):
    """Later edge perturbations: same pair gate for both directions/all heads.

    Gates are nonnegative; ones reproduce baseline and zeros suppress messages.
    Values slightly above one are allowed for central finite-difference checks.
    Self loops remain one. This does not delete topology or renormalise alpha.
    The caller selects no_grad for evaluation, or enable_grad for gate gradients.
    """
    require_frozen_gat(model, cnn)
    graph = tr.rag.build_region_graph(label_map)
    with AttentionProbe(model.graph_stage.conv, graph, pair_gates) as probe:
        output = model(cnn, [label_map], segmentation_method=model.superpixel_method, return_attention=True)
        probe.require_connected(output)
    return output


def technical_checks(device):
    """Independent gradient and numerical gate references; synthetic data only."""
    labels = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int32)
    graph = tr.rag.build_region_graph(labels)
    generator = torch.Generator().manual_seed(327)
    x = torch.randn(4, 256, generator=generator, dtype=torch.float64) * .3
    readout = torch.randn(5, 4, 256, generator=generator, dtype=torch.float64) * .2
    stage = tr.gat.LocalGraphStage('A3').double().eval().requires_grad_(False)
    before = tr.cpu_tree(stage.state_dict())
    packed = {'x': x, 'edge_index': torch.from_numpy(graph['edge_index']), 'ptr': torch.tensor([0, 4])}
    with torch.no_grad():
        baseline = stage(packed, True)
    with torch.enable_grad(), AttentionProbe(stage.conv, graph) as probe:
        output = stage(packed, True)
        require(output['attention']['alpha'] is probe.alpha, 'GAT probe disconnected.')
        target = 2
        scalar = (output['x'] * readout[target]).sum()
        derivative = torch.autograd.grad(scalar, probe.alpha, retain_graph=True)[0]
        other = torch.autograd.grad((output['x'] * readout[1]).sum(), probe.alpha)[0]
        scores = aggregate_relations(probe.alpha, derivative, probe.edges, graph)
        other_scores = aggregate_relations(probe.alpha, other, probe.edges, graph)
    torch.testing.assert_close(output['x'].detach(), baseline['x'], atol=0, rtol=0)
    # Independent NumPy messages + ELU derivative, with explicit source/target loops.
    z = (x.numpy() @ stage.conv.lin.weight.detach().numpy().T).reshape(4, 4, 64)
    edges = scores['edge_index']; a = scores['alpha']
    preactivation = np.broadcast_to(stage.conv.bias.detach().numpy().reshape(4, 64), (4, 4, 64)).copy()
    for e, (source, destination) in enumerate(edges.T):
        for h in range(4):
            preactivation[destination, h] += a[e, h] * z[source, h]
    elu_derivative = np.where(preactivation > 0, 1., np.exp(preactivation))
    reference_grad = np.zeros_like(a)
    weights = readout[target].numpy().reshape(4, 4, 64)
    for e, (source, destination) in enumerate(edges.T):
        for h in range(4):
            reference_grad[e, h] = np.sum(weights[destination, h] * elu_derivative[destination, h] * z[source, h])
    np.testing.assert_allclose(scores['alpha_gradient'], reference_grad, atol=1e-11, rtol=1e-9)
    reference_pairs = []
    for u, v in graph['undirected_edges']:
        total = 0.
        for e, (source, destination) in enumerate(edges.T):
            if (source, destination) in ((u, v), (v, u)):
                for h in range(4):
                    total += a[e, h] * reference_grad[e, h]
        reference_pairs.append(total)
    np.testing.assert_allclose(scores['pair_signed'], reference_pairs, atol=1e-11, rtol=1e-9)
    require(not np.allclose(scores['pair_signed'], other_scores['pair_signed']), 'Target class has no effect in reference.')
    def gated(gates):
        with AttentionProbe(stage.conv, graph, gates):
            return stage(packed, True)
    ones = torch.ones(len(reference_pairs), dtype=torch.float64, requires_grad=True)
    gated_output = gated(ones)
    torch.testing.assert_close(gated_output['x'], baseline['x'], atol=0, rtol=0)
    gate_grad = torch.autograd.grad((gated_output['x'] * readout[target]).sum(), ones)[0]
    np.testing.assert_allclose(gate_grad.numpy(), reference_pairs, atol=1e-11, rtol=1e-9)
    numerical = []
    epsilon = 1e-5
    for i in range(len(ones)):
        plus = ones.detach().clone(); minus = plus.clone()
        plus[i] += epsilon; minus[i] -= epsilon
        with torch.no_grad():
            numerical.append(float(((gated(plus)['x'] - gated(minus)['x']) * readout[target]).sum() / (2 * epsilon)))
    np.testing.assert_allclose(numerical, reference_pairs, atol=1e-8, rtol=1e-6)
    zero_one_pair = ones.detach().clone(); zero_one_pair[0] = 0
    masked = gated(zero_one_pair)['attention']['alpha'].detach().numpy()
    mapping = scores['edge_pair_index']
    require(np.all(masked[mapping == 0] == 0), 'Pair mask failed to suppress both directions/all heads.')
    np.testing.assert_array_equal(masked[mapping != 0], a[mapping != 0])
    permutation = np.arange(len(mapping))[::-1]
    shuffled = aggregate_relations(a[permutation], reference_grad[permutation], edges[:, permutation], graph)
    np.testing.assert_allclose(shuffled['pair_signed'], reference_pairs, atol=1e-11, rtol=1e-9)
    invalid = edges.copy(); invalid[:, 0] = [0, 3]  # diagonal-only contact is not an edge
    try:
        edge_pair_indices(invalid, graph)
    except ValueError:
        pass
    else:
        raise ValueError('A non-neighbour relation was accepted.')
    try:
        with AttentionProbe(stage.conv, graph):
            raise RuntimeError('deliberate hook-cleanup test')
    except RuntimeError:
        pass
    after = stage(packed, True)
    torch.testing.assert_close(after['x'], baseline['x'], atol=0, rtol=0)
    require(not after['attention']['alpha'].requires_grad, 'Probe remained installed after an exception.')
    tr.assert_tree_equal(before, tr.cpu_tree(stage.state_dict()))
    checks = {'numpy_message_ELU_gradient_reference': True, 'target_class_sensitivity': True,
              'head_direction_signed_sum': True, 'pair_gate_autograd_equivalence': True,
              'pair_gate_finite_difference': True, 'finite_difference_max_abs_error': float(np.max(np.abs(np.asarray(numerical)-reference_pairs))),
              'both_directions_masked_self_and_other_edges_preserved': True,
              'edge_order_invariance_and_non_neighbour_rejection': True, 'hook_cleanup_and_parameters_unchanged': True}
    if device.type == 'cuda':
        gpu = copy.deepcopy(stage).float().to(device)
        gp = {k: v.float().to(device) if v.is_floating_point() else v.to(device) for k, v in packed.items()}
        with torch.enable_grad(), AttentionProbe(gpu.conv, graph) as probe:
            output = gpu(gp, True)
            grad = torch.autograd.grad((output['x'] * readout[target].float().to(device)).sum(), probe.alpha)[0]
            actual = aggregate_relations(probe.alpha, grad, probe.edges, graph)
        np.testing.assert_allclose(actual['alpha_gradient'], reference_grad, atol=5e-5, rtol=5e-4)
        np.testing.assert_allclose(actual['pair_signed'], reference_pairs, atol=5e-5, rtol=5e-4)
        checks['CPU_CUDA_gradient_agreement'] = 'PASS'
    else:
        checks['CPU_CUDA_gradient_agreement'] = 'NOT_RUN_CPU_ONLY'
    return checks


class Progress(ar.Progress):
    def show(self):
        done = self.done - self.initial
        eta = ev.clock((time.perf_counter() - self.phase_started) / done * (self.total - self.done)) if done else '--'
        if self.total and self.done == self.total:
            eta = '00:00:00'
        print(f'GAT Relations | {self.stage} | Saved {self.done}/{self.total} | '
              f'Elapsed {ev.clock(time.perf_counter() - STARTED)} | Phase ETA {eta}', flush=True)


class OutputLock(ar.OutputLock):
    def __init__(self, output):
        self.path = output / 'gat_relations.lock'


def load_rollout_context(root, variant, project, info, codes):
    folder = root / 'environment/data_audit/attention_rollout_v1' / f'train_preview_{variant}_v1'
    paths = [folder / f'attention_rollout_{part}_v1.json' for part in ('config', 'summary', 'state')]
    config, summary, state = (ev.read_json(path) for path in paths)
    identity = config['identity']; signature = tr.fingerprint(identity)
    require(config['signature'] == summary['signature'] == state['signature'] == signature and
            summary['status'] == 'ATTENTION_ROLLOUT_PREVIEW_COMPLETE' and state['status'] == 'COMPLETE',
            'A completed matching attention_rollout_v1 preview is required.')
    require(identity['variant'] == summary['variant'] == variant and identity['split'] == 'train' and
            identity['checkpoint_sha256'] == info['best_sha256'] == summary['checkpoint_sha256'] and
            identity['data_sha256'] == project['source_sha256'] and identity['versions'] == ev.runtime_versions(),
            'Rollout data, environment or checkpoint differs from this model.')
    def names(values):
        result = {p.replace('\\', '/').rsplit('/', 1)[-1]: h for p, h in values.items()}
        require(len(result) == len(values), 'Ambiguous recorded code basenames.')
        return result
    require(names(identity['code_sha256']) == names(codes), 'Code differs from the completed rollout preview.')
    chosen = ar.choose_train_rows(project)
    selected = [{'id_code': r['id_code'], 'diagnosis': int(r['diagnosis']),
                 'image_sha256': project['processed'][r['id_code']]['output_sha256']} for r in chosen]
    require(identity['selected'] == selected and state['entries'] == summary['results'] and
            len(state['entries']) == 5 and [e['id_code'] for e in state['entries']] == [r['id_code'] for r in chosen],
            'Rollout samples differ from the five fixed training images.')
    require(summary['validation_images_read'] == summary['test_images_read'] == summary['optimizer_steps'] == 0,
            'Unexpected rollout split/optimiser usage.')
    hashes = {str(path): d.sha256(path) for path in paths}
    for row, entry in zip(chosen, state['entries']):
        require(entry['label'] == int(row['diagnosis']), 'Rollout label mismatch.')
        saved = ar.load_saved_evidence(folder, entry)
        require(str(saved['variant']) == variant and saved['labels'].shape == (384, 384), 'Wrong saved segmentation.')
        for name, digest in entry['files'].items():
            hashes[str(folder / name)] = digest
    return folder, chosen, state['entries'], hashes, signature


def save_evidence(output, arrays, record):
    image_id = record['id_code']
    path = output / f'{image_id}_relations_v1.npz'; temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    table = output / f'{image_id}_pairs_v1.csv'; temp = table.with_name(table.name + '.tmp')
    positive_rank = np.zeros(len(arrays['pairs']), dtype=int); negative_rank = positive_rank.copy()
    positive_rank[arrays['positive_order']] = np.arange(1, len(arrays['positive_order']) + 1)
    negative_rank[arrays['negative_order']] = np.arange(1, len(arrays['negative_order']) + 1)
    try:
        with temp.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id_code', 'target_class', 'pair_index', 'region_u', 'region_v', 'shared_border_pixels',
                             'mean_attention', 'signed_sensitivity', 'positive_sensitivity', 'negative_magnitude',
                             'positive_rank_0_if_inapplicable', 'negative_rank_0_if_inapplicable'])
            for i, (u, v) in enumerate(arrays['pairs']):
                writer.writerow([image_id, record['target_class'], i, int(u), int(v), int(arrays['shared_border_lengths'][i]),
                                 float(arrays['pair_mean_attention'][i]), float(arrays['pair_signed'][i]),
                                 float(arrays['pair_positive'][i]), float(arrays['pair_negative_magnitude'][i]),
                                 int(positive_rank[i]), int(negative_rank[i])])
        temp.replace(table)
    finally:
        temp.unlink(missing_ok=True)
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(arrays), 'Saved evidence fields changed.')
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    return {**record, 'evidence_file': path.name, 'files': {p.name: d.sha256(p) for p in (path, table)}}


def load_evidence(output, entry):
    for name, digest in entry['files'].items():
        require(Path(name).name == name and d.sha256(output / name) == digest, 'Relation evidence missing or changed.')
    require(entry['evidence_file'] == entry['id_code'] + '_relations_v1.npz' and
            entry['evidence_file'] in entry['files'], 'Invalid relation evidence filename.')
    with np.load(output / entry['evidence_file'], allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    require(str(values['id_code']) == entry['id_code'] and int(values['label']) == entry['label'] and
            int(values['target_class']) == entry['target_class'], 'Cached relation identity mismatch.')
    return values


def region_anchors(labels, geometry):
    points = []
    for region_id, (cy, cx, _) in enumerate(geometry):
        yy, xx = np.where(labels == region_id)
        distances = (yy - (cy * labels.shape[0] - .5))**2 + (xx - (cx * labels.shape[1] - .5))**2
        index = int(np.argmin(distances))
        points.append((int(xx[index]), int(yy[index])))
    return np.asarray(points)


def shared_border_pixels(labels, u, v):
    border = np.zeros(labels.shape, dtype=bool)
    horizontal = ((labels[:, :-1] == u) & (labels[:, 1:] == v)) | ((labels[:, :-1] == v) & (labels[:, 1:] == u))
    vertical = ((labels[:-1] == u) & (labels[1:] == v)) | ((labels[:-1] == v) & (labels[1:] == u))
    border[:, :-1] |= horizontal; border[:, 1:] |= horizontal
    border[:-1] |= vertical; border[1:] |= vertical
    require(border.any(), 'Attempted to display non-adjacent regions.')
    return border


def save_preview(output, variant, entries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    fig, axes = plt.subplots(len(entries), 3, figsize=(11, 3.7 * len(entries)), squeeze=False)
    for row, entry in enumerate(entries):
        arrays = load_evidence(output, entry)
        rgb = arrays['rgb_uint8'].astype(float) / 255
        labels = arrays['labels']; pairs = arrays['pairs']; scores = arrays['pair_signed']
        anchors = region_anchors(labels, arrays['geometry'])
        axes[row, 0].imshow(rgb, interpolation='nearest')
        axes[row, 0].set_title(f"{entry['id_code']}\nTrue {entry['label']} | predicted / target {entry['target_class']}", fontsize=10)
        for column, sign, name, colour in ((1, 1, 'Positive', np.array([1., .60, .10])),
                                           (2, -1, 'Negative', np.array([.1, .80, 1.]))):
            selected = ranked_indices(scores, sign)[:DISPLAY_PAIRS]
            image = rgb.copy(); region_union = np.zeros(labels.shape, dtype=bool); borders = region_union.copy()
            for i in selected:
                u, v = pairs[i]
                region_union |= (labels == u) | (labels == v)
                borders |= shared_border_pixels(labels, u, v)
            image[region_union] = .78 * image[region_union] + .22 * colour
            image[borders] = colour
            ax = axes[row, column]; ax.imshow(image, interpolation='nearest')
            maximum = float(np.max(sign * scores[selected])) if len(selected) else 0.
            placed_labels = []
            for rank, i in enumerate(selected, 1):
                u, v = pairs[i]; points = anchors[[u, v]]
                strength = sign * scores[i] / maximum
                line, = ax.plot(points[:, 0], points[:, 1], color=colour, linewidth=1.1 + 2 * strength,
                                marker='o', markersize=3, zorder=4)
                line.set_path_effects([patheffects.Stroke(linewidth=2.5 + 2 * strength, foreground='black'),
                                      patheffects.Normal()])
                position = points.mean(axis=0)
                # Adjacent high-ranked pairs can cluster: separate rank labels,
                # while keeping the coloured relationship line at its true location.
                candidates = [position]
                for radius in (18, 28, 40, 54):
                    candidates.extend(position + radius * np.array([np.cos(a), np.sin(a)])
                                      for a in np.linspace(-np.pi / 2, 3 * np.pi / 2, 8, endpoint=False))
                candidates = [np.clip(point, 9, np.array(labels.shape[::-1]) - 10) for point in candidates]
                label_position = next((point for point in candidates if
                                       all(np.linalg.norm(point - used) >= 18 for used in placed_labels)), candidates[-1])
                placed_labels.append(label_position)
                leader = dict(arrowstyle='-', color='white', linewidth=.6) if np.linalg.norm(label_position-position) > 2 else None
                ax.annotate(str(rank), xy=position, xytext=label_position, color='white', ha='center', va='center',
                            fontsize=8, zorder=5, arrowprops=leader,
                            bbox=dict(boxstyle='round,pad=.12', facecolor='black', edgecolor='none', alpha=.8))
            ax.set_title(f'{name} sensitivity | top {len(selected)} pairs\nmax magnitude {maximum:.2e} logit / gate', fontsize=10)
            if not len(selected):
                ax.text(.5, .5, f'No {name.lower()} pairs', transform=ax.transAxes, ha='center', color='white',
                        bbox=dict(facecolor='black', alpha=.7))
        for ax in axes[row]:
            ax.set_axis_off()
    fig.suptitle(f'{variant} | predicted-class GAT relationship sensitivity', fontsize=15, y=.985)
    fig.subplots_adjust(left=.02, right=.98, top=.95, bottom=.06, hspace=.28, wspace=.06)
    fig.text(.5, .036, 'Lines join shared-border superpixel pairs; numbers are ranks within each sign. Self-loops excluded.',
             ha='center', fontsize=9)
    fig.text(.5, .022, 'Positive / negative refers to local target-logit sensitivity, not disease severity or causality.',
             ha='center', fontsize=9)
    fig.text(.5, .008, 'Line strengths are relative within each image/sign. Background pairs are retained. Faithfulness is not evaluated here.',
             ha='center', fontsize=9)
    path = output / 'gat_relations_preview_v1.jpg'; temp = path.with_name(path.stem + '.tmp.jpg')
    try:
        fig.savefig(temp, dpi=140, facecolor='white'); temp.replace(path)
    finally:
        plt.close(fig); temp.unlink(missing_ok=True)
    return path


def run_preview(args, progress):
    root = args.root.resolve(); variant = args.variant
    require(d.sha256(Path(ar.__file__)) == ROLLOUT_CODE_SHA256, 'Use the delivered unchanged attention_rollout_v1.py.')
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA is unavailable.')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    project, info, source_files, codes = ar.locked_context(root, variant, progress)
    source_folder, chosen, previous, rollout_hashes, rollout_signature = load_rollout_context(root, variant, project, info, codes)
    source_files.update(rollout_hashes)
    codes[str(Path(__file__).resolve())] = d.sha256(Path(__file__))
    progress.stage = 'checking class gradients, paired gates and shared-border mapping'
    checks = technical_checks(device)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot
    identity = {'version': VERSION, 'variant': variant, 'checkpoint_sha256': info['best_sha256'],
                'best_epoch': info['summary']['best_epoch'], 'rollout_signature': rollout_signature,
                'selected_ids': [r['id_code'] for r in chosen], 'split': 'train', 'method': METHOD,
                'data_sha256': project['source_sha256'], 'input_file_sha256': source_files,
                'code_sha256': codes, 'versions': ev.runtime_versions(), 'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    signature = tr.fingerprint(identity)
    output = root / 'environment/data_audit/gat_relations_v1' / f'train_preview_{variant}_v1'
    require(args.resume or not output.exists(), f'Output exists. Continue: python gat_relations_v1.py --variant {variant} --resume')
    require(not args.resume or (output / 'gat_relations_config_v1.json').is_file(), 'No relation preview to resume.')
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'gat_relations_state_v1.json'
    with OutputLock(output):
        if args.resume:
            existing = ev.read_json(output / 'gat_relations_config_v1.json')
            require(existing['signature'] == signature and existing['identity'] == identity,
                    'Resume model/data/code/method/environment differs; original output was not overwritten.')
            state = ev.read_json(state_path) if state_path.is_file() else {'signature': signature, 'entries': []}
        else:
            ev.write_json(output / 'gat_relations_config_v1.json', {'identity': identity, 'signature': signature,
                          'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': SOURCES})
            state = {'signature': signature, 'entries': []}
        entries = state['entries']
        require(state['signature'] == signature and len(entries) <= 5 and
                [e['id_code'] for e in entries] == [r['id_code'] for r in chosen[:len(entries)]], 'Saved image order changed.')
        for entry, row, old in zip(entries, chosen, previous):
            require(entry['label'] == int(row['diagnosis']) and entry['target_class'] == old['prediction'], 'Cached class identity changed.')
            require(str(load_evidence(output, entry)['variant']) == variant, 'Cached model variant differs.')
        initial = len(entries); read_count = 0
        state.update(status='RUNNING', error=None); ev.write_json(state_path, state)
        try:
            tr.seed_runtime(42)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            model = before = None
            if initial < 5:
                progress.stage = f'loading trained {variant}, epoch {info["summary"]["best_epoch"]}'
                model, before = ev.load_best_model(info, device)
                dataset = d.AptosDataset(project, 'train', verify_file_hash=True)
                dataset.rows = list(chosen); dataset.augment = None
            progress.reset(f'{variant} fixed training explanations', total=5, done=initial)
            for index in range(initial, 5):
                row = chosen[index]; old = previous[index]
                progress.stage = f'{variant} | {row["id_code"]} | original target grade {old["prediction"]}'
                saved_rollout = ar.load_saved_evidence(source_folder, old)
                sample = dataset[index]; read_count += 1
                require(sample['id_code'] == row['id_code'] and sample['split'] == 'train', 'Wrong image retrieved.')
                rgb = np.rint(ar.as_numpy(sample['rgb']).transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                np.testing.assert_array_equal(rgb, saved_rollout['rgb_uint8'])
                cnn = sample['cnn'].unsqueeze(0).to(device)
                arrays, record = explain_relations(model, cnn, saved_rollout['labels'])
                np.testing.assert_allclose(arrays['logits'], saved_rollout['logits'], atol=5e-4, rtol=5e-4)
                require(record['prediction'] == old['prediction'], 'Prediction differs from the saved rollout.')
                np.testing.assert_allclose(arrays['geometry'], saved_rollout['geometry'], atol=2e-6, rtol=0)
                record.update(id_code=sample['id_code'], label=sample['label'], exact_rollout_maps_reused=True)
                arrays.update(id_code=np.asarray(sample['id_code']), label=np.asarray(sample['label']),
                              variant=np.asarray(variant), rgb_uint8=rgb)
                entry = save_evidence(output, arrays, record)
                entries.append(entry); ev.write_json(state_path, state)
                progress.done = len(entries)
                print(f"{entry['id_code']} | target={entry['target_class']} | pairs={entry['physical_pairs']} | "
                      f"positive={entry['positive_pairs']} negative={entry['negative_pairs']} | gradient / alignment: PASS", flush=True)
            if model is not None:
                tr.assert_tree_equal(before, tr.cpu_tree(model.state_dict()))
                require(all(p.grad is None for p in model.parameters()), 'Unexpected parameter gradients.')
            progress.stage = 'verifying evidence and saving relation preview'
            ev.verify_sources(project, source_files, codes)
            for row in chosen:
                image_id = row['id_code']
                require(d.sha256(root / 'processed_images_384_v1' / (image_id + '.png')) ==
                        project['processed'][image_id]['output_sha256'], 'Selected PNG changed.')
            preview = save_preview(output, variant, entries)
            summary = {'status': 'GAT_RELATIONS_PREVIEW_COMPLETE', 'version': VERSION, 'signature': signature,
                       'variant': variant, 'best_epoch': info['summary']['best_epoch'], 'checkpoint_sha256': info['best_sha256'],
                       'rollout_signature': rollout_signature, 'device': str(device),
                       'train_images': 5, 'train_PNGs_decoded_this_process': read_count, 'resumed_images': initial,
                       'validation_images_read': 0, 'test_images_read': 0, 'optimizer_steps': 0,
                       'exact_rollout_segmentations_reused': True, 'parameters_and_BN_buffers_unchanged': True,
                       'upstream_files_unchanged': True, 'parameter_gradients_accumulated': False,
                       'technical_checks': checks, 'method': METHOD, 'sources': SOURCES, 'results': entries,
                       'preview_file': preview.name, 'preview_sha256': d.sha256(preview), 'visual_review': 'AWAITING_USER_REVIEW',
                       'elapsed_seconds_this_process': time.perf_counter() - STARTED,
                       'CUDA_peak_allocated_MiB': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
                       'limits': 'Five fixed training cases only. Signed scores are local predicted-logit sensitivities; '
                                 'they do not establish lesion localisation, causal contribution, or a finite probability drop. '
                                 'Faithfulness against random pair/region perturbations is a separate stage. '
                                 'No test-driven selection or changes to frozen preprocessing/models.'}
            ev.write_json(output / 'gat_relations_summary_v1.json', summary)
            state.update(status='COMPLETE', error=None); ev.write_json(state_path, state)
        except (Exception, KeyboardInterrupt) as exc:
            state.update(status='STOPPED', error=f'{type(exc).__name__}: {exc}')
            ev.write_json(state_path, state)
            raise
    print(f'\nGAT RELATIONS CHECK: PASS | {variant} | trained best epoch {info["summary"]["best_epoch"]}')
    print('Predicted-class gradients / shared-border pairs / self-loop exclusion / prediction preservation: PASS')
    print('Train images: 5 | Validation/test images read: 0 | Optimizer steps: 0')
    print('Parameters / BN / input files unchanged: PASS | Backward only for explanation gradients')
    print('RELATION PREVIEW: AWAITING VISUAL REVIEW | Faithfulness not evaluated')
    print(f'Total elapsed: {ev.clock(time.perf_counter()-STARTED)}\nSaved: {output}')
    print('Return gat_relations_summary_v1.json and gat_relations_preview_v1.jpg.')
    print('After review passes: remember to commit gat_relations_v1.py.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--variant', choices=('A2', 'A3'), default='A3')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--self-test', action='store_true', help='Only tiny derivative/gate references; no project images.')
    args = parser.parse_args()
    try:
        with Progress() as progress:
            if args.self_test:
                require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable.')
                device = torch.device('cuda' if args.device == 'cuda' or
                                      (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
                torch.set_num_threads(min(4, torch.get_num_threads()))
                checks = technical_checks(device)
                print('RELATION NUMERICAL CHECK: PASS | Project images/checkpoints read: 0')
                print(checks)
            else:
                run_preview(args, progress)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'\nGAT RELATIONS STOPPED | {type(exc).__name__}: {exc}', flush=True)
        print(f'If a preview was started, continue: python gat_relations_v1.py --variant {args.variant} --resume')
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
