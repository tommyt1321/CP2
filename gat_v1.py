r"""CP2 one-layer, four-head Graph Attention Network (GAT).

Install once in the existing venv: python -m pip install torch-geometric==2.8.0.post1
Run: python gat_v1.py
Reads the latest region_adjacency_v1 run (4 TRAIN images, two methods).
No RGB decoding, CNN rerun, optimiser update, validation or test-image use.

Reusable API for LIVE features during later end-to-end training:
    stage = LocalGraphStage('A3')  # register ONCE in the full nn.Module
    packed = pack_graph_batch(regional_result, current_graphs)
    refined = stage(packed, return_attention=True)
    padded = unpack_graph_features(refined['x'], packed['ptr'])
    # Keep regional_result geometry, areas and mask in the same region order.

A0/A1 return the original x unchanged and contain no GAT parameters.
A2/A3 use physical neighbours plus one self-connection per node. The physical
graph is never edited. Returned alpha is the actual message coefficient, with
autograd attached; in train mode it is AFTER dropout and need not sum to one.
Use eval mode with gradients enabled for later class-conditioned explanations.
The saved attention here is RANDOM-INITIALISATION evidence, NOT an explanation.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import threading
import time
from uuid import uuid4

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading GAT libraries...', flush=True)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
try:
    import torch_geometric
    from torch_geometric.nn import GATConv
except ModuleNotFoundError as exc:
    if exc.name != 'torch_geometric':
        raise
    raise SystemExit('Missing PyTorch Geometric. In the existing venv run:\n'
                     'python -m pip install torch-geometric==2.8.0.post1') from exc
import region_adjacency_v1 as rag

VERSION = 'gat_v1'
ABLATIONS = {'A0': ('slic', False), 'A1': ('boundary_slic', False),
             'A2': ('slic', True), 'A3': ('boundary_slic', True)}
SETTINGS = {
    'operator': 'torch_geometric.nn.GATConv (original additive GAT)',
    'layers': 1, 'input_channels': 256, 'heads': 4, 'channels_per_head': 64,
    'concat': True, 'output_channels': 256, 'negative_slope': 0.2,
    'attention_dropout': 0.1, 'feature_dropout': 0.0, 'activation': 'ELU(alpha=1)',
    'bias': True, 'residual': False, 'normalisation': None, 'edge_dim': None,
    'self_connections': 'exactly one per node, added internally by GATConv',
    'physical_edges': 'unchanged shared-border directed pairs; source -> target',
    'softmax': 'over incoming neighbours INCLUDING self, separately for each head',
    'edge_weighting': 'no boundary length, geometry, boundary-map value or diagnosis',
    'initialisation': 'PyG Glorot weights/attention vectors; zero bias; CPU seed 42',
    'ablation': ABLATIONS,
    'implementation_choices': '64 channels/head preserves the existing 256-dimensional interface. '
        'Dropout 0.1, ELU, bias and no residual are explicit project choices, '
        'not proposal-specified or experimentally selected hyperparameters.',
    'attention_export': 'actual coefficients used by message passing, after dropout; '
        'eval exports sum to one per target/head; training exports need not',
}
SOURCES = {
    'proposal': 'CP1 sections 3.6.2 (one layer, four heads, one-hop), 3.10.1 '
                '(A0/A1 bypass), and 3.11 (PyTorch Geometric).',
    'paper': 'Velickovic et al. (2018), Graph Attention Networks, ICLR. '
             'https://arxiv.org/abs/1710.10903',
    'paper_scope': 'Section 2.1: additive attention, neighbour+self softmax, '
                   'LeakyReLU slope 0.2, concatenated heads. This project does '
                   'not copy the paper benchmark architectures or dropout rates.',
    'author_explanation': 'https://petar-v.com/GAT/',
    'implementation': 'https://pytorch-geometric.readthedocs.io/en/latest/generated/'
                      'torch_geometric.nn.conv.GATConv.html',
    'installation': 'https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html',
}
require = rag.require


def validate_inputs(x, edge_index, ptr):
    """Validate discrete graph metadata without detaching the feature path."""
    require(x.ndim == 2 and x.shape[0] > 0 and x.shape[1] == 256 and x.is_floating_point(),
            'Expected nonempty floating node features [N,256].')
    require(edge_index.dtype == torch.long and edge_index.ndim == 2 and edge_index.shape[0] == 2,
            'Expected source/target edge_index [2,E] with int64 IDs.')
    require(ptr.dtype == torch.long and ptr.ndim == 1 and len(ptr) >= 2,
            'Expected int64 graph pointers.')
    require(edge_index.device == x.device == ptr.device, 'Mixed graph devices.')
    points = ptr.detach().cpu().tolist()
    require(points[0] == 0 and points[-1] == len(x) and
            all(a < b for a, b in zip(points, points[1:])), 'Invalid graph pointers.')
    edges = edge_index.detach().cpu().numpy()
    require(np.all(edges >= 0) and np.all(edges < len(x)), 'Edge endpoint outside node range.')
    require(np.all(edges[0] != edges[1]), 'Physical graph must not contain self-loops.')
    require(np.unique(edges, axis=1).shape[1] == edges.shape[1], 'Duplicate physical edges.')
    graph_ids = np.repeat(np.arange(len(points)-1), np.diff(points))
    require(np.array_equal(graph_ids[edges[0]], graph_ids[edges[1]]), 'Cross-image graph edge.')
    require(bool(torch.isfinite(x).all()), 'Non-finite node features.')


class LocalGraphStage(nn.Module):
    """One reusable module; construct once, then train jointly with the CNN/ViT.

    Input is the packed dict from region_adjacency_v1. Use the matching
    superpixel method in ABLATIONS[variant][0]. No label/diagnosis enters GAT.
    Return x [N,256], plus optional attention {edge_index [2,E+N], alpha
    [E+N,4], is_self_loop [E+N]}. Alpha is NOT detached, so a downstream
    class logit can later be differentiated with respect to these coefficients.
    """
    def __init__(self, variant='A3', seed=42, attention_dropout=0.1):
        super().__init__()
        require(variant in ABLATIONS, 'Expected ablation A0, A1, A2 or A3.')
        require(0 <= attention_dropout < 1, 'Attention dropout must be in [0,1).')
        self.variant = variant
        self.superpixel_method, self.use_gat = ABLATIONS[variant]
        self.conv = None
        if self.use_gat:
            # Isolate CPU initialisation without reseeding the user's CUDA RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.conv = GATConv(256, 64, heads=4, concat=True, negative_slope=0.2,
                                    dropout=attention_dropout, add_self_loops=True,
                                    edge_dim=None, bias=True, residual=False)

    def forward(self, packed, return_attention=False):
        x = packed['x']
        if not self.use_gat:
            return {'x': x, 'attention': None}  # exact identity; no dropout/ELU
        validate_inputs(x, packed['edge_index'], packed['ptr'])
        if return_attention:
            y, (edges, alpha) = self.conv(x, packed['edge_index'], return_attention_weights=True)
            attention = {'edge_index': edges, 'alpha': alpha, 'is_self_loop': edges[0] == edges[1]}
        else:
            y = self.conv(x, packed['edge_index'])
            attention = None
        return {'x': F.elu(y), 'attention': attention}


def check_attention(result, packed):
    """For EVAL outputs only: exact graph support and target/head sums."""
    x = packed['x']; a = result['attention']; edge = a['edge_index']; alpha = a['alpha']
    diagonal = torch.arange(len(x), device=x.device)
    expected = torch.cat((packed['edge_index'], torch.stack((diagonal, diagonal))), dim=1)
    require(torch.equal(edge, expected), 'Message edges differ from physical edges plus one self-loop/node.')
    require(result['x'].shape == x.shape and alpha.shape == (edge.shape[1], 4), 'GAT output shape mismatch.')
    require(bool(torch.isfinite(result['x']).all() and torch.isfinite(alpha).all() and (alpha >= 0).all()),
            'Non-finite features or invalid attention.')
    sums = alpha.new_zeros((len(x), 4)).index_add_(0, edge[1], alpha)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=2e-5, rtol=2e-5)
    return float((sums-1).abs().max().detach())


def independent_reference(stage, x, physical_edges):
    """Small CPU/float64 reference: explicit incoming-neighbour lists, no PyG softmax/scatter."""
    conv = stage.conv; n = len(x)
    z = F.linear(x, conv.lin.weight).reshape(n, 4, 64)
    incoming = [[] for _ in range(n)]
    for source, target in physical_edges.detach().cpu().T.tolist():
        incoming[target].append(source)
    outputs = []
    for target in range(n):
        sources = incoming[target] + [target]
        zs = z[sources]
        score = F.leaky_relu((zs * conv.att_src).sum(-1) +
                            (z[target] * conv.att_dst).sum(-1), negative_slope=0.2)
        alpha = score.softmax(dim=0)
        outputs.append((alpha[..., None] * zs).sum(0).reshape(256))
    return F.elu(torch.stack(outputs) + conv.bias)


def technical_check(device):
    # Deliberately asymmetric edges detect accidental source/target reversal.
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(5, 256, generator=generator, dtype=torch.float64, requires_grad=True)
    edge = torch.tensor([[0, 1, 2, 2], [1, 0, 1, 3]], dtype=torch.long)
    ptr = torch.tensor([0, 4, 5])  # second graph is an isolated node
    stage = LocalGraphStage('A3').double().eval()
    packed = {'x': x, 'edge_index': edge, 'ptr': ptr}
    result = stage(packed, True); check_attention(result, packed)
    reference = independent_reference(stage, x, edge)
    torch.testing.assert_close(result['x'], reference, atol=1e-11, rtol=1e-10)
    probe = torch.randn(5, 256, generator=generator, dtype=torch.float64)
    variables = (x, *stage.parameters())
    actual_grad = torch.autograd.grad((result['x']*probe).sum(), variables, retain_graph=True)
    reference_grad = torch.autograd.grad((reference*probe).sum(), variables)
    for actual, expected in zip(actual_grad, reference_grad):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    require(bool((result['attention']['alpha'][-1] == 1).all()), 'Isolated-node attention is not one.')
    # This is a connectivity gradient only, not a clinical/class explanation.
    alpha_grad = torch.autograd.grad((result['x']*probe).sum(), result['attention']['alpha'])[0]
    require(bool(torch.isfinite(alpha_grad).all() and alpha_grad.abs().sum() > 0),
            'Returned attention is disconnected from the output gradient.')
    changed = x.detach().clone(); changed[2] += 2
    updated = stage({**packed, 'x': changed})['x']
    torch.testing.assert_close(updated[0], result['x'][0], atol=0, rtol=0)
    torch.testing.assert_close(updated[4], result['x'][4], atol=0, rtol=0)
    require(float((updated[1]-result['x'][1]).abs().max().detach()) > 1e-5,
            'Direct neighbour change had no effect.')
    permutation = torch.tensor([2, 0, 3, 1, 4]); inverse = torch.argsort(permutation)
    permuted = stage({**packed, 'x': x.detach()[permutation], 'edge_index': inverse[edge]})['x']
    torch.testing.assert_close(permuted[inverse], result['x'], atol=1e-11, rtol=1e-10)
    for variant in ('A0', 'A1'):
        bypass = LocalGraphStage(variant).train(); bx = x.detach().clone().requires_grad_()
        identity = bypass({'x': bx}, True)
        require(identity['x'] is bx and identity['attention'] is None and
                sum(p.numel() for p in bypass.parameters()) == 0, 'Ablation bypass is not parameter-free identity.')
        identity['x'].sum().backward()
        require(torch.equal(bx.grad, torch.ones_like(bx)), 'Bypass gradient mismatch.')
    checks = {'independent_forward_reference': True, 'independent_gradient_reference': True,
              'targetwise_softmax_and_self_loops': True, 'one_hop_and_image_isolation': True,
              'isolated_node': True, 'permutation_equivariance': True,
              'attention_autograd_connection': True, 'A0_A1_identity_and_gradients': True}
    if device.type == 'cuda':
        cpu = LocalGraphStage('A3').eval()
        gpu = copy.deepcopy(cpu).to(device)
        cx = x.detach().float().requires_grad_(); gx = cx.detach().to(device).requires_grad_()
        cr = cpu({'x': cx, 'edge_index': edge, 'ptr': ptr}, True)
        gr = gpu({'x': gx, 'edge_index': edge.to(device), 'ptr': ptr.to(device)}, True)
        torch.testing.assert_close(gr['x'].cpu(), cr['x'], atol=5e-4, rtol=5e-4)
        torch.testing.assert_close(gr['attention']['alpha'].cpu(), cr['attention']['alpha'], atol=5e-4, rtol=5e-4)
        cr['x'].square().mean().backward(); gr['x'].square().mean().backward()
        for cg, gg in [(cx.grad, gx.grad), *[(a.grad, b.grad) for a, b in zip(cpu.parameters(), gpu.parameters())]]:
            torch.testing.assert_close(gg.cpu(), cg, atol=5e-4, rtol=5e-4)
        checks['cpu_cuda_forward_attention_and_gradients'] = True
    else:
        checks['cpu_cuda_forward_attention_and_gradients'] = 'NOT_RUN_CPU_ONLY'
    return checks


class Progress:
    def __init__(self):
        self.done = 0; self.graphs = 0; self.stage = 'starting'
        self.stop = threading.Event(); self.thread = threading.Thread(target=self.refresh, daemon=True)
    def show(self):
        elapsed = time.perf_counter()-STARTED
        eta = f'{elapsed/self.done*(14-self.done):.1f}s' if self.done else '--'
        print(f'GAT checks {self.done}/14 | Graphs {self.graphs}/8 | Elapsed {elapsed:.1f}s | '
              f'ETA {eta} | {self.stage}', flush=True)
    def refresh(self):
        while not self.stop.wait(1 if sys.stdout.isatty() else 5):
            self.show()
    def advance(self, stage):
        self.done += 1; self.stage = stage
    def __enter__(self):
        self.show(); self.thread.start(); return self
    def __exit__(self, *args):
        self.stop.set(); self.thread.join(); self.show()


def resolve_run(root, chosen):
    if chosen is not None:
        path = chosen if chosen.is_absolute() else root/chosen
    else:
        runs = sorted((p for p in (root/'environment/data_audit/region_adjacency_v1').glob('run_*')
                       if p.is_dir()), reverse=True)
        require(bool(runs), 'No region_adjacency_v1 run found; run that module first.')
        path = runs[0]
    require((path/'region_adjacency_summary_v1.json').is_file(),
            'Latest/selected graph run is incomplete. Use --graph-run to select the accepted run.')
    return path.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--graph-run', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args(); output = None
    try:
        import dataset_v1 as d
        require(d.VERSION == 'input_pipeline_v1' and rag.VERSION == 'region_adjacency_v1',
                'Expected the existing dataset_v1.py and region_adjacency_v1.py.')
        require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable.')
        device = torch.device('cuda' if args.device == 'cuda' or
                              (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
        torch.set_num_threads(min(4, torch.get_num_threads()))
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        root = args.root.resolve(); source = resolve_run(root, args.graph_run)
        output = root/'environment/data_audit'/VERSION/(
            'run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        output.mkdir(parents=True, exist_ok=False)
        methods = ('slic', 'boundary_slic'); groups = {m: [] for m in methods}; results = []; batches = {}
        with Progress() as progress:
            project = d.load_project(root)
            summary_path = source/'region_adjacency_summary_v1.json'
            config_path = source/'region_adjacency_config_v1.json'
            summary = json.loads(summary_path.read_text()); config = json.loads(config_path.read_text())
            require(summary.get('status') == 'TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW' and
                    config.get('version') == 'region_adjacency_v1', 'Source graph run did not pass.')
            require(config.get('source_metadata_sha256') == project['source_sha256'],
                    'Graph run and current split/preprocessing metadata differ.')
            evidence = summary['results']; ids = list(dict.fromkeys(r['id_code'] for r in evidence))
            require(len(ids) == 4 and len(evidence) == 8 and
                    Counter((r['id_code'], r['method']) for r in evidence) ==
                    Counter((image_id, m) for image_id in ids for m in methods),
                    'Expected four train samples and exactly one graph per method/sample.')
            train_ids = {r['id_code'] for r in project['rows']['train']}
            hashes = {str(summary_path): d.sha256(summary_path), str(config_path): d.sha256(config_path)}
            progress.advance('accepted graph run and fixed split metadata matched')
            checks = technical_check(device)
            progress.advance('independent attention/gradient reference, locality and A0/A1 bypass passed')
            for row in evidence:
                image_id, method = row['id_code'], row['method']
                require(image_id in train_ids and row['split'] == 'train' and row['epoch'] == 0,
                        'Source must be the accepted epoch-zero training sample.')
                filename = f'{image_id}_{method}_graph.npz'; path = source/filename
                require(row['output_file'] == filename and d.sha256(path) == row['output_sha256'],
                        'Graph filename/hash differs from source summary.')
                hashes[str(path)] = row['output_sha256']
                with np.load(path, allow_pickle=False) as saved:
                    data = {k: saved[k].copy() for k in saved.files}
                graph = {k: data[k] for k in ('num_nodes', 'undirected_edges', 'boundary_lengths',
                         'edge_index', 'directed_boundary_lengths', 'degree', 'areas')}
                graph['num_nodes'] = int(graph['num_nodes']); stats = rag.validate_graph(graph)
                require(stats['nodes'] == row['nodes'] and stats['directed_edges'] == row['directed_edges'],
                        'Graph statistics differ from source summary.')
                n = graph['num_nodes']
                require(data['tokens'].shape == (n, 256) and data['tokens'].dtype == np.float32 and
                        data['geometry'].shape == (n, 3) and np.isfinite(data['geometry']).all(),
                        'Regional feature/geometry metadata mismatch.')
                groups[method].append((row, graph, torch.from_numpy(data['tokens']).to(device),
                                       torch.from_numpy(data['geometry']).to(device),
                                       torch.from_numpy(data['areas']).to(device)))
            progress.advance('all eight cached train graph files verified and loaded')
            # Same seed in two separate instances for this interface check.
            # Future A2/A3 models are independently trained, never tied together.
            for method, items in groups.items():
                variant = 'A2' if method == 'slic' else 'A3'
                stage = LocalGraphStage(variant).to(device).eval()
                initial = {k: v.detach().clone() for k, v in stage.state_dict().items()}
                rows, graphs, xs, geometries, areas = zip(*items)
                counts = torch.tensor([len(x) for x in xs], dtype=torch.long, device=device)
                tokens = pad_sequence(xs, batch_first=True).detach().requires_grad_()
                regional = {'tokens': tokens, 'region_counts': counts,
                            'key_padding_mask': torch.arange(tokens.shape[1], device=device)[None, :] >= counts[:, None],
                            'geometry': pad_sequence(geometries, batch_first=True),
                            'areas': pad_sequence(areas, batch_first=True)}
                packed = rag.pack_graph_batch(regional, graphs)
                with torch.no_grad():
                    result = stage(packed, True)
                    softmax_error = check_attention(result, packed)
                    replay = stage(packed, True)
                    torch.testing.assert_close(result['x'], replay['x'], atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(result['attention']['alpha'], replay['attention']['alpha'], atol=2e-5, rtol=2e-5)
                    unpacked = rag.unpack_graph_features(result['x'], packed['ptr'])
                    require(torch.equal(unpacked['key_padding_mask'], regional['key_padding_mask']),
                            'GAT changed region count or padding layout.')
                    for index, (row, graph, x, geometry, area) in enumerate(items):
                        a, b = packed['ptr'][index:index+2].tolist()
                        single = {'x': x, 'edge_index': torch.from_numpy(graph['edge_index']).to(device),
                                  'ptr': torch.tensor([0, len(x)], device=device)}
                        separate = stage(single, True)
                        torch.testing.assert_close(separate['x'], result['x'][a:b], atol=5e-5, rtol=5e-5)
                        attention = result['attention']; select = (attention['edge_index'][1] >= a) & (attention['edge_index'][1] < b)
                        local_edges = attention['edge_index'][:, select]-a
                        local_alpha = attention['alpha'][select]
                        require(torch.equal(local_edges, separate['attention']['edge_index']), 'Batched edge order mismatch.')
                        torch.testing.assert_close(local_alpha, separate['attention']['alpha'], atol=5e-5, rtol=5e-5)
                        target = output/f'{row["id_code"]}_{method}_gat.npz'
                        exported = {'tokens': result['x'][a:b].cpu().numpy(),
                                    'message_edge_index': local_edges.cpu().numpy(),
                                    'attention': local_alpha.cpu().numpy(),
                                    'is_self_loop': attention['is_self_loop'][select].cpu().numpy(),
                                    'geometry': geometry.cpu().numpy(), 'areas': area.cpu().numpy()}
                        np.savez_compressed(target, **exported)
                        with np.load(target, allow_pickle=False) as saved:
                            require(all(np.array_equal(saved[k], v) for k, v in exported.items()), 'GAT save/readback mismatch.')
                        results.append({'id_code': row['id_code'], 'method': method, 'variant': variant,
                            'split': 'train', 'epoch': 0, 'nodes': b-a, 'token_shape': [b-a, 256],
                            'physical_directed_edges': graph['edge_index'].shape[1],
                            'self_connections': b-a, 'attention_shape': list(local_alpha.shape),
                            'source_file': row['output_file'], 'source_sha256': row['output_sha256'],
                            'output_file': target.name, 'output_sha256': d.sha256(target)})
                        print(f'{row["id_code"]} | {method} | tokens={[b-a, 256]} | '
                              f'attention={list(local_alpha.shape)} | neighbour+self=PASS', flush=True)
                        progress.graphs += 1; progress.advance(f'{progress.graphs}/8 graph outputs checked and saved')
                # Train-mode dropout and full packed->GAT->unpacked gradient route.
                # This probe loss tests connectivity; it is NOT classification training.
                stage.train()
                live = stage(packed, True); live['attention']['alpha'].retain_grad()
                restored = rag.unpack_graph_features(live['x'], packed['ptr'])
                restored['tokens'].square().mean().backward()
                require(tokens.grad is not None and bool(torch.isfinite(tokens.grad).all()) and
                        float(tokens.grad.abs().sum()) > 0, 'No finite input-token gradient.')
                require(bool((tokens.grad[regional['key_padding_mask']] == 0).all()), 'Padded slots received gradients.')
                for name, parameter in stage.named_parameters():
                    require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                            f'Missing/non-finite GAT gradient: {name}')
                ag = live['attention']['alpha'].grad
                require(ag is not None and bool(torch.isfinite(ag).all()) and float(ag.abs().sum()) > 0,
                        'Returned live attention has no downstream gradient.')
                require(all(torch.equal(v, initial[k]) for k, v in stage.state_dict().items()),
                        'A parameter changed despite zero optimiser steps.')
                stage.zero_grad(set_to_none=True)
                batches[method] = {'variant': variant, 'nodes': len(packed['x']),
                    'input_shape': list(packed['x'].shape), 'output_shape': list(result['x'].shape),
                    'padded_shape': list(unpacked['tokens'].shape), 'counts': counts.tolist(),
                    'attention_shape': list(result['attention']['alpha'].shape),
                    'eval_softmax_max_abs_error': softmax_error,
                    'trainable_parameters': sum(p.numel() for p in stage.parameters()),
                    'backward_mode': 'train, attention dropout 0.1, batch of four graphs',
                    'optimizer_steps': 0}
                progress.advance(f'{variant} batched replay, training-mode backward and padding gradients passed')
            require(all(d.sha256(Path(p)) == h for p, h in hashes.items()), 'Upstream evidence changed.')
            checks.update(batched_vs_individual_eval=True, eval_replay_within_tolerance=True,
                variable_token_padding=True, train_mode_backward_both_methods=True, padding_gradients_zero=True,
                all_eight_output_roundtrips=True, parameters_unchanged=True, upstream_files_unchanged=True)
            d.write_json(output/'gat_config_v1.json', {
                'version': VERSION, 'settings': SETTINGS, 'sources': SOURCES, 'seed': 42,
                'upstream_run': str(source), 'source_files_sha256': hashes,
                'source_metadata_sha256': project['source_sha256'], 'upstream_graph_config': config,
                'code_sha256': {Path(m.__file__).name: d.sha256(Path(m.__file__)) for m in (d, rag)} |
                              {Path(__file__).name: d.sha256(Path(__file__))},
                'device': str(device), 'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                    'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__}})
            d.write_json(output/'gat_summary_v1.json', {
                'status': 'TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW', 'checks': checks,
                'results': results, 'batches': batches, 'train_samples_reused': 4, 'graphs': 8,
                'image_files_read': 0, 'validation_test_samples_used': 0, 'cnn_executed': False,
                'gat_implemented': True, 'gat_trained': False, 'optimizer_steps': 0,
                'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
                'elapsed_seconds': time.perf_counter()-STARTED,
                'limitation': 'Four cached epoch-zero train samples only. Random untrained projection/GAT features; '
                    'attention is not a class explanation. No grading, end-to-end CNN training or performance test. '
                    'Live current-augmentation features/graphs are required for later joint training. '
                    'CUDA scatter reductions can have small numerical differences; replay is tolerance-based.'})
            progress.advance('configuration, method sources and summary saved')
        print('\nGAT CHECK: PASS (small-batch technical checks only)')
        print(f'Device: {device} | PyG: {torch_geometric.__version__} | 1 layer | 4 heads x 64 = 256 channels')
        print('Independent attention/gradient reference: PASS | One-hop isolation: PASS | A0/A1 bypass: PASS')
        print('CPU/CUDA agreement: ' + ('PASS' if device.type == 'cuda' else 'NOT RUN (CPU only)'))
        for method, stats in batches.items():
            print(f'{method}: {stats}')
        print('Train-mode backward: PASS | Padding gradients: PASS | Optimizer steps: 0')
        print('GAT: initialised, untrained | Attention: technical evidence only, not a clinical explanation')
        print('Reused train samples: 4 | Graphs: 8 | Image files read: 0 | Validation/test samples used: 0')
        print('Upstream files unchanged: PASS | CNN rerun: No')
        print(f'Total elapsed: {time.perf_counter()-STARTED:.1f}s\nSaved: {output}')
        print('Return this summary. No image preview is required.')
        print('After review passes: remember to commit gat_v1.py.')
        return 0
    except Exception as exc:
        print(f'\nGAT CHECK: FAIL | {type(exc).__name__}: {exc}', flush=True)
        if output is not None:
            (output/'gat_failure_v1.json').write_text(json.dumps(
                {'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}, indent=2)+'\n', encoding='utf-8')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
