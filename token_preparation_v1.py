r"""CP2 Transformer token preparation only (no ViT encoder/classifier yet).

Run in the existing venv: python token_preparation_v1.py
Uses the latest accepted gat_v1 run and its original graph run: the same four
TRAIN images, checked through A0/A1/A2/A3. No new dependency, image decoding,
CNN/GAT rerun or optimizer step. Saved embeddings are untrained check evidence.

Reusable live training API (construct/register TokenPreparation ONCE):
    preparer = TokenPreparation()
    # A0/A1: regional_result from RegionalFeatureExtractor directly.
    prepared = preparer(regional_result)
    # A2/A3: keep original geometry/areas, replace only pooled feature tensors:
    after_gat = {**regional_result,
                 **unpack_graph_features(gat_result['x'], packed_graph['ptr'])}
    prepared = preparer(after_gat)
    # prepared['tokens']: [B, 1+Rmax, 256], CLS first.
    # prepared['key_padding_mask']: True means IGNORE as a key (PyTorch MHA).

No fixed token-slot positional table: superpixel IDs can change between images.
Geometry stays aligned with region rows; it is (normalised y, normalised x,
relative full-canvas area), as already stored by regional_features_v1.
Future attention blocks must pass the mask on EVERY layer. Zero padding alone
is insufficient; masked queries may still produce outputs and must be ignored.
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
    print('Loading token-preparation libraries...', flush=True)
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

VERSION = 'token_preparation_v1'
ABLATIONS = {'A0': ('slic', False), 'A1': ('boundary_slic', False),
             'A2': ('slic', True), 'A3': ('boundary_slic', True)}
SETTINGS = {
    'input_channels': 256, 'embedding_dim': 256,
    'token_projection': 'independent nn.Linear(256,256,bias=True)',
    'geometry_order': ['centroid_y_normalised', 'centroid_x_normalised', 'relative_area'],
    'geometry_definition': '[(mean_pixel_y+0.5)/384, (mean_pixel_x+0.5)/384, area/(384*384)]',
    'geometry_embedding': 'nn.Linear(3,256,bias=False), added to projected region feature',
    'cls_token': 'one learned [1,1,256] vector, prepended, no retinal geometry',
    'embedding_dropout': 0.1, 'normalisation': None, 'extra_activation': None,
    'fixed_slot_position_table': False, 'sequence_order': 'CLS, original regions 0..R-1, padding',
    'key_padding_mask': 'bool [B,1+Rmax]; True=ignore key; CLS always False',
    'token_region_ids': 'CLS=-1; real region=r; padding=-2',
    'padding': 'zero input/geometry before projection, zero output after dropout; no padding gradients',
    'initialisation': 'CPU seed42; Xavier uniform linear weights, zero projection bias; CLS normal std0.02',
    'proposal_requirements': 'learnable projection, centroid/area position, learned class token',
    'implementation_choices': '256-dimensional embedding, linear geometry encoder, its bias=False, '
        'initialisation and dropout0.1 are project choices; not proposal-specified or validation-selected.',
    'ablation': ABLATIONS,
}
SOURCES = {
    'proposal': 'CP1 section 3.7, PDF page50/printed44; geometry and CLS requirements.',
    'vit': 'Dosovitskiy et al. (2021), An Image is Worth 16x16 Words: Transformers '
           'for Image Recognition at Scale, ICLR. https://arxiv.org/abs/2010.11929',
    'vit_scope': 'Section3.1: learnable linear embedding, prepend CLS, add positional representation; '
        'AppendixB.1: embedding dropout. This project uses irregular regional vectors and a '
        'centroid/area encoder instead of original ViT patch flattening and learned slot positions.',
    'regional_tokens': 'Playout et al. (2024), A Region-Based Approach to Diabetic Retinopathy '
        'Classification with Superpixel Tokenization, MICCAI, section2.1. '
        'https://papers.miccai.org/miccai-2024/paper/3756_paper.pdf',
    'regional_scope': 'Supports CNN-to-superpixel regional tokens; not a claim that this paper '
        'used this exact three-coordinate positional encoder, CLS implementation or dimensions.',
    'mask_api': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html',
    'linear_initialisation': 'Glorot and Bengio (2010), Understanding the difficulty of training deep '
                            'feedforward neural networks. https://proceedings.mlr.press/v9/glorot10a.html',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class TokenPreparation(nn.Module):
    """Project region vectors and add aligned geometry, then prepend CLS.

    Expects regional dict: tokens[B,M,256], geometry[B,M,3], region_counts[B],
    key_padding_mask[B,M]. Extra keys such as areas are allowed, never altered.
    Counts and geometry must describe CURRENT augmented-image regions, including
    all canvas regions; padded slots are not anatomical/black-background regions.
    """
    def __init__(self, *, seed=42, embedding_dropout=0.1):
        super().__init__()
        require(0 <= embedding_dropout < 1, 'Dropout must be in [0,1).')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.token_projection = nn.Linear(256, 256, bias=True)
            self.geometry_embedding = nn.Linear(3, 256, bias=False)
            self.cls_token = nn.Parameter(torch.empty(1, 1, 256))
            nn.init.xavier_uniform_(self.token_projection.weight)
            nn.init.zeros_(self.token_projection.bias)
            nn.init.xavier_uniform_(self.geometry_embedding.weight)
            nn.init.normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, regional):
        x = regional['tokens']; geometry = regional['geometry']
        counts = regional['region_counts']; mask = regional['key_padding_mask']
        require(x.ndim == 3 and x.shape[0] > 0 and x.shape[1] > 0 and x.shape[2] == 256
                and x.is_floating_point(), 'Expected floating region tokens [B,M,256].')
        B, M, _ = x.shape
        require(geometry.shape == (B, M, 3) and geometry.is_floating_point(),
                'Expected aligned geometry [B,M,3] in y,x,relative-area order.')
        require(counts.dtype == torch.long and counts.shape == (B,) and
                bool(((counts > 0) & (counts <= M)).all()), 'Invalid region counts.')
        require(mask.dtype == torch.bool and mask.shape == (B, M), 'Invalid padding mask.')
        require(all(t.device == x.device for t in (geometry, counts, mask)), 'Mixed input devices.')
        expected_mask = torch.arange(M, device=x.device)[None, :] >= counts[:, None]
        require(torch.equal(mask, expected_mask), 'Region mask must be valid-prefix then padding.')
        valid_geometry = geometry[~mask]
        require(bool(torch.isfinite(x[~mask]).all() and torch.isfinite(valid_geometry).all()),
                'Non-finite real region features/geometry.')
        require(bool(((valid_geometry > 0) & (valid_geometry <= 1)).all()),
                'Geometry must contain normalised positive y,x,area values <=1.')
        area_sum = geometry[..., 2].masked_fill(mask, 0).sum(1)
        require(torch.allclose(area_sum, torch.ones_like(area_sum), atol=2e-5, rtol=0),
                'Relative areas must sum to one per full-canvas image.')
        # Mask before linear layers: padded NaNs/large values cannot affect outputs or parameter gradients.
        clean_x = x.masked_fill(mask[..., None], 0)
        clean_geometry = geometry.to(x.dtype).masked_fill(mask[..., None], 0)
        region_tokens = self.token_projection(clean_x) + self.geometry_embedding(clean_geometry)
        cls = self.cls_token.to(region_tokens.dtype).expand(B, -1, -1)
        full_mask = torch.cat((torch.zeros(B, 1, dtype=torch.bool, device=x.device), mask), dim=1)
        tokens = self.dropout(torch.cat((cls, region_tokens), dim=1)).masked_fill(full_mask[..., None], 0)
        ids = torch.arange(M, device=x.device).expand(B, -1).masked_fill(mask, -2)
        ids = torch.cat((torch.full((B, 1), -1, dtype=torch.long, device=x.device), ids), dim=1)
        return {'tokens': tokens, 'key_padding_mask': full_mask, 'token_region_ids': ids,
                'region_counts': counts.clone(), 'sequence_lengths': counts+1}


def make_batch(xs, geometries, areas, device):
    """Only the CLI uses cached arrays. Training should pass LIVE tensors directly."""
    xs = [torch.from_numpy(x).to(device) for x in xs]
    counts = torch.tensor([len(x) for x in xs], device=device, dtype=torch.long)
    tokens = pad_sequence(xs, batch_first=True)
    return {'tokens': tokens, 'region_counts': counts,
            'key_padding_mask': torch.arange(tokens.shape[1], device=device)[None, :] >= counts[:, None],
            'geometry': pad_sequence([torch.from_numpy(g).to(device) for g in geometries], batch_first=True),
            'areas': pad_sequence([torch.from_numpy(a).to(device) for a in areas], batch_first=True)}


def technical_check(device):
    state = torch.get_rng_state().clone()
    model = TokenPreparation().double().eval()
    require(torch.equal(state, torch.get_rng_state()), 'Initialisation changed caller CPU RNG.')
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(2, 3, 256, generator=generator, dtype=torch.float64, requires_grad=True)
    geo = torch.tensor([[[.2,.3,.4],[.7,.8,.6],[0,0,0]],
                        [[.1,.2,.2],[.5,.6,.3],[.8,.9,.5]]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False,False,True],[False,False,False]])
    batch = {'tokens': x, 'geometry': geo, 'key_padding_mask': mask, 'region_counts': torch.tensor([2,3])}
    output = model(batch)
    # Independent NumPy calculation detects transposed geometry/order and CLS offset errors.
    W = model.token_projection.weight.detach().numpy(); b = model.token_projection.bias.detach().numpy()
    G = model.geometry_embedding.weight.detach().numpy()
    expected = np.zeros((2,4,256), dtype=np.float64)
    expected[:,0] = model.cls_token.detach().numpy()[0,0]
    for i,n in enumerate((2,3)):
        expected[i,1:n+1] = x.detach().numpy()[i,:n] @ W.T + b + geo.detach().numpy()[i,:n] @ G.T
    np.testing.assert_allclose(output['tokens'].detach().numpy(), expected, atol=1e-11, rtol=1e-10)
    require(output['sequence_lengths'].tolist() == [3,4] and
            output['token_region_ids'].tolist() == [[-1,0,1,-2],[-1,0,1,2]], 'CLS/region IDs shifted.')
    probe = torch.randn(2,4,256, generator=generator, dtype=torch.float64)
    (output['tokens']*probe).sum().backward()
    region_probe = probe[:,1:].masked_fill(mask[...,None],0).numpy()
    np.testing.assert_allclose(model.token_projection.weight.grad.numpy(),
        np.einsum('brd,brc->dc',region_probe,x.detach().numpy()), atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(model.geometry_embedding.weight.grad.numpy(),
        np.einsum('brd,brc->dc',region_probe,geo.detach().numpy()), atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(model.token_projection.bias.grad.numpy(), region_probe.sum((0,1)), atol=1e-10)
    np.testing.assert_allclose(model.cls_token.grad.numpy()[0,0], probe[:,0].numpy().sum(0), atol=1e-10)
    require(bool((x.grad[mask] == 0).all() and (geo.grad[mask] == 0).all()), 'Padding received input gradients.')
    poisoned = {k:v.detach().clone() for k,v in batch.items()}
    poisoned['tokens'][mask] = float('nan'); poisoned['geometry'][mask] = float('nan')
    torch.testing.assert_close(model(poisoned)['tokens'], output['tokens'], atol=0, rtol=0)
    # Reordering valid region rows together with geometry reorders only those tokens.
    perm = torch.tensor([1,0,2]); inverse = torch.argsort(perm)
    swapped = model({**batch, 'tokens': x[:,perm], 'geometry': geo[:,perm]})['tokens']
    torch.testing.assert_close(swapped[:,1:][:,inverse], output['tokens'][:,1:], atol=0, rtol=0)
    changed = geo.detach().clone(); changed[0,0,0] += .1
    moved = model({**batch,'geometry':changed})['tokens']
    torch.testing.assert_close(moved[:,0], output['tokens'][:,0], atol=0, rtol=0)
    require(float((moved[0,1]-output['tokens'][0,1]).abs().max().detach()) > 1e-5,
            'Geometry encoder did not affect the corresponding token.')
    # Temporary MHA probe checks the FUTURE consumer's mask convention, not a ViT implementation.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        attention = nn.MultiheadAttention(256,4,dropout=0,batch_first=True).double().eval()
    values = output['tokens'].detach(); full_mask = output['key_padding_mask']
    normal,_ = attention(values,values,values,key_padding_mask=full_mask,need_weights=False)
    altered = values.clone(); altered[full_mask] = 1e4
    padded,_ = attention(altered,altered,altered,key_padding_mask=full_mask,need_weights=False)
    torch.testing.assert_close(normal[~full_mask], padded[~full_mask], atol=1e-10, rtol=1e-9)
    checks = {'numpy_forward_and_parameter_gradient_reference': True, 'CLS_and_region_ID_alignment': True,
              'geometry_sensitive_and_permutation_equivariant': True, 'padding_input_gradients_zero': True,
              'padded_NaNs_ignored': True, 'temporary_MHA_key_mask_probe': True,
              'initialisation_preserves_CPU_RNG': True}
    if device.type == 'cuda':
        cpu = TokenPreparation().eval(); gpu = copy.deepcopy(cpu).to(device)
        cb = {k:v.detach().float() if v.is_floating_point() else v.clone() for k,v in batch.items()}
        cb['tokens'].requires_grad_(); gb = {k:v.detach().to(device) for k,v in cb.items()}
        gb['tokens'].requires_grad_(); cy=cpu(cb)['tokens']; gy=gpu(gb)['tokens']
        torch.testing.assert_close(gy.cpu(),cy,atol=5e-4,rtol=5e-4)
        cy.square().mean().backward(); gy.square().mean().backward()
        for a,b in [(cb['tokens'].grad,gb['tokens'].grad),*[(a.grad,b.grad) for a,b in zip(cpu.parameters(),gpu.parameters())]]:
            torch.testing.assert_close(b.cpu(),a,atol=5e-4,rtol=5e-4)
        checks['cpu_cuda_forward_and_gradient_agreement'] = True
    else:
        checks['cpu_cuda_forward_and_gradient_agreement'] = 'NOT_RUN_CPU_ONLY'
    return checks


class Progress:
    def __init__(self):
        self.done=0; self.sequences=0; self.stage='starting'
        self.stop=threading.Event(); self.thread=threading.Thread(target=self.refresh,daemon=True)
    def show(self):
        elapsed=time.perf_counter()-STARTED
        eta=f'{elapsed/self.done*(8-self.done):.1f}s' if self.done else '--'
        print(f'Token checks {self.done}/8 | Sequences {self.sequences}/16 | Elapsed {elapsed:.1f}s | '
              f'ETA {eta} | {self.stage}',flush=True)
    def refresh(self):
        while not self.stop.wait(1 if sys.stdout.isatty() else 5): self.show()
    def advance(self,stage): self.done+=1; self.stage=stage
    def __enter__(self): self.show(); self.thread.start(); return self
    def __exit__(self,*args): self.stop.set(); self.thread.join(); self.show()


def resolve_run(root,chosen):
    if chosen is not None:
        path=chosen if chosen.is_absolute() else root/chosen
    else:
        runs=sorted((p for p in (root/'environment/data_audit/gat_v1').glob('run_*') if p.is_dir()),reverse=True)
        require(bool(runs),'No gat_v1 run found; run gat_v1.py first.')
        path=runs[0]
    require((path/'gat_summary_v1.json').is_file(),
            'Latest/selected GAT run is incomplete; use --gat-run to select the accepted run.')
    return path.resolve()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--gat-run',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args(); output=None
    try:
        import dataset_v1 as d
        require(d.VERSION=='input_pipeline_v1','Expected existing dataset_v1.py.')
        require(args.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable.')
        device=torch.device('cuda' if args.device=='cuda' or
                            (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
        torch.set_num_threads(min(4,torch.get_num_threads()))
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        root=args.root.resolve(); source=resolve_run(root,args.gat_run)
        output=root/'environment/data_audit'/VERSION/(
            'run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        output.mkdir(parents=True,exist_ok=False)
        hashes={}; cache={}; results={}
        def read_json(path):
            hashes[str(path)]=d.sha256(path)
            return json.loads(path.read_text(encoding='utf-8'))
        def read_npz(path,expected_hash):
            require(path.is_file() and d.sha256(path)==expected_hash, f'Source file missing or changed: {path}')
            hashes[str(path)]=expected_hash
            with np.load(path,allow_pickle=False) as values:
                return {k:values[k].copy() for k in ('tokens','geometry','areas')}
        with Progress() as progress:
            project=d.load_project(root)
            summary=read_json(source/'gat_summary_v1.json'); config=read_json(source/'gat_config_v1.json')
            require(summary.get('status')=='TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW' and
                    config.get('version')=='gat_v1','Source GAT run did not pass its technical checks.')
            require(config.get('source_metadata_sha256')==project['source_sha256'],
                    'GAT run and current split/preprocessing metadata differ.')
            evidence=summary['results']; ids=list(dict.fromkeys(r['id_code'] for r in evidence))
            methods=('slic','boundary_slic')
            require(len(ids)==4 and len(evidence)==8 and
                    Counter((r['id_code'],r['method']) for r in evidence)==
                    Counter((image_id,m) for image_id in ids for m in methods),
                    'Expected the same four training samples with two methods.')
            train_ids={r['id_code'] for r in project['rows']['train']}
            graph_source=Path(config['upstream_run'])
            require(graph_source.is_dir(),'Original graph run referenced by GAT is missing.')
            graph_summary=read_json(graph_source/'region_adjacency_summary_v1.json')
            graph_config=read_json(graph_source/'region_adjacency_config_v1.json')
            for filename in ('region_adjacency_summary_v1.json','region_adjacency_config_v1.json'):
                path=str(graph_source/filename)
                require(config['source_files_sha256'].get(path)==hashes[path],
                        'Original graph metadata differs from the accepted GAT source.')
            require(graph_config['source_metadata_sha256']==project['source_sha256'],
                    'Original graph run uses different split/preprocessing metadata.')
            graph_rows={(r['id_code'],r['method']):r for r in graph_summary['results']}
            progress.advance('GAT run, original graph run and fixed split metadata matched')
            checks=technical_check(device)
            progress.advance('projection/geometry/CLS, padding gradients and mask-consumer checks passed')
            for row in evidence:
                image_id,method=row['id_code'],row['method']; key=(image_id,method)
                require(image_id in train_ids and row['split']=='train' and row['epoch']==0,
                        'Expected epoch-zero train samples only.')
                graph_row=graph_rows[key]
                raw_name=f'{image_id}_{method}_graph.npz'; gat_name=f'{image_id}_{method}_gat.npz'
                require(row['source_file']==raw_name and row['output_file']==gat_name and
                        graph_row['output_file']==raw_name and graph_row['output_sha256']==row['source_sha256'],
                        'GAT features do not reference the matching original region graph.')
                raw=read_npz(graph_source/raw_name,row['source_sha256'])
                refined=read_npz(source/gat_name,row['output_sha256']); n=row['nodes']
                for item in (raw,refined):
                    require(item['tokens'].shape==(n,256) and item['tokens'].dtype==np.float32 and
                            np.isfinite(item['tokens']).all(),'Unexpected source features.')
                    require(item['geometry'].shape==(n,3) and item['geometry'].dtype==np.float32 and
                            item['areas'].shape==(n,) and item['areas'].dtype==np.int64,
                            'Invalid source region metadata.')
                require(np.array_equal(raw['geometry'],refined['geometry']) and
                        np.array_equal(raw['areas'],refined['areas']) and
                        int(raw['areas'].sum())==384*384 and np.all(raw['areas']>0),
                        'Raw and GAT region identities/areas are not aligned.')
                np.testing.assert_allclose(raw['geometry'][:,2],raw['areas']/(384*384),atol=1e-7,rtol=0)
                cache[key]={'raw':raw,'gat':refined,'raw_file':raw_name,'gat_file':gat_name}
            progress.advance('raw and GAT features verified for the same four train images')
            for variant,(method,use_gat) in ABLATIONS.items():
                items=[cache[(image_id,method)]['gat' if use_gat else 'raw'] for image_id in ids]
                batch=make_batch([x['tokens'] for x in items],[x['geometry'] for x in items],
                                 [x['areas'] for x in items],device)
                snapshots={k:v.clone() for k,v in batch.items()}
                # Independent instances with identical initial settings, not tied trained models.
                model=TokenPreparation().to(device).eval()
                initial={k:v.detach().clone() for k,v in model.state_dict().items()}
                with torch.no_grad():
                    prepared=model(batch); replay=model(batch)
                    torch.testing.assert_close(prepared['tokens'],replay['tokens'],atol=2e-5,rtol=2e-5)
                    require(torch.equal(prepared['key_padding_mask'][:,1:],batch['key_padding_mask']) and
                            not bool(prepared['key_padding_mask'][:,0].any()), 'CLS mask extension failed.')
                    require(bool((prepared['tokens'][prepared['key_padding_mask']]==0).all()), 'Nonzero output padding.')
                    for i,n in enumerate(batch['region_counts'].tolist()):
                        one={k:v[i:i+1,:n] if k in ('tokens','geometry','areas','key_padding_mask')
                             else v[i:i+1] for k,v in batch.items()}
                        torch.testing.assert_close(model(one)['tokens'][0],prepared['tokens'][i,:n+1],atol=5e-5,rtol=5e-5)
                        require(prepared['token_region_ids'][i,:n+1].tolist()==[-1]+list(range(n)),
                                'Region-to-token correspondence differs from source IDs.')
                        print(f'{ids[i]} | {variant} | regions={n} | tokens_with_CLS={[n+1,256]} | alignment=PASS',flush=True)
                        progress.sequences+=1
                    require(bool(torch.isfinite(prepared['tokens']).all()),'Non-finite prepared tokens.')
                    exported={k:v.cpu().numpy() for k,v in prepared.items()}
                    exported.update(id_codes=np.asarray(ids),geometry=batch['geometry'].cpu().numpy(),
                                    areas=batch['areas'].cpu().numpy())
                    target=output/f'{variant}_transformer_tokens_v1.npz'
                    np.savez_compressed(target,**exported)
                    with np.load(target,allow_pickle=False) as saved:
                        require(all(np.array_equal(saved[k],v) for k,v in exported.items()),'Saved token batch roundtrip failed.')
                model.train(); batch['tokens'].requires_grad_()
                live=model(batch)
                live['tokens'].square().mean().backward()
                require(batch['tokens'].grad is not None and bool(torch.isfinite(batch['tokens'].grad).all()) and
                        float(batch['tokens'].grad.abs().sum())>0,'No finite nonzero upstream feature gradient.')
                require(bool((batch['tokens'].grad[batch['key_padding_mask']]==0).all()),'Padding feature gradient is nonzero.')
                for name,parameter in model.named_parameters():
                    require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and
                            float(parameter.grad.abs().sum())>0,f'Missing/non-finite/zero gradient for {name}.')
                require(all(torch.equal(v,snapshots[k]) for k,v in batch.items()),'Input tensors changed.')
                require(all(torch.equal(v,initial[k]) for k,v in model.state_dict().items()),'Parameters changed without optimizer.')
                model.zero_grad(set_to_none=True)
                results[variant]={'method':method,'source_features':'GAT output' if use_gat else 'pooled CNN regions',
                    'image_ids':ids,'counts':batch['region_counts'].tolist(),
                    'input_shape':list(batch['tokens'].shape),'output_shape':list(prepared['tokens'].shape),
                    'sequence_lengths':prepared['sequence_lengths'].tolist(),
                    'masked_padding_slots':int(prepared['key_padding_mask'].sum()),
                    'trainable_parameters':sum(p.numel() for p in model.parameters()),
                    'output_file':target.name,'output_sha256':d.sha256(target),
                    'backward_mode':'train, embedding dropout 0.1; probe loss only','optimizer_steps':0}
                progress.advance(f'{variant}: eval replay, aligned CLS/mask, backward and saved batch passed')
            require(all(d.sha256(Path(p))==h for p,h in hashes.items()),'Upstream files changed during check.')
            checks.update(A0_A1_use_raw_and_A2_A3_use_GAT_features=True,all_four_variants=True,
                eval_replay_within_tolerance=True,batched_vs_individual_eval=True,
                train_mode_backward_all_parameters=True,all_four_NPZ_roundtrips=True,
                input_tensors_and_parameters_unchanged=True,upstream_files_unchanged=True)
            d.write_json(output/'token_preparation_config_v1.json',{
                'version':VERSION,'settings':SETTINGS,'sources':SOURCES,'seed':42,'device':str(device),
                'source_files_sha256':hashes,'source_metadata_sha256':project['source_sha256'],
                'upstream_gat_run':str(source),'upstream_graph_run':str(graph_source),'upstream_gat_config':config,
                'code_sha256':{Path(__file__).name:d.sha256(Path(__file__)),Path(d.__file__).name:d.sha256(Path(d.__file__))},
                'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}})
            d.write_json(output/'token_preparation_summary_v1.json',{
                'status':'TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW','checks':checks,'results':results,
                'train_samples_reused':4,'ablation_sequences':16,'image_files_read':0,'validation_test_samples_used':0,
                'cnn_executed':False,'gat_executed':False,'token_preparation_implemented':True,
                'vit_encoder_implemented':False,'classifier_implemented':False,'optimizer_steps':0,
                'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                'elapsed_seconds':time.perf_counter()-STARTED,
                'limitation':'Four cached training samples through A0-A3. All new projection/geometry/CLS parameters '
                    'are untrained. No full ViT or grading performance test. Temporary MHA only checks mask semantics. '
                    'Later joint training must use live features and current aligned geometry; cached NPZs are evidence only.'})
            progress.advance('four batch files, configuration, method sources and summary saved')
        print('\nTOKEN PREPARATION CHECK: PASS (small-batch technical checks only)')
        print(f'Device: {device} | Linear token projection: 256 -> 256 | Geometry: 3 -> 256 | CLS: prepended')
        print('Projection/geometry/CLS reference: PASS | Region alignment: PASS | Padding mask: PASS')
        print('CPU/CUDA agreement: '+('PASS' if device.type=='cuda' else 'NOT RUN (CPU only)'))
        for variant,stats in results.items():
            print(f'{variant}: counts={stats["counts"]} | sequence={stats["output_shape"]} | '
                  f'padding={stats["masked_padding_slots"]} | source={stats["source_features"]}')
        print('Train-mode backward: PASS | Padding gradients: PASS | '
              f'Trainable parameters: {results["A0"]["trainable_parameters"]} | Optimizer steps: 0')
        print('Reused train samples: 4 | A0-A3 sequences: 16 | Image files read: 0 | Validation/test samples used: 0')
        print('Upstream unchanged: PASS | CNN/GAT rerun: No | Token preparation: initialised, untrained')
        print('ViT encoder: not implemented | Classifier: not implemented')
        print(f'Total elapsed: {time.perf_counter()-STARTED:.1f}s\nSaved: {output}')
        print('Return this summary. No image preview is required.')
        print('After review passes: remember to commit token_preparation_v1.py.')
        return 0
    except Exception as exc:
        print(f'\nTOKEN PREPARATION CHECK: FAIL | {type(exc).__name__}: {exc}',flush=True)
        if output is not None:
            (output/'token_preparation_failure_v1.json').write_text(json.dumps(
                {'status':'FAIL','error':f'{type(exc).__name__}: {exc}'},indent=2)+'\n',encoding='utf-8')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
