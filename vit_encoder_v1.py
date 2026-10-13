r"""CP2 four-block regional ViT encoder; classifier/training are later stages.

Run in the existing venv: python vit_encoder_v1.py
Reads the latest token_preparation_v1 run (four TRAIN images, A0-A3), with
no image decoding or CNN/GAT/token-preparation rerun. No new dependency.
Cached outputs are untrained technical evidence, never training features.

Reusable LIVE API (construct/register the module once):
    encoder = RegionalViTEncoder()
    encoded = encoder(preparer(current_regional_features), return_attention=True)
    cls = encoded['cls']  # [B,256], for a later five-class classifier
    attention = encoded['attentions']  # tuple of four [B,4,L,L] tensors

Attention axes: batch, head, query, key. Returned probabilities are BEFORE
attention dropout; valid rows sum to one, padded rows/columns are zero.
They are the original autograd-connected tensors used by the forward path.
Use eval mode for future Attention Rollout. Rollout is not implemented here.
No extra CLS, geometry encoding, positional table, graph mask or causal mask.
"""
import argparse
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
    print('Loading ViT-encoder libraries...', flush=True)
import numpy as np
import torch
from torch import nn

VERSION = 'vit_encoder_v1'
VARIANTS = ('A0', 'A1', 'A2', 'A3')
SETTINGS = {
    'blocks': 4, 'embedding_dim': 256, 'heads': 4, 'head_dim': 64,
    'mlp_hidden_dim': 1024, 'mlp_activation': 'GELU, exact',
    'normalisation': 'pre-LayerNorm before MSA and MLP; final LayerNorm',
    'layer_norm_eps': 1e-6, 'residuals': 'after MSA and after MLP',
    'attention_dropout': 0.1, 'projection_and_mlp_dropout': 0.1,
    'attention': 'softmax(QK^T/sqrt(64)); global within each image; QKV and output biases',
    'padding': 'True=ignore key in every block; padded query rows and block outputs zero',
    'attention_output': 'tuple of four [B,4,L,L] pre-dropout probabilities, autograd connected',
    'attention_evidence': 'all four images checked; NPZ stores all layers/heads for first image only',
    'initialisation': 'CPU seed42; independent block weights; Xavier uniform Linear weights, '
                      'zero Linear biases, LayerNorm weight1/bias0',
    'proposal_requirement': 'CP1 section3.7: four-block ViT; retain attention for later rollout',
    'implementation_choices': 'D256 matches upstream; heads4, MLP1024, pre-LN, eps1e-6, '
        'dropout0.1 and initialisation are project choices, not proposal-prescribed/tuned values.',
    'classifier': False, 'pretrained_transformer': False, 'attention_rollout': False,
}
SOURCES = {
    'proposal': 'CP1 section3.7, PDF page50/printed44.',
    'vit': 'Dosovitskiy et al. (2021), An Image is Worth 16x16 Words: Transformers for '
           'Image Recognition at Scale, ICLR. https://arxiv.org/abs/2010.11929',
    'vit_scope': 'Section3.1: pre-LN, attention/MLP residual blocks, two-layer GELU MLP, '
        'final normalised CLS. Our 4-block regional model is not the original ViT-Base architecture.',
    'attention': 'Vaswani et al. (2017), Attention Is All You Need, NeurIPS. '
                 'https://arxiv.org/abs/1706.03762',
    'attention_scope': 'Scaled dot-product and multi-head attention; no original decoder used here.',
    'future_rollout': 'Abnar and Zuidema (2020), Quantifying Attention Flow in Transformers, '
        'ACL, pp.4190-4197. https://aclanthology.org/2020.acl-main.385/',
    'rollout_scope': 'Motivates retaining layer attention. This stage neither computes rollout '
        'nor validates attention as a clinical explanation.',
    'initialisation': 'Glorot and Bengio (2010), Understanding the difficulty of training deep '
        'feedforward neural networks. https://proceedings.mlr.press/v9/glorot10a.html',
    'reference_api': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class GlobalAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.qkv = nn.Linear(256, 3*256)
        self.projection = nn.Linear(256, 256)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, 4, 64).permute(2, 0, 3, 1, 4).unbind(0)
        # Compute scores in at least float32, preserving float64 for reference checks.
        with torch.autocast(device_type=x.device.type, enabled=False):
            qk_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            scores = (q.to(qk_dtype) @ k.to(qk_dtype).transpose(-2, -1)) * (64**-0.5)
            scores = scores.masked_fill(mask[:, None, None, :], float('-inf'))
            alpha = scores.softmax(-1).masked_fill(mask[:, None, :, None], 0)
            mixed = self.attention_dropout(alpha) @ v.to(qk_dtype)
        mixed = mixed.to(v.dtype).transpose(1, 2).reshape(B, L, 256)
        return self.output_dropout(self.projection(mixed)), alpha


class EncoderBlock(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(256, eps=1e-6)
        self.attention = GlobalAttention(dropout)
        self.norm2 = nn.LayerNorm(256, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(1024, 256), nn.Dropout(dropout))

    def forward(self, x, mask):
        branch, alpha = self.attention(self.norm1(x), mask)
        x = (x+branch).masked_fill(mask[..., None], 0)
        x = (x+self.mlp(self.norm2(x))).masked_fill(mask[..., None], 0)
        return x, alpha


class RegionalViTEncoder(nn.Module):
    """Consume TokenPreparation output directly; keep live upstream autograd.

    Input fields: tokens[B,L,256], bool key_padding_mask[B,L], region_counts[B],
    sequence_lengths[B], token_region_ids[B,L]. CLS is at position0, followed
    by a valid prefix of region tokens. There must be at least one region.
    """
    def __init__(self, *, seed=42, dropout=0.1):
        super().__init__()
        require(0 <= dropout < 1, 'Dropout must be in [0,1).')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.blocks = nn.ModuleList([EncoderBlock(dropout) for _ in range(4)])
            self.norm = nn.LayerNorm(256, eps=1e-6)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, prepared, return_attention=False):
        x = prepared['tokens']; mask = prepared['key_padding_mask']
        counts = prepared['region_counts']; ids = prepared['token_region_ids']
        lengths = prepared['sequence_lengths']
        require(x.ndim == 3 and x.shape[0] > 0 and x.shape[1] >= 2 and
                x.shape[2] == 256 and x.is_floating_point(), 'Expected tokens [B,L,256], L>=2.')
        B, L, _ = x.shape
        require(mask.dtype == torch.bool and mask.shape == (B, L), 'Invalid key padding mask.')
        require(counts.dtype == torch.long and counts.shape == (B,) and
                bool(((counts > 0) & (counts < L)).all()), 'Invalid region counts.')
        require(lengths.dtype == torch.long and lengths.shape == (B,), 'Invalid sequence lengths.')
        require(ids.dtype == torch.long and ids.shape == (B, L), 'Invalid region IDs.')
        require(all(t.device == x.device for t in (mask, counts, lengths, ids)), 'Mixed input devices.')
        expected = torch.arange(L, device=x.device)[None, :] >= (counts+1)[:, None]
        require(torch.equal(mask, expected) and torch.equal(lengths, counts+1),
                'Expected unmasked CLS, valid region prefix, then padding.')
        expected_ids = (torch.arange(L, device=x.device)-1).expand(B, -1).masked_fill(mask, -2)
        require(torch.equal(ids, expected_ids), 'CLS/region IDs are not aligned.')
        require(bool(torch.isfinite(x[~mask]).all()), 'Non-finite valid token.')
        x = x.masked_fill(mask[..., None], 0)  # sanitize even NaNs in padding
        attention = []
        for block in self.blocks:
            x, alpha = block(x, mask)
            if return_attention:
                attention.append(alpha)  # actual tensor used for context, no detach/restacking
        x = self.norm(x).masked_fill(mask[..., None], 0)
        return {'tokens': x, 'cls': x[:, 0], 'attentions': tuple(attention),
                'key_padding_mask': mask.clone(), 'region_counts': counts.clone(),
                'sequence_lengths': lengths.clone(), 'token_region_ids': ids.clone()}


def make_prepared(x, counts):
    """Small reference-check helper; production calls use TokenPreparation."""
    counts = torch.as_tensor(counts, device=x.device, dtype=torch.long)
    mask = torch.arange(x.shape[1], device=x.device)[None, :] >= (counts+1)[:, None]
    ids = (torch.arange(x.shape[1], device=x.device)-1).expand(x.shape[0], -1).masked_fill(mask, -2)
    return dict(tokens=x, key_padding_mask=mask, region_counts=counts,
                sequence_lengths=counts+1, token_region_ids=ids)


def check_attention(output):
    mask = output['key_padding_mask']; error = 0.
    require(len(output['attentions']) == 4, 'Expected four layers of attention.')
    for alpha in output['attentions']:
        B, L = mask.shape
        require(alpha.shape == (B, 4, L, L) and bool(torch.isfinite(alpha).all()), 'Invalid attention shape/value.')
        require(bool((alpha >= 0).all()), 'Negative attention probability.')
        require(bool((alpha.masked_select(mask[:, None, None, :]) == 0).all()) and
                bool((alpha.masked_select(mask[:, None, :, None]) == 0).all()), 'Padding participates in attention.')
        wanted = (~mask)[:, None, :].expand(B, 4, L).to(alpha.dtype)
        error = max(error, float((alpha.sum(-1)-wanted).abs().max().detach()))
    require(error < 5e-5, 'Valid attention rows do not sum to one.')
    require(bool((output['tokens'][mask] == 0).all()), 'Nonzero padded output.')
    return error


def technical_check(device):
    state = torch.get_rng_state().clone()
    model = RegionalViTEncoder(dropout=0).double().eval()
    require(torch.equal(state, torch.get_rng_state()), 'Initialisation altered caller CPU RNG.')
    require(not torch.equal(model.blocks[0].attention.qkv.weight, model.blocks[1].attention.qkv.weight),
            'Different blocks unexpectedly have identical initial weights.')
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(2, 4, 256, generator=generator, dtype=torch.float64, requires_grad=True)
    batch = make_prepared(x, [2,3]); result = model(batch, True)
    check_attention(result)
    # Independent official PyTorch Transformer implementation, copied weights.
    with torch.random.fork_rng(devices=[]):
        refs = nn.ModuleList([nn.TransformerEncoderLayer(256, 4, 1024, dropout=0,
            activation='gelu', layer_norm_eps=1e-6, batch_first=True, norm_first=True).double()
            for _ in range(4)])
    parameter_pairs = []
    for block, ref in zip(model.blocks, refs):
        pairs = [(block.attention.qkv.weight, ref.self_attn.in_proj_weight),
                 (block.attention.qkv.bias, ref.self_attn.in_proj_bias),
                 (block.attention.projection.weight, ref.self_attn.out_proj.weight),
                 (block.attention.projection.bias, ref.self_attn.out_proj.bias)]
        for ours, theirs in ((block.norm1,ref.norm1), (block.norm2,ref.norm2),
                            (block.mlp[0],ref.linear1), (block.mlp[3],ref.linear2)):
            pairs.extend(((ours.weight,theirs.weight), (ours.bias,theirs.bias)))
        parameter_pairs.extend(pairs)
    final_norm = copy.deepcopy(model.norm)
    parameter_pairs.extend(zip(model.norm.parameters(), final_norm.parameters()))
    with torch.no_grad():
        for ours, theirs in parameter_pairs:
            theirs.copy_(ours)
    xr = x.detach().clone().requires_grad_(); mask = batch['key_padding_mask']
    yr = xr.masked_fill(mask[...,None],0)
    for ref in refs:
        yr = ref(yr, src_key_padding_mask=mask).masked_fill(mask[...,None],0)
    yr = final_norm(yr).masked_fill(mask[...,None],0)
    torch.testing.assert_close(result['tokens'], yr, atol=1e-10, rtol=1e-9)
    probe = torch.randn(2,256, generator=generator, dtype=torch.float64)
    # A nonuniform CLS probe avoids a near-constant squared norm after LayerNorm.
    loss = (result['cls']*probe).sum()
    for alpha in result['attentions']:
        alpha.retain_grad()
    loss.backward(); (yr[:,0]*probe).sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, atol=1e-9, rtol=1e-8)
    for ours, theirs in parameter_pairs:
        torch.testing.assert_close(ours.grad, theirs.grad, atol=1e-9, rtol=1e-8)
    require(bool((x.grad[mask] == 0).all()), 'Padding input gradient is nonzero.')
    require(all(a.grad is not None and bool(torch.isfinite(a.grad).all()) and
                float(a.grad.abs().sum()) > 0 for a in result['attentions']), 'Attention gradient disconnected.')
    # A head-wise MHA reference additionally checks query/key axes and scaling.
    z = model.blocks[0].norm1(x.detach().masked_fill(mask[...,None],0))
    _, expected_alpha = refs[0].self_attn(z,z,z,key_padding_mask=mask,
                                        need_weights=True,average_attn_weights=False)
    expected_alpha = expected_alpha.masked_fill(mask[:,None,:,None],0)
    torch.testing.assert_close(result['attentions'][0], expected_alpha, atol=1e-10, rtol=1e-9)
    with torch.no_grad():
        poison = x.detach().clone(); poison[mask] = float('nan')
        torch.testing.assert_close(model(make_prepared(poison,[2,3]))['tokens'], result['tokens'], atol=0, rtol=0)
        extended = torch.cat((poison, torch.full((2,2,256),float('nan'),dtype=x.dtype)),1)
        torch.testing.assert_close(model(make_prepared(extended,[2,3]))['tokens'][:,:4],
                                   result['tokens'], atol=1e-10, rtol=1e-9)
        perm = [0,2,1,3]
        swapped = model(make_prepared(x.detach()[:,perm],[2,3]))['tokens']
        torch.testing.assert_close(swapped[:,perm], result['tokens'], atol=1e-10, rtol=1e-9)
        changed = x.detach().clone(); changed[1,-1,0] += 2
        moved = model(make_prepared(changed,[2,3]))
        torch.testing.assert_close(moved['tokens'][0], result['tokens'][0], atol=0, rtol=0)
        require(float((moved['cls'][1]-result['cls'][1]).abs().max()) > 1e-6, 'CLS lacks global region influence.')
        single = model(make_prepared(x.detach()[:1,:2],[1]),True)
        check_attention(single)
    checks = dict(torch_encoder_forward_and_gradient_reference=True, per_head_MHA_reference=True,
        padding_keys_queries_and_gradients_zero=True, padded_NaNs_and_extra_padding_ignored=True,
        image_isolation=True, region_permutation_equivariance=True, global_region_to_CLS_influence=True,
        returned_attention_gradient_connected=True, one_region_case=True,
        independent_blocks_and_preserved_CPU_RNG=True)
    if device.type == 'cuda':
        cpu = RegionalViTEncoder().eval(); gpu = copy.deepcopy(cpu).to(device)
        cx = x.detach().float().requires_grad_(); gx = cx.detach().to(device).requires_grad_()
        cy = cpu(make_prepared(cx,[2,3]),True); gy = gpu(make_prepared(gx,[2,3]),True)
        torch.testing.assert_close(gy['tokens'].cpu(),cy['tokens'],atol=5e-4,rtol=5e-4)
        for a,b in zip(cy['attentions'],gy['attentions']):
            torch.testing.assert_close(b.cpu(),a,atol=5e-4,rtol=5e-4)
        cp = probe.float(); (cy['cls']*cp).mean().backward(); (gy['cls']*cp.to(device)).mean().backward()
        for a,b in [(cx.grad,gx.grad),*[(a.grad,b.grad) for a,b in zip(cpu.parameters(),gpu.parameters())]]:
            torch.testing.assert_close(b.cpu(),a,atol=5e-4,rtol=5e-4)
        checks['cpu_cuda_forward_attention_gradient_agreement'] = True
    else:
        checks['cpu_cuda_forward_attention_gradient_agreement'] = 'NOT_RUN_CPU_ONLY'
    return checks


class Progress:
    def __init__(self):
        self.done=0; self.sequences=0; self.stage='starting'
        self.stop=threading.Event(); self.thread=threading.Thread(target=self.refresh,daemon=True)
    def show(self):
        elapsed=time.perf_counter()-STARTED
        eta=f'{elapsed/self.done*(8-self.done):.1f}s' if self.done else '--'
        print(f'ViT checks {self.done}/8 | Sequences {self.sequences}/16 | Elapsed {elapsed:.1f}s | '
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
        runs=sorted((p for p in (root/'environment/data_audit/token_preparation_v1').glob('run_*')
                     if p.is_dir()),reverse=True)
        require(bool(runs),'No token-preparation run found; run token_preparation_v1.py first.')
        path=runs[0]
    require((path/'token_preparation_summary_v1.json').is_file(),
            'Latest/selected token run incomplete; use --token-run for the accepted run.')
    return path.resolve()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--token-run',type=Path)
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
        root=args.root.resolve(); source=resolve_run(root,args.token_run)
        output=root/'environment/data_audit'/VERSION/(
            'run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        output.mkdir(parents=True,exist_ok=False)
        hashes={}; cache={}; results={}
        def read_json(path):
            hashes[str(path)]=d.sha256(path)
            return json.loads(path.read_text(encoding='utf-8'))
        with Progress() as progress:
            project=d.load_project(root)
            summary=read_json(source/'token_preparation_summary_v1.json')
            config=read_json(source/'token_preparation_config_v1.json')
            require(summary.get('status')=='TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW' and
                    config.get('version')=='token_preparation_v1','Source token run did not pass technical checks.')
            require(config.get('source_metadata_sha256')==project['source_sha256'],
                    'Token run and current split/preprocessing metadata differ.')
            require(set(summary['results'])==set(VARIANTS),'Expected A0-A3 token batches.')
            ids=summary['results']['A0']['image_ids']
            train_ids={r['id_code'] for r in project['rows']['train']}
            require(len(ids)==4 and len(set(ids))==4 and set(ids)<=train_ids, 'Expected four distinct training IDs.')
            progress.advance('token-preparation run and fixed split metadata matched')
            checks=technical_check(device)
            progress.advance('independent Transformer reference, masking and attention gradients passed')
            for variant in VARIANTS:
                row=summary['results'][variant]
                require(row['image_ids']==ids,'Ablations use different image order.')
                filename=row['output_file']
                require(Path(filename).name==filename,'Expected a local token-batch filename.')
                path=source/filename
                require(path.is_file() and d.sha256(path)==row['output_sha256'], f'Token source changed: {path}')
                hashes[str(path)]=row['output_sha256']
                fields=('tokens','key_padding_mask','token_region_ids','region_counts','sequence_lengths')
                with np.load(path,allow_pickle=False) as saved:
                    item={k:saved[k].copy() for k in fields}
                    require(saved['id_codes'].tolist()==ids,'Saved token IDs differ from summary.')
                require(item['tokens'].dtype==np.float32 and item['tokens'].shape==tuple(row['output_shape']),
                        'Invalid saved float32 token batch.')
                require(item['region_counts'].tolist()==row['counts'] and
                        item['sequence_lengths'].tolist()==row['sequence_lengths'],'Saved counts differ.')
                cache[variant]={k:torch.from_numpy(v).to(device) for k,v in item.items()}
            for a,b in (('A0','A2'),('A1','A3')):
                for key in ('region_counts','key_padding_mask','token_region_ids'):
                    require(torch.equal(cache[a][key],cache[b][key]),'GAT ablation region identities differ.')
            progress.advance('all four cached token batches verified and loaded')
            for variant in VARIANTS:
                batch=cache[variant]; snapshots={k:v.clone() for k,v in batch.items()}
                model=RegionalViTEncoder().to(device).eval()
                initial={k:v.detach().clone() for k,v in model.state_dict().items()}
                with torch.no_grad():
                    encoded=model(batch,True); replay=model(batch,False)
                    torch.testing.assert_close(encoded['tokens'],replay['tokens'],atol=2e-5,rtol=2e-5)
                    error=check_attention(encoded)
                    require(bool(torch.isfinite(encoded['tokens']).all()),'Non-finite encoder output.')
                    for i,n in enumerate(batch['region_counts'].tolist()):
                        one={k:v[i:i+1,:n+1] if k in ('tokens','key_padding_mask','token_region_ids')
                             else v[i:i+1] for k,v in batch.items()}
                        torch.testing.assert_close(model(one)['tokens'][0], encoded['tokens'][i,:n+1],
                                                   atol=1e-4,rtol=1e-4)
                        print(f'{ids[i]} | {variant} | valid_tokens={n+1} | CLS=[256] | global_attention=PASS',flush=True)
                        progress.sequences+=1
                    exported={k:v.cpu().numpy() for k,v in encoded.items() if k!='attentions'}
                    exported.update(id_codes=np.asarray(ids),
                        attention_first_image=np.stack([a[0].cpu().numpy() for a in encoded['attentions']]),
                        attention_image_id=np.asarray(ids[0]))
                    target=output/f'{variant}_vit_features_v1.npz'
                    np.savez_compressed(target,**exported)
                    with np.load(target,allow_pickle=False) as saved:
                        require(all(np.array_equal(saved[k],v) for k,v in exported.items()),'Saved feature roundtrip failed.')
                # Probe checks differentiability only, not optimisation or a grading loss.
                model.train(); batch['tokens'].requires_grad_()
                live=model(batch,True)
                generator=torch.Generator().manual_seed(107)
                probe=torch.randn(4,256,generator=generator).to(device)
                for alpha in live['attentions']: alpha.retain_grad()
                (live['cls']*probe).mean().backward()
                require(batch['tokens'].grad is not None and bool(torch.isfinite(batch['tokens'].grad).all()) and
                        float(batch['tokens'].grad.abs().sum())>0,'Missing upstream token gradient.')
                require(bool((batch['tokens'].grad[batch['key_padding_mask']]==0).all()),'Nonzero padding gradient.')
                for name,parameter in model.named_parameters():
                    require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and
                            float(parameter.grad.abs().sum())>0,f'Missing/non-finite/zero gradient: {name}')
                require(all(a.grad is not None and bool(torch.isfinite(a.grad).all()) and
                            float(a.grad.abs().sum())>0 for a in live['attentions']), 'Detached attention output.')
                require(all(torch.equal(v,snapshots[k]) for k,v in batch.items()),'Input tensors changed.')
                require(all(torch.equal(v,initial[k]) for k,v in model.state_dict().items()),'Parameters changed.')
                results[variant]={'image_ids':ids,'counts':batch['region_counts'].tolist(),
                    'input_shape':list(batch['tokens'].shape),'output_shape':list(encoded['tokens'].shape),
                    'cls_shape':list(encoded['cls'].shape),
                    'attention_shapes':[list(a.shape) for a in encoded['attentions']],
                    'attention_row_sum_max_error':error,'attention_evidence_image_id':ids[0],
                    'masked_padding_slots':int(batch['key_padding_mask'].sum()),
                    'trainable_parameters':sum(p.numel() for p in model.parameters()),
                    'output_file':target.name,'output_sha256':d.sha256(target),
                    'backward_mode':'train, dropout0.1, four sequences, nonuniform CLS probe','optimizer_steps':0}
                model.zero_grad(set_to_none=True)
                del live, encoded, replay, model, initial, exported
                progress.advance(f'{variant}: global attention, CLS, train backward and saved evidence passed')
            require(all(d.sha256(Path(p))==h for p,h in hashes.items()),'Upstream files changed.')
            checks.update(all_four_variants=True,eval_replay_and_attention_flag_invariance=True,
                batched_vs_individual_eval=True,train_mode_all_parameter_and_attention_gradients=True,
                input_tensors_parameters_upstream_files_unchanged=True,all_four_NPZ_roundtrips=True)
            d.write_json(output/'vit_encoder_config_v1.json',{
                'version':VERSION,'settings':SETTINGS,'sources':SOURCES,'seed':42,'device':str(device),
                'source_files_sha256':hashes,'source_metadata_sha256':project['source_sha256'],
                'upstream_token_run':str(source),'upstream_token_config':config,
                'code_sha256':{Path(__file__).name:d.sha256(Path(__file__)),Path(d.__file__).name:d.sha256(Path(d.__file__))},
                'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}})
            d.write_json(output/'vit_encoder_summary_v1.json',{
                'status':'TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW','checks':checks,'results':results,
                'train_samples_reused':4,'ablation_sequences':16,'image_files_read':0,'validation_test_samples_used':0,
                'cnn_executed':False,'gat_executed':False,'token_preparation_executed':False,
                'vit_encoder_implemented':True,'classifier_implemented':False,'attention_rollout_implemented':False,
                'optimizer_steps':0,
                'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                'elapsed_seconds':time.perf_counter()-STARTED,
                'limitation':'Four cached TRAIN samples through A0-A3, randomly initialised encoder; no grading '
                    'performance or clinical explanation. Attention evidence is pre-dropout in eval mode. '
                    'Joint training must use live CNN/GAT/preparation outputs, never these cached untrained features.'})
            progress.advance('four feature batches, attention evidence, method sources and summary saved')
        print('\nVIT ENCODER CHECK: PASS (small-batch technical checks only)')
        print(f'Device: {device} | 4 blocks | 4 heads x 64 | D=256 | MLP=1024 | pre-LN')
        print('Transformer forward/gradient reference: PASS | Global attention: PASS | Padding mask: PASS')
        print('CPU/CUDA agreement: '+('PASS' if device.type=='cuda' else 'NOT RUN (CPU only)'))
        for variant,stats in results.items():
            print(f'{variant}: sequence={stats["output_shape"]} | CLS={stats["cls_shape"]} | '
                  f'attention/layer={stats["attention_shapes"][0]} | padding={stats["masked_padding_slots"]}')
        print('Train-mode backward: PASS | Attention gradients: PASS | Padding gradients: PASS')
        print(f'Trainable parameters: {results["A0"]["trainable_parameters"]} | Optimizer steps: 0')
        print('Reused train samples: 4 | A0-A3 sequences: 16 | Image files read: 0 | Validation/test samples used: 0')
        print('Upstream unchanged: PASS | CNN/GAT/token preparation rerun: No | ViT: initialised, untrained')
        print('Classifier: not implemented | Attention Rollout: not implemented')
        print(f'Total elapsed: {time.perf_counter()-STARTED:.1f}s\nSaved: {output}')
        print('Return this summary. No image preview is required.')
        print('After review passes: remember to commit vit_encoder_v1.py.')
        return 0
    except Exception as exc:
        print(f'\nVIT ENCODER CHECK: FAIL | {type(exc).__name__}: {exc}',flush=True)
        if output is not None:
            (output/'vit_encoder_failure_v1.json').write_text(json.dumps(
                {'status':'FAIL','error':f'{type(exc).__name__}: {exc}'},indent=2)+'\n',encoding='utf-8')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
