r"""CP2 five-class head, train-only weighted CE and reusable complete neural model.

Run: python classifier_v1.py
First checks four cached ViT CLS batches (same four TRAIN images, A0-A3).
Then rereads ONE of those training images and runs the LIVE complete model for
each variant, batch=1, with classification-loss backward and zero optimizer steps.
This integration check keeps CNN BatchNorm in eval mode to preserve its buffers;
all neural parameters still require gradients. It is not a training run.

Live API (construct/register once; do not train from the diagnostic NPZ caches):
    model = RegionalDRModel('A3').to(device)
    weights, counts = training_class_weights(project['rows']['train'])
    criterion = WeightedDRCrossEntropy(weights).to(device)
    # label_maps must match CURRENT augmented RGB; use model.superpixel_method.
    output = model(cnn, label_maps, segmentation_method='boundary_slic')
    loss = criterion(output['logits'], labels)  # no softmax before CE
    probabilities, grades = class_probabilities(output['logits'])

Segmentation and edge construction are fixed, not gradient-trained. Inference
uses model.eval() plus no_grad; explanations may require gradients in eval mode.
No optimizer, scheduler, early stopping, performance metrics or Rollout here.
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
    print('Loading classifier libraries...', flush=True)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'classifier_v1'
ABLATIONS = {'A0': ('slic', False), 'A1': ('boundary_slic', False),
             'A2': ('slic', True), 'A3': ('boundary_slic', True)}
SETTINGS = {
    'classifier': 'nn.Linear(256,5,bias=True) on final normalised ViT CLS',
    'class_order': [0,1,2,3,4], 'extra_norm_activation_dropout': None,
    'initialisation': 'CPU seed42, Xavier uniform weight, zero bias; preserves caller CPU RNG',
    'probabilities': 'softmax over five logits; prediction=argmax; no probability calibration yet',
    'loss': 'class-weighted cross-entropy on logits and integer targets, label_smoothing=0',
    'class_weights': 'w[c]=N_train/(5*n_train[c]); full train partition ONLY',
    'reduction': 'sum_i w[y_i]*negative_log_probability_i / sum_i w[y_i]',
    'imbalance_handling': 'class weighting only; no oversampling, focal loss or additional sampling',
    'original_DenseNet_classifier': False,
    'proposal_requirements': 'CP1 sections3.7/3.8: five-class CLS head, softmax, train-only weighted CE, joint neural training',
    'implementation_choices': 'single biased Linear head, Xavier initialisation, inverse-frequency '
        'weight formula and PyTorch weighted mean are project choices; not validation-selected.',
    'live_check': 'one existing train image, all four variants sequentially, batch1, '
        'CNN BatchNorm eval but gradients enabled, other modules train; zero optimizer steps',
}
SOURCES = {
    'proposal': 'CP1 sections3.7/3.8, PDF pages50-51/printed44-45.',
    'vit_head': 'Dosovitskiy et al. (2021), An Image is Worth 16x16 Words, ICLR, section3.1. '
                'https://arxiv.org/abs/2010.11929',
    'vit_scope': 'CLS readout and a single linear head (their fine-tuning setting); our '
        'Transformer/head are newly initialised. Their zero-initialised transfer head is not adopted.',
    'weighted_ce_api': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html',
    'class_weight_rule': 'https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html',
    'weight_scope': 'Balanced inverse-frequency convention; implemented directly without adding sklearn. '
        'Proposal cites Sekar et al. for weighted CE; this is not a claim of reproducing their exact weights.',
    'initialisation': 'Glorot and Bengio (2010), Understanding the difficulty of training deep '
        'feedforward neural networks. https://proceedings.mlr.press/v9/glorot10a.html',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class DRClassificationHead(nn.Module):
    """Final CLS [B,256] -> unnormalised class logits [B,5]."""
    def __init__(self, *, seed=42):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.linear = nn.Linear(256,5,bias=True)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, cls):
        require(cls.ndim==2 and cls.shape[0]>0 and cls.shape[1]==256 and cls.is_floating_point(),
                'Expected final CLS [B,256].')
        require(bool(torch.isfinite(cls).all()), 'Non-finite CLS features.')
        return self.linear(cls)


def validate_logits(logits):
    require(logits.ndim==2 and logits.shape[0]>0 and logits.shape[1]==5 and
            logits.is_floating_point() and bool(torch.isfinite(logits).all()), 'Expected finite logits [B,5].')


def class_probabilities(logits):
    validate_logits(logits)
    probabilities = logits.softmax(dim=-1)
    return probabilities, probabilities.argmax(dim=-1)


def training_class_weights(train_rows):
    """Accept only training manifest rows, never a preview batch/validation/test."""
    require(len(train_rows)>0, 'Empty training rows.')
    counts = torch.zeros(5,dtype=torch.long)
    seen = set()
    for row in train_rows:
        require(row['split']=='train', 'Class weights must use training rows only.')
        require(str(row['diagnosis']) in ('0','1','2','3','4'), 'Invalid DR class.')
        require(row['id_code'] not in seen, 'Repeated training image ID.')
        seen.add(row['id_code']); counts[int(row['diagnosis'])] += 1
    require(bool((counts>0).all()), 'Training partition must contain all five classes.')
    weights = (len(train_rows)/(5*counts.double())).float()
    return weights, counts


class WeightedDRCrossEntropy(nn.Module):
    """CE on raw logits. Mean divides by the sum of target-class weights."""
    def __init__(self, class_weights):
        super().__init__()
        weights = torch.as_tensor(class_weights).detach().clone()
        require(weights.shape==(5,) and weights.is_floating_point() and
                bool(torch.isfinite(weights).all() and (weights>0).all()), 'Invalid class weights.')
        self.register_buffer('class_weights',weights)

    def forward(self, logits, targets):
        validate_logits(logits)
        require(targets.dtype==torch.long and targets.shape==(logits.shape[0],) and
                targets.device==logits.device and bool(((targets>=0)&(targets<5)).all()),
                'Expected aligned int64 labels [B] in 0..4.')
        return F.cross_entropy(logits,targets,weight=self.class_weights,reduction='mean',label_smoothing=0.0)


class RegionalDRModel(nn.Module):
    """All trainable neural stages in one registered model, with live gradients.

    CNN receives the existing normalised branch. Label maps must be generated
    from its aligned current RGB after augmentation. The method argument guards
    against accidentally supplying ordinary SLIC to A1/A3; it cannot itself
    prove how an external label map was generated. Graphs are rebuilt from maps.
    No diagnosis enters the forward pass. Normal model.train() enables CNN BN;
    only the read-only CLI diagnostic explicitly keeps BN buffers unchanged.
    """
    def __init__(self, variant='A3', *, pretrained=True, seed=42):
        super().__init__()
        require(variant in ABLATIONS, 'Expected A0, A1, A2 or A3.')
        from regional_features_v1 import RegionalFeatureExtractor
        from gat_v1 import LocalGraphStage
        from token_preparation_v1 import TokenPreparation
        from vit_encoder_v1 import RegionalViTEncoder
        self.variant = variant
        self.superpixel_method, self.use_gat = ABLATIONS[variant]
        self.regional = RegionalFeatureExtractor(pretrained=pretrained,seed=seed)
        self.graph_stage = LocalGraphStage(variant,seed=seed)
        self.preparer = TokenPreparation(seed=seed)
        self.encoder = RegionalViTEncoder(seed=seed)
        self.classifier = DRClassificationHead(seed=seed)

    def forward(self, cnn, label_maps, *, segmentation_method, return_attention=False):
        require(segmentation_method==self.superpixel_method, 'Superpixel method does not match ablation.')
        regional = self.regional(cnn,label_maps)
        gat_attention = None
        if self.use_gat:
            from region_adjacency_v1 import build_region_graph, pack_graph_batch, unpack_graph_features
            packed = pack_graph_batch(regional,[build_region_graph(labels) for labels in label_maps])
            refined = self.graph_stage(packed,return_attention=return_attention)
            current = {**regional, **unpack_graph_features(refined['x'],packed['ptr'])}
            gat_attention = refined['attention']
        else:
            current = regional  # exact bypass, no graph stage or graph parameters
        encoded = self.encoder(self.preparer(current),return_attention=return_attention)
        return {'logits':self.classifier(encoded['cls']), 'cls':encoded['cls'],
                'region_counts':encoded['region_counts'], 'key_padding_mask':encoded['key_padding_mask'],
                'token_region_ids':encoded['token_region_ids'], 'geometry':regional['geometry'],
                'vit_attentions':encoded['attentions'], 'gat_attention':gat_attention}


def technical_check(device):
    state = torch.get_rng_state().clone()
    head = DRClassificationHead().double().eval()
    require(torch.equal(state,torch.get_rng_state()), 'Head initialisation changed caller CPU RNG.')
    generator = torch.Generator().manual_seed(91)
    x = torch.randn(5,256,generator=generator,dtype=torch.float64,requires_grad=True)
    y = torch.arange(5,dtype=torch.long)
    w = torch.tensor([.4,2.1,.8,4.,2.6],dtype=torch.float64)
    logits = head(x); probs,predictions = class_probabilities(logits)
    reference = x.detach().numpy() @ head.linear.weight.detach().numpy().T + head.linear.bias.detach().numpy()
    shifted = reference-reference.max(axis=1,keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
    pn = np.exp(logp); wn = w.numpy(); yn = y.numpy()
    expected_loss = (-logp[np.arange(5),yn]*wn[yn]).sum()/wn[yn].sum()
    loss = WeightedDRCrossEntropy(w)(logits,y)
    np.testing.assert_allclose(logits.detach().numpy(),reference,atol=1e-11,rtol=1e-10)
    np.testing.assert_allclose(probs.detach().numpy(),pn,atol=1e-11,rtol=1e-10)
    np.testing.assert_allclose(float(loss.detach()),expected_loss,atol=1e-11,rtol=1e-10)
    loss.backward()
    dz = pn.copy(); dz[np.arange(5),yn] -= 1; dz *= (wn[yn]/wn[yn].sum())[:,None]
    np.testing.assert_allclose(head.linear.weight.grad.numpy(),dz.T@x.detach().numpy(),atol=1e-10,rtol=1e-9)
    np.testing.assert_allclose(head.linear.bias.grad.numpy(),dz.sum(axis=0),atol=1e-10,rtol=1e-9)
    np.testing.assert_allclose(x.grad.numpy(),dz@head.linear.weight.detach().numpy(),atol=1e-10,rtol=1e-9)
    require(torch.equal(predictions,logits.argmax(-1)), 'Softmax changed predicted class.')
    extreme = torch.tensor([[10000.,9999.,-10000.,0.,-1.]],dtype=torch.float64,requires_grad=True)
    extreme_loss = WeightedDRCrossEntropy(w)(extreme,torch.tensor([2]))
    extreme_loss.backward()
    require(bool(torch.isfinite(extreme_loss) and torch.isfinite(extreme.grad).all()), 'Unstable extreme-logit CE.')
    common_shift = torch.arange(5,dtype=torch.float64)[:,None]*100
    torch.testing.assert_close(WeightedDRCrossEntropy(w)(logits.detach()+common_shift,y),
                               loss.detach(),atol=1e-10,rtol=1e-9)
    torch.testing.assert_close(WeightedDRCrossEntropy(w*7)(logits.detach(),y),loss.detach(),atol=1e-10,rtol=1e-9)
    # This weighted-mean convention cancels class weights for a one-sample batch.
    torch.testing.assert_close(WeightedDRCrossEntropy(w)(logits[:1].detach(),y[:1]),
                               F.cross_entropy(logits[:1].detach(),y[:1]),atol=1e-10,rtol=1e-9)
    rows=[{'id_code':f'{c}_{i}','split':'train','diagnosis':str(c)} for c in range(5) for i in range(c+1)]
    tw,counts=training_class_weights(rows)
    torch.testing.assert_close(tw.double()*counts,torch.full((5,),3.,dtype=torch.float64),atol=1e-6,rtol=0)
    invalid=copy.deepcopy(rows); invalid[0]['split']='validation'
    try:
        training_class_weights(invalid)
    except ValueError:
        pass
    else:
        raise ValueError('Validation rows were accepted for class weights.')
    checks=dict(numpy_linear_softmax_weighted_CE_and_gradient_reference=True,
        all_five_classes_in_synthetic_reference=True, stable_extreme_logits=True,
        shift_and_weight_scale_invariance=True, batch1_weight_cancellation_documented=True,
        train_only_weight_guard=True, preserved_CPU_RNG=True)
    if device.type=='cuda':
        cpu=DRClassificationHead().eval(); gpu=copy.deepcopy(cpu).to(device)
        cx=x.detach().float().requires_grad_(); gx=cx.detach().to(device).requires_grad_()
        cy=cpu(cx); gy=gpu(gx)
        torch.testing.assert_close(gy.cpu(),cy,atol=5e-4,rtol=5e-4)
        cl=WeightedDRCrossEntropy(w.float())(cy,y)
        gl=WeightedDRCrossEntropy(w.float()).to(device)(gy,y.to(device))
        torch.testing.assert_close(gl.cpu(),cl,atol=5e-4,rtol=5e-4)
        cl.backward(); gl.backward()
        for a,b in [(cx.grad,gx.grad),*[(a.grad,b.grad) for a,b in zip(cpu.parameters(),gpu.parameters())]]:
            torch.testing.assert_close(b.cpu(),a,atol=5e-4,rtol=5e-4)
        checks['head_loss_CPU_CUDA_agreement']=True
    else:
        checks['head_loss_CPU_CUDA_agreement']='NOT_RUN_CPU_ONLY'
    return checks


class Progress:
    def __init__(self):
        self.done=0; self.cached=0; self.live=0; self.stage='starting'
        self.stop=threading.Event(); self.thread=threading.Thread(target=self.refresh,daemon=True)
    def show(self):
        elapsed=time.perf_counter()-STARTED
        eta=f'{elapsed/self.done*(12-self.done):.1f}s' if self.done else '--'
        print(f'Classifier checks {self.done}/12 | Cached {self.cached}/16 | Live {self.live}/4 | '
              f'Elapsed {elapsed:.1f}s | ETA {eta} | {self.stage}',flush=True)
    def refresh(self):
        while not self.stop.wait(1 if sys.stdout.isatty() else 5): self.show()
    def advance(self,stage): self.done+=1; self.stage=stage
    def __enter__(self): self.show(); self.thread.start(); return self
    def __exit__(self,*args): self.stop.set(); self.thread.join(); self.show()


def resolve_run(root,chosen):
    if chosen is not None:
        path=chosen if chosen.is_absolute() else root/chosen
    else:
        runs=sorted((p for p in (root/'environment/data_audit/vit_encoder_v1').glob('run_*') if p.is_dir()),reverse=True)
        require(bool(runs),'No ViT run found; run vit_encoder_v1.py first.')
        path=runs[0]
    require((path/'vit_encoder_summary_v1.json').is_file(),
            'Latest/selected ViT run incomplete; use --vit-run for the accepted run.')
    return path.resolve()


def check_live_models(project,image_id,weights,device,progress,output):
    import dataset_v1 as d
    import boundary_map_v1 as bm
    import superpixels_v1 as sp
    import regional_features_v1 as rf
    dataset=d.AptosDataset(project,'train',verify_file_hash=True)
    index=next(i for i,r in enumerate(dataset.rows) if r['id_code']==image_id)
    sample=dataset[index]  # one RGB file, one aligned deterministic train augmentation
    rgb=sample['rgb']; cnn=sample['cnn'][None].to(device)
    rgb_before=rgb.clone(); cnn_before=cnn.clone()
    target=torch.tensor([sample['label']],dtype=torch.long,device=device)
    criterion=WeightedDRCrossEntropy(weights).to(device)
    progress.stage='live: same train RGB loaded; generating the two superpixel maps'
    with torch.no_grad(): details=bm.boundary_map(rgb,return_details=True)
    maps={}
    for method,gamma in (('slic',0.),('boundary_slic',2.)):
        result=sp.segment_superpixels(details['lab'],details['boundary'][0],boundary_weight=gamma)
        maps[method]=result['labels']
    snapshots={k:v.copy() for k,v in maps.items()}; live={}
    for variant,(method,use_gat) in ABLATIONS.items():
        progress.stage=f'live {variant}: loading ImageNet backbone and running complete loss backward'
        model=RegionalDRModel(variant).to(device).train()
        model.regional.backbone.eval()  # diagnostic only: no BatchNorm buffer changes
        initial=rf.state_sha256(model)
        logits_output=model(cnn,[maps[method]],segmentation_method=method,return_attention=True)
        require(logits_output['logits'].shape==(1,5),'Live logits shape mismatch.')
        vit_attention=logits_output['vit_attentions']
        for alpha in vit_attention: alpha.retain_grad()
        if use_gat: logits_output['gat_attention']['alpha'].retain_grad()
        loss=criterion(logits_output['logits'],target); loss.backward()
        modules={'densenet':model.regional.backbone,'channel_projection':model.regional.projection,
                 'token_preparation':model.preparer,'vit_encoder':model.encoder,'classifier':model.classifier}
        if use_gat: modules['gat']=model.graph_stage
        else:
            require(sum(p.numel() for p in model.graph_stage.parameters())==0 and
                    logits_output['gat_attention'] is None,'No-GAT ablation has an active graph stage.')
        norms={}
        for name,module in modules.items():
            grads=[p.grad for p in module.parameters() if p.requires_grad]
            require(bool(grads) and all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
                    f'{variant}: missing/nonfinite gradient in {name}')
            norms[name]=sum(float(g.detach().abs().sum()) for g in grads)
            require(norms[name]>0,f'{variant}: zero total gradient in {name}')
        require(all(a.grad is not None and bool(torch.isfinite(a.grad).all()) and
                    float(a.grad.abs().sum())>0 for a in vit_attention),'Live ViT attention gradient disconnected.')
        if use_gat:
            ag=logits_output['gat_attention']['alpha'].grad
            require(ag is not None and bool(torch.isfinite(ag).all()) and float(ag.abs().sum())>0,
                    'Live GAT attention gradient disconnected.')
        require(rf.state_sha256(model)==initial,'Live parameters/BatchNorm buffers changed.')
        require(torch.equal(cnn,cnn_before) and torch.equal(rgb,rgb_before),'Live input tensors changed.')
        probabilities,predictions=class_probabilities(logits_output['logits'].detach())
        live[variant]={'image_id':image_id,'label':sample['label'],'method':method,
            'regions':logits_output['region_counts'].tolist(), 'logits_shape':[1,5],
            'gradient_l1_by_module':norms,'vit_attention_gradients':True,'gat_attention_gradients':True if use_gat else 'BYPASS',
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'probe_CE':float(loss.detach()),'logits':logits_output['logits'].detach().cpu().tolist(),
            'probabilities':probabilities.cpu().tolist(),'untrained_prediction':predictions.cpu().tolist(),
            'parameters_and_BN_buffers_unchanged':True,'optimizer_steps':0}
        print(f'LIVE {variant} | {image_id} | regions={live[variant]["regions"][0]} | logits=[1,5] | '
              f'classification loss -> all neural stages: PASS',flush=True)
        model.zero_grad(set_to_none=True)
        del loss, logits_output, vit_attention, model, modules, grads, alpha, module
        if use_gat: del ag
        progress.live+=1; progress.advance(f'{variant}: complete live classification backward passed')
    require(all(np.array_equal(maps[k],v) for k,v in snapshots.items()),'Live label maps changed.')
    np.savez_compressed(output/'live_segmentation_evidence_v1.npz',id_code=np.asarray(image_id),
                        **{k+'_labels':v for k,v in maps.items()})
    return live


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--vit-run',type=Path)
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
        root=args.root.resolve(); source=resolve_run(root,args.vit_run)
        output=root/'environment/data_audit'/VERSION/(
            'run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        output.mkdir(parents=True,exist_ok=False)
        hashes={}; results={}
        def read_json(path):
            hashes[str(path)]=d.sha256(path)
            return json.loads(path.read_text(encoding='utf-8'))
        with Progress() as progress:
            project=d.load_project(root)
            summary=read_json(source/'vit_encoder_summary_v1.json'); config=read_json(source/'vit_encoder_config_v1.json')
            require(summary.get('status')=='TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW' and
                    config.get('version')=='vit_encoder_v1','Source ViT run did not pass technical checks.')
            require(config.get('source_metadata_sha256')==project['source_sha256'],'ViT and current split metadata differ.')
            require(set(summary['results'])==set(ABLATIONS),'Expected four ViT ablations.')
            ids=summary['results']['A0']['image_ids']; train_rows={r['id_code']:r for r in project['rows']['train']}
            require(len(ids)==4 and len(set(ids))==4 and set(ids)<=set(train_rows),'Expected four distinct train IDs.')
            weights,counts=training_class_weights(project['rows']['train'])
            require(counts.tolist()==[1257,236,646,124,190] and int(counts.sum())==2453,'Fixed training counts differ.')
            labels=torch.tensor([int(train_rows[i]['diagnosis']) for i in ids],device=device,dtype=torch.long)
            progress.advance('accepted ViT run matched; class weights derived from all 2453 TRAIN labels')
            checks=technical_check(device)
            progress.advance('independent head/softmax/loss/gradient checks passed')
            for variant in ABLATIONS:
                row=summary['results'][variant]; name=row['output_file']
                require(row['image_ids']==ids and Path(name).name==name,'ViT batch identity differs.')
                path=source/name
                require(path.is_file() and d.sha256(path)==row['output_sha256'],f'ViT source changed: {path}')
                hashes[str(path)]=row['output_sha256']
                with np.load(path,allow_pickle=False) as saved:
                    cls=saved['cls'].copy()
                    require(saved['id_codes'].tolist()==ids and np.array_equal(cls,saved['tokens'][:,0]),'Saved CLS/IDs differ.')
                    require(cls.dtype==np.float32 and cls.shape==(4,256),'Invalid CLS batch.')
                x=torch.from_numpy(cls).to(device); snapshot=x.clone()
                head=DRClassificationHead().to(device).eval()
                initial={k:v.detach().clone() for k,v in head.state_dict().items()}
                criterion=WeightedDRCrossEntropy(weights).to(device)
                with torch.no_grad():
                    logits=head(x); probabilities,predictions=class_probabilities(logits)
                    torch.testing.assert_close(head(x),logits,atol=0,rtol=0)
                    torch.testing.assert_close(probabilities.sum(-1),torch.ones(4,device=device),atol=1e-6,rtol=0)
                    require(torch.equal(predictions,logits.argmax(-1)),'Prediction mapping error.')
                    loss=criterion(logits,labels)
                    exported={'id_codes':np.asarray(ids),'labels':labels.cpu().numpy(),'logits':logits.cpu().numpy(),
                              'probabilities':probabilities.cpu().numpy(),'untrained_predictions':predictions.cpu().numpy()}
                    dest=output/f'{variant}_classifier_outputs_v1.npz'
                    np.savez_compressed(dest,**exported)
                    with np.load(dest,allow_pickle=False) as saved:
                        require(all(np.array_equal(v,saved[k]) for k,v in exported.items()),'Classifier output roundtrip failed.')
                head.train(); x.requires_grad_(); criterion(head(x),labels).backward()
                require(x.grad is not None and bool(torch.isfinite(x.grad).all()) and float(x.grad.abs().sum())>0,
                        'No finite nonzero CLS gradient.')
                for name,param in head.named_parameters():
                    require(param.grad is not None and bool(torch.isfinite(param.grad).all()) and
                            float(param.grad.abs().sum())>0,f'Missing/nonfinite/zero head gradient: {name}')
                require(torch.equal(x,snapshot) and all(torch.equal(v,initial[k]) for k,v in head.state_dict().items()),
                        'Cached check changed features or parameters.')
                results[variant]={'image_ids':ids,'labels':labels.tolist(),'cls_shape':[4,256],
                    'logits_shape':[4,5],'probability_shape':[4,5],'head_parameters':sum(p.numel() for p in head.parameters()),
                    'probe_weighted_CE':float(loss),'output_file':dest.name,'output_sha256':d.sha256(dest),'optimizer_steps':0}
                print(f'{variant} | cached CLS=[4,256] -> logits=[4,5] | weighted CE/backward=PASS',flush=True)
                progress.cached+=4; progress.advance(f'{variant}: cached head/loss outputs and gradients passed')
            # Verify exact reusable modules used by the accepted encoder/current integration.
            import boundary_map_v1 as bm, superpixels_v1 as sp, regional_features_v1 as rf
            import region_adjacency_v1 as rag, gat_v1 as gat, token_preparation_v1 as tp, vit_encoder_v1 as vit
            modules=(d,bm,sp,rf,rag,gat,tp,vit)
            code_hashes={Path(m.__file__).name:d.sha256(Path(m.__file__)) for m in modules}
            require(config['code_sha256'].get('vit_encoder_v1.py')==code_hashes['vit_encoder_v1.py'],
                    'Encoder code differs from the accepted cached run.')
            code_hashes[Path(__file__).name]=d.sha256(Path(__file__))
            image_path=root/'processed_images_384_v1'/(ids[0]+'.png')
            hashes[str(image_path)]=project['processed'][ids[0]]['output_sha256']
            progress.advance('live model ready; one of the same four TRAIN images selected')
            live=check_live_models(project,ids[0],weights,device,progress,output)
            require(all(d.sha256(Path(p))==h for p,h in hashes.items()),'Upstream files/image changed.')
            checks.update(cached_all_four_variants=True, probabilities_sum_to_one=True,
                full_train_counts_only=True, input_features_parameters_and_files_unchanged=True,
                live_all_four_variants_classification_backward=True, no_GAT_has_no_graph_parameters=True,
                live_attention_gradients=True, no_optimizer_steps=True, output_roundtrips=True)
            d.write_json(output/'classifier_config_v1.json',{
                'version':VERSION,'settings':SETTINGS,'sources':SOURCES,'seed':42,'device':str(device),
                'train_class_counts':counts.tolist(),'train_class_weights':weights.tolist(),
                'source_files_sha256':hashes,'source_metadata_sha256':project['source_sha256'],
                'upstream_vit_run':str(source),'upstream_vit_config':config,'code_sha256':code_hashes,
                'component_settings':{m.VERSION:m.SETTINGS for m in modules if hasattr(m,'SETTINGS')},
                'live_augmentation':{'image_id':ids[0],'split':'train','epoch':0,'seed':42},
                'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}})
            d.write_json(output/'classifier_summary_v1.json',{
                'status':'TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW','checks':checks,'results':results,'live_results':live,
                'train_samples_reused':4,'cached_classifications':16,'live_unique_train_images':1,'live_forwards':4,
                'image_files_read':1,'validation_test_samples_used':0,'class_weights_training_labels_used':2453,
                'classifier_implemented':True,'complete_neural_model_implemented':True,
                'attention_rollout_implemented':False,'training_loop_implemented':False,'optimizer_steps':0,
                'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                'elapsed_seconds':time.perf_counter()-STARTED,
                'limitation':'Untrained outputs/losses are interface evidence, not model performance. '
                    'Live backward uses one image with CNN BN in eval mode and all parameters trainable. '
                    'No full train-mode multi-image/AMP/optimizer-memory capacity test. In weighted-mean CE '
                    'a batch of one cancels its class weight; this is only a gradient integration check.'})
            progress.advance('classifier batches, live results, method sources and summary saved')
        print('\nCLASSIFIER CHECK: PASS (small-batch technical checks only)')
        print(f'Device: {device} | Head: Linear 256 -> 5 | Head parameters: 1285')
        print('Class order: [0,1,2,3,4] | CE input: logits | Softmax: probability output')
        print(f'Train-only class counts: {counts.tolist()}')
        print('Train-only class weights: '+str([round(float(w),6) for w in weights]))
        print('Linear/softmax/weighted CE/gradient reference: PASS | Probability sums: PASS')
        print('Head/loss CPU/CUDA agreement: '+('PASS' if device.type=='cuda' else 'NOT RUN (CPU only)'))
        for variant,stats in results.items():
            print(f'{variant}: cached logits={stats["logits_shape"]} | probabilities={stats["probability_shape"]} | '
                  f'live logits=[1,5] | neural parameters={live[variant]["trainable_parameters"]} | backward=PASS')
        print('Complete live CNN -> optional GAT -> ViT -> classifier -> CE backward: PASS (A0-A3, batch=1)')
        print('Live mode: CNN BN eval with gradients enabled; other neural modules train | Optimizer steps: 0')
        print('Train images reused: 4 cached / 1 live subset | Image files read: 1 | Validation/test samples used: 0')
        print('Upstream files/parameters/BN buffers unchanged: PASS | New neural modules: initialised, untrained')
        print('Training loop: not implemented | Attention Rollout: not implemented')
        if device.type=='cuda':
            print(f'CUDA peak allocated: {torch.cuda.max_memory_allocated(device)/2**20:.1f} MiB (this check only)')
        print(f'Total elapsed: {time.perf_counter()-STARTED:.1f}s\nSaved: {output}')
        print('Return this summary. No image preview is required.')
        print('After review passes: remember to commit classifier_v1.py.')
        return 0
    except Exception as exc:
        print(f'\nCLASSIFIER CHECK: FAIL | {type(exc).__name__}: {exc}',flush=True)
        if output is not None:
            (output/'classifier_failure_v1.json').write_text(json.dumps(
                {'status':'FAIL','error':f'{type(exc).__name__}: {exc}'},indent=2)+'\n',encoding='utf-8')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
