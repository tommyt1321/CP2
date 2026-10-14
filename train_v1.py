r"""CP2 joint training loop. DEFAULT: a small TRAIN-only optimizer/checkpoint check.

    python train_v1.py

Smoke mode uses the same four accepted training IDs, A0-A3, two optimizer
updates per variant. It enables CNN BatchNorm training and dropout, verifies
checkpoint restoration, and NEVER reads validation/test images. Smoke weights
are diagnostic and cannot initialise/resume formal experiments.

Formal training is an explicit later command, after a matching smoke PASS:
    python train_v1.py --mode train --variant A0
    python train_v1.py --mode train --variant A0 --resume PATH_TO_LAST_PT

Training uses all 2453 TRAIN images, validates on 527 VALIDATION images, and
never reads TEST images. Fresh formal runs start from ImageNet CNN plus newly
initialised other modules, not smoke checkpoints. Resume is at epoch boundaries.
Existing images/split metadata/upstream code remain unchanged. All outputs go
to a unique environment/data_audit/training_v1/run_* directory.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import threading
import time
from uuid import uuid4

STARTED=time.perf_counter()
if __name__=='__main__': print('Loading training libraries...',flush=True)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import dataset_v1 as d
import boundary_map_v1 as bm
import superpixels_v1 as sp
import regional_features_v1 as rf
import region_adjacency_v1 as rag
import gat_v1 as gat
import token_preparation_v1 as tp
import vit_encoder_v1 as vit
import classifier_v1 as classifier

VERSION='training_v1'
MODULES=(d,bm,sp,rf,rag,gat,tp,vit,classifier)
SETTINGS={
    'seed':42,'batch_size':2,'accumulation_steps':2,'epochs':30,
    'backbone_lr':1e-5,'new_lr':1e-4,'weight_decay':1e-4,
    'adam_betas':[0.9,0.999],'adam_eps':1e-8,'clip_grad_norm':1.0,
    'scheduler':'CosineAnnealingLR, one step after each completed training epoch, eta_min=0, no restarts/warmup',
    'precision':'float32; AMP disabled in v1','patience':7,'min_qwk_delta':1e-4,
    'macro_f1_tie_tolerance':1e-12,
    'decay_policy':'exclude all bias, one-dimensional norm parameters and learned CLS token',
    'loss':'train-only inverse-frequency weighted CE; effective-batch numerator divided by target-weight sum',
    'partial_batch':'keep all images; merge a final singleton effective group into the preceding group',
    'BN':'train mode with actual microbatch size; no frozen backbone',
    'segmentation':'CPU float32 boundary/SLIC after current train augmentation; rebuild every microbatch',
    'evaluation':'fixed five-class QWK and macro-F1; weighted CE numerator/denominator over whole validation split',
    'selection':'strictly highest validation QWK; numerical QWK tie uses higher macro-F1',
    'early_stopping':'7 epochs without QWK gain >1e-4 relative to patience anchor; macro-F1 ties do not reset patience',
    'resume':'epoch boundary; model, optimizer, scheduler, RNG, best/early-stop state and history',
    'proposal':'joint fine-tuning; AdamW+cosine; smaller CNN LR; weighted CE; validation QWK selection',
    'implementation_choices':'Numerical hyperparameters are initial project settings, not claimed optimal or tuned.',
}
SOURCES={
    'proposal':'CP1 section3.8, PDF51/printed45.',
    'adamw':'Loshchilov and Hutter (2019), Decoupled Weight Decay Regularization, ICLR. https://arxiv.org/abs/1711.05101',
    'cosine':'Loshchilov and Hutter (2017), SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR. https://arxiv.org/abs/1608.03983',
    'cosine_scope':'Uses cosine shape only, with AdamW and no warm restarts; not full SGDR.',
    'optimizer_api':'https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html',
    'scheduler_api':'https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html',
    'checkpoint_api':'https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html',
    'weighted_ce':'https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html',
    'qwk':'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html',
    'macro_f1':'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html',
    'scope':'Metrics implemented directly; fixed labels0..4, macro-F1 zero_division0; no new sklearn dependency.',
}


def require(condition,message):
    if not condition: raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def seed_runtime(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # Some CUDA graph scatter reductions may remain nondeterministic; no bitwise claim.


def rng_state():
    ns=np.random.get_state()
    return {'python':random.getstate(),'numpy':[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]],
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); ns=state['numpy']
    np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        require(torch.cuda.is_available() and len(state['torch_cuda'])==torch.cuda.device_count(),
                'Checkpoint CUDA RNG devices differ.')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['torch_cuda']])


def cpu_tree(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a.cpu(),b.cpu(),atol=0,rtol=0)
    elif isinstance(a,dict):
        require(a.keys()==b.keys(),'Checkpoint state keys differ.')
        for k in a: assert_tree_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        require(type(a)==type(b) and len(a)==len(b),'Checkpoint state sequence differs.')
        for x,y in zip(a,b): assert_tree_equal(x,y)
    else: require(a==b,'Checkpoint scalar differs.')


class Progress:
    def __init__(self):
        self.stage='starting'; self.done=0; self.total=0; self.updates=0; self.started=time.perf_counter()
        self.stop=threading.Event(); self.thread=threading.Thread(target=self.refresh,daemon=True)
    def reset(self,stage,total):
        self.stage=stage; self.done=0; self.total=total; self.started=time.perf_counter()
    def show(self):
        elapsed=time.perf_counter()-STARTED; phase=time.perf_counter()-self.started
        eta=f'{phase/self.done*(self.total-self.done):.1f}s' if self.done else '--'
        print(f'Training | {self.stage} | Images {self.done}/{self.total} | Updates {self.updates} | '
              f'Elapsed {elapsed:.1f}s | Phase ETA {eta}',flush=True)
    def refresh(self):
        while not self.stop.wait(1 if sys.stdout.isatty() else 5): self.show()
    def __enter__(self): self.show(); self.thread.start(); return self
    def __exit__(self,*args): self.stop.set(); self.thread.join(); self.show()


def make_optimizer(model,cfg):
    grouped={name:[] for name in ('backbone_decay','backbone_no_decay','new_decay','new_no_decay')}
    for name,p in model.named_parameters():
        require(p.requires_grad,f'Unexpected frozen neural parameter: {name}')
        prefix='backbone' if name.startswith('regional.backbone.') else 'new'
        decay=p.ndim>1 and not name.endswith('.bias') and name!='preparer.cls_token'
        grouped[prefix+('_decay' if decay else '_no_decay')].append(p)
    groups=[]
    for name,params in grouped.items():
        if params:
            groups.append({'params':params,'name':name,
                'lr':cfg['backbone_lr'] if name.startswith('backbone') else cfg['new_lr'],
                'weight_decay':cfg['weight_decay'] if name.endswith('_decay') and not name.endswith('_no_decay') else 0.})
    ids=[id(p) for g in groups for p in g['params']]
    require(len(ids)==len(set(ids))==len(list(model.parameters())),'Optimizer parameter coverage/duplication error.')
    optimizer=torch.optim.AdamW(groups,betas=tuple(cfg['adam_betas']),eps=cfg['adam_eps'],foreach=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=0.)
    return optimizer,scheduler


def index_groups(n,effective_batch,seed,epoch,shuffle=True):
    require(n>=2 and effective_batch>=2,'Training needs at least two samples/effective batch.')
    indices=torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch)).tolist() if shuffle else list(range(n))
    groups=[indices[i:i+effective_batch] for i in range(0,n,effective_batch)]
    if len(groups)>1 and len(groups[-1])==1: groups[-2].extend(groups.pop())
    return groups


def prepare_microbatch(dataset,indices,method,device):
    samples=[dataset[i] for i in indices]
    cnn=torch.stack([s['cnn'] for s in samples]).to(device)
    labels=torch.tensor([s['label'] for s in samples],dtype=torch.long,device=device)
    maps=[]
    # All geometry comes from the exact same augmented RGB as each CNN tensor.
    with torch.no_grad():
        for sample in samples:
            details=bm.boundary_map(sample['rgb'],return_details=True)
            maps.append(sp.segment_superpixels(details['lab'],details['boundary'][0],
                boundary_weight=2. if method=='boundary_slic' else 0.)['labels'])
    return {'cnn':cnn,'labels':labels,'maps':maps,'ids':[s['id_code'] for s in samples]}


def train_group(model,optimizer,dataset,indices,weights,cfg,device,progress=None,audit=False):
    require(dataset.split=='train' and model.training,'Optimizer updates require training data/model mode.')
    require(all(m.training for m in model.modules() if isinstance(m,nn.BatchNorm2d)), 'CNN BN must train here.')
    targets=torch.tensor([int(dataset.rows[i]['diagnosis']) for i in indices],device=device)
    denominator=weights[targets].sum()
    require(bool(torch.isfinite(denominator) and denominator>0),'Invalid effective-batch weight denominator.')
    optimizer.zero_grad(set_to_none=True); numerator_total=0.; last=None
    for start in range(0,len(indices),cfg['batch_size']):
        chosen=indices[start:start+cfg['batch_size']]
        batch=prepare_microbatch(dataset,chosen,model.superpixel_method,device)
        logits=model(batch['cnn'],batch['maps'],segmentation_method=model.superpixel_method)['logits']
        numerator=F.cross_entropy(logits,batch['labels'],weight=weights,reduction='sum')
        require(bool(torch.isfinite(numerator)),'Nonfinite classification loss; no optimizer step performed.')
        (numerator/denominator).backward()
        numerator_total+=float(numerator.detach()); last=batch
        if progress: progress.done+=len(chosen)
        del logits,numerator
    audit_norms={}
    if audit:
        for name,module in (('backbone',model.regional.backbone),('projection',model.regional.projection),
                            ('preparer',model.preparer),('encoder',model.encoder),('classifier',model.classifier),
                            ('gat',model.graph_stage)):
            params=list(module.parameters())
            if not params: continue
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),f'Invalid {name} gradients.')
            audit_norms[name]=sum(float(p.grad.detach().abs().sum()) for p in params)
            require(audit_norms[name]>0,f'Zero {name} gradient.')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'],error_if_nonfinite=True)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    if progress: progress.updates+=1
    return {'numerator':numerator_total,'denominator':float(denominator),
            'loss':numerator_total/float(denominator),'gradient_norm_before_clip':float(norm),
            'gradient_l1_by_module':audit_norms,'last_batch':last}


def metrics_from_confusion(confusion):
    cm=np.asarray(confusion,dtype=np.float64)
    require(cm.shape==(5,5) and np.isfinite(cm).all() and (cm>=0).all() and cm.sum()>0,'Invalid confusion matrix.')
    n=float(cm.sum()); true=cm.sum(1); pred=cm.sum(0)
    distance=(np.arange(5)[:,None]-np.arange(5)[None,:])**2/16.
    expected=np.outer(true,pred)/n; denom=float((distance*expected).sum())
    qwk=1-float((distance*cm).sum())/denom if denom>0 else None
    f1=np.divide(2*np.diag(cm),true+pred,out=np.zeros(5),where=(true+pred)>0)
    return {'qwk':qwk,'macro_f1':float(f1.mean()),'accuracy':float(np.trace(cm)/n),
            'per_class_f1':f1.tolist(),'confusion_matrix':cm.astype(np.int64).tolist(),'samples':int(n)}


def evaluate(model,dataset,weights,batch_size,device,progress=None):
    require(dataset.split=='validation','Formal evaluation uses validation only; test is not supported.')
    model.eval(); cm=np.zeros((5,5),dtype=np.int64); numerator=denominator=0.; ids=[]; targets=[]; predictions=[]
    with torch.no_grad():
        for start in range(0,len(dataset),batch_size):
            indices=list(range(start,min(start+batch_size,len(dataset))))
            batch=prepare_microbatch(dataset,indices,model.superpixel_method,device)
            logits=model(batch['cnn'],batch['maps'],segmentation_method=model.superpixel_method)['logits']
            num=F.cross_entropy(logits,batch['labels'],weight=weights,reduction='sum')
            require(bool(torch.isfinite(num)),'Nonfinite validation loss.')
            numerator+=float(num); denominator+=float(weights[batch['labels']].sum())
            y=batch['labels'].cpu().numpy(); p=logits.argmax(-1).cpu().numpy()
            np.add.at(cm,(y,p),1); ids.extend(batch['ids']); targets.extend(y.tolist()); predictions.extend(p.tolist())
            if progress: progress.done+=len(indices)
    result=metrics_from_confusion(cm); result['weighted_loss']=numerator/denominator
    return result,{'id_codes':np.asarray(ids),'labels':np.asarray(targets,dtype=np.int64),
                   'predictions':np.asarray(predictions,dtype=np.int64)}


def is_better(metrics,best,cfg):
    qwk=metrics['qwk']
    if qwk is None or not math.isfinite(qwk): return False
    if best is None: return True
    difference=qwk-best['qwk']
    if difference>cfg['macro_f1_tie_tolerance']: return True
    return abs(difference)<=cfg['macro_f1_tie_tolerance'] and metrics['macro_f1']>best['macro_f1']


def atomic_checkpoint(path,payload):
    temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(payload,temporary); temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def checkpoint_payload(model,optimizer,scheduler,signature,kind,epoch,global_step,state):
    return {'version':VERSION,'kind':kind,'signature':signature,'variant':model.variant,
            'epoch_completed':epoch,'global_step':global_step,'model':cpu_tree(model.state_dict()),
            'optimizer':cpu_tree(optimizer.state_dict()),'scheduler':cpu_tree(scheduler.state_dict()),
            'rng':rng_state(),'selection_state':copy.deepcopy(state)}


def load_checkpoint(path,model,optimizer,scheduler,signature,kind,device):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    require(payload.get('version')==VERSION and payload.get('kind')==kind and
            payload.get('signature')==signature and payload.get('variant')==model.variant,
            'Checkpoint kind/variant/settings/data/code/environment differs; smoke cannot resume formal training.')
    model.load_state_dict(payload['model'],strict=True)
    # Construct the scheduler before loading optimizer state so saved LRs survive.
    scheduler.load_state_dict(payload['scheduler']); optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'])
    return payload


def technical_checks(cfg):
    # Uneven microbatches, all five classes, independent whole-batch weighted CE.
    gen=torch.Generator().manual_seed(71)
    values=torch.randn(5,5,generator=gen,dtype=torch.float64)
    x=values.clone().requires_grad_(); z=values.clone().requires_grad_()
    y=torch.arange(5); weights=torch.tensor([.4,2.1,.8,4.,2.6],dtype=torch.float64)
    whole=F.cross_entropy(x,y,weight=weights); whole.backward()
    total=0.
    for a,b in ((0,2),(2,4),(4,5)):
        loss=F.cross_entropy(z[a:b],y[a:b],weight=weights,reduction='sum')/weights[y].sum()
        total+=float(loss.detach()); loss.backward()
    torch.testing.assert_close(z.grad,x.grad,atol=1e-12,rtol=1e-11)
    require(abs(total-float(whole.detach()))<1e-12,'Effective-batch loss differs from weighted CE.')
    groups=index_groups(2453,4,42,0)
    require(sorted(i for g in groups for i in g)==list(range(2453)) and len(groups[-1])==5,
            'Final singleton merge dropped/duplicated training data.')
    require(index_groups(17,4,42,0)==index_groups(17,4,42,0) and
            index_groups(17,4,42,0)!=index_groups(17,4,42,1),'Epoch sampling is not reproducible/varying.')
    perfect=metrics_from_confusion(np.eye(5,dtype=np.int64))
    require(perfect['qwk']==1 and perfect['macro_f1']==1,'Perfect metric reference failed.')
    reverse=metrics_from_confusion(np.fliplr(np.eye(5,dtype=np.int64)))
    require(abs(reverse['qwk']+1)<1e-12 and abs(reverse['macro_f1']-.2)<1e-12,'Reversed metric reference failed.')
    constant=np.zeros((5,5),dtype=np.int64); constant[:,0]=1
    result=metrics_from_confusion(constant)
    require(abs(result['qwk'])<1e-12 and abs(result['macro_f1']-1/15)<1e-12,'Constant metric reference failed.')
    degenerate=np.zeros((5,5),dtype=np.int64); degenerate[0,0]=1
    require(metrics_from_confusion(degenerate)['qwk'] is None,'Undefined kappa was silently assigned a value.')
    require(not is_better({'qwk':.69,'macro_f1':.9},{'qwk':.7,'macro_f1':.2},cfg) and
            is_better({'qwk':.7,'macro_f1':.3},{'qwk':.7,'macro_f1':.2},cfg),'QWK selection priority broken.')
    return {'effective_batch_weighted_loss_and_gradients':True,'partial_batch_coverage':True,
            'epoch_sampling':True,'five_class_metric_references':True,'QWK_selection_priority':True}


def resolve_run(root,chosen,folder,summary_name):
    if chosen is not None:
        path=chosen if chosen.is_absolute() else root/chosen
    else:
        runs=sorted((p for p in (root/'environment/data_audit'/folder).glob('run_*') if p.is_dir()),reverse=True)
        if summary_name=='smoke_summary_v1.json':
            runs=[p for p in runs if (p/'training_config_v1.json').is_file() and
                  json.loads((p/'training_config_v1.json').read_text(encoding='utf-8')).get('mode')=='smoke']
        require(bool(runs),f'No {folder} run found.')
        path=runs[0]
    require((path/summary_name).is_file(),f'Latest/selected {folder} run incomplete; select its accepted run explicitly.')
    return path.resolve()


def load_context(root,chosen,cfg,device):
    project=d.load_project(root)
    source=resolve_run(root,chosen,'classifier_v1','classifier_summary_v1.json')
    summary_path=source/'classifier_summary_v1.json'; config_path=source/'classifier_config_v1.json'
    source_summary=json.loads(summary_path.read_text(encoding='utf-8'))
    source_config=json.loads(config_path.read_text(encoding='utf-8'))
    require(source_summary.get('status')=='TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW' and
            source_config.get('version')=='classifier_v1','Classifier source must have passed.')
    require(source_config['source_metadata_sha256']==project['source_sha256'],'Split/preprocessing metadata changed.')
    codes={Path(m.__file__).name:d.sha256(Path(m.__file__)) for m in MODULES}
    require(all(codes.get(k)==v for k,v in source_config['code_sha256'].items()),
            'Neural component code differs from accepted classifier run.')
    codes[Path(__file__).name]=d.sha256(Path(__file__))
    weights,counts=classifier.training_class_weights(project['rows']['train'])
    require(counts.tolist()==source_config['train_class_counts'],'Training label counts changed.')
    ids=source_summary['results']['A0']['image_ids']
    train_ids={r['id_code'] for r in project['rows']['train']}
    require(len(ids)==4 and len(set(ids))==4 and set(ids)<=train_ids,'Expected four accepted training IDs.')
    import torchvision,torch_geometric
    versions={'python':platform.python_version(),'torch':str(torch.__version__),
              'torchvision':str(torchvision.__version__),'pyg':str(torch_geometric.__version__),'numpy':np.__version__}
    core={'settings':cfg,'data_sha256':project['source_sha256'],'code_sha256':codes,
          'versions':versions,'device_type':device.type}
    provenance={'upstream_classifier_run':str(source),
                'source_files_sha256':{str(summary_path):d.sha256(summary_path),str(config_path):d.sha256(config_path)},
                'train_class_counts':counts.tolist(),'train_class_weights':weights.tolist(),
                'core':core,'core_fingerprint':fingerprint(core),'sources':SOURCES,
                'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}
    return project,weights.to(device),ids,provenance


def verify_readonly(project,provenance,image_ids):
    for name,hash_value in project['source_sha256'].items():
        require(d.sha256(project['root']/name)==hash_value,'Split/preprocessing metadata changed during run.')
    for name,hash_value in provenance['source_files_sha256'].items():
        require(d.sha256(Path(name))==hash_value,'Upstream classifier evidence changed.')
    for image_id in image_ids:
        require(d.sha256(project['root']/'processed_images_384_v1'/(image_id+'.png'))==
                project['processed'][image_id]['output_sha256'],'Original processed image changed.')


def smoke_check(project,weights,ids,provenance,cfg,device,output,progress):
    checks=technical_checks(cfg); results={}
    lookup={r['id_code']:r for r in project['rows']['train']}
    for variant in classifier.ABLATIONS:
        seed_runtime(cfg['seed'])
        dataset=d.AptosDataset(project,'train',seed=cfg['seed'],verify_file_hash=True)
        dataset.rows=[lookup[i] for i in ids]  # local dataset view only; project/split CSV unchanged
        progress.reset(f'SMOKE {variant}: loading trainable model',4)
        model=classifier.RegionalDRModel(variant,seed=cfg['seed']).to(device)
        optimizer,scheduler=make_optimizer(model,cfg)
        signature=fingerprint({'core':provenance['core_fingerprint'],'variant':variant})
        steps=[]; checkpoint=output/f'{variant}_smoke.pt'
        for epoch in range(2):
            progress.reset(f'SMOKE {variant} update {epoch+1}/2',4)
            dataset.set_epoch(epoch); model.train()
            before=cpu_tree(model.state_dict())
            indices=index_groups(4,4,cfg['seed'],epoch)[0]
            stats=train_group(model,optimizer,dataset,indices,weights,cfg,device,progress,audit=True)
            after=cpu_tree(model.state_dict()); deltas={}
            for prefix in ('regional.backbone.','regional.projection.','preparer.','encoder.','classifier.','graph_stage.'):
                names=[name for name,_ in model.named_parameters() if name.startswith(prefix)]
                if not names: continue
                deltas[prefix]=sum(float((after[n]-before[n]).abs().sum()) for n in names)
                require(deltas[prefix]>0,f'{variant}: {prefix} parameters did not update.')
            expected_microbatches=math.ceil(4/cfg['batch_size'])
            bn_names=[n for n in before if n.endswith('num_batches_tracked')]
            require(bool(bn_names) and all(int(after[n]-before[n])==expected_microbatches for n in bn_names),
                    'CNN BatchNorm counters did not update for each microbatch.')
            require(all(bool(torch.isfinite(v).all()) for v in after.values() if v.is_floating_point()),
                    'Nonfinite model state after update.')
            require(all(int(state['step'])==epoch+1 for state in optimizer.state.values()),'AdamW step state incorrect.')
            lr_used=[g['lr'] for g in optimizer.param_groups]; scheduler.step()
            for g in optimizer.param_groups:
                base=cfg['backbone_lr'] if g['name'].startswith('backbone') else cfg['new_lr']
                expected=base*(1+math.cos(math.pi*(epoch+1)/cfg['epochs']))/2
                require(math.isclose(g['lr'],expected,rel_tol=1e-10,abs_tol=1e-15),'Cosine LR differs from reference.')
            probe=stats.pop('last_batch')
            model.eval()
            with torch.no_grad():
                reference=model(probe['cnn'],probe['maps'],segmentation_method=model.superpixel_method)['logits'].cpu()
            payload=checkpoint_payload(model,optimizer,scheduler,signature,'smoke',epoch+1,epoch+1,{})
            atomic_checkpoint(checkpoint,payload)
            steps.append({**stats,'epoch_index':epoch,'optimizer_step':epoch+1,
                          'parameter_l1_changes':deltas,'BN_batches_added':expected_microbatches,'lr_used':lr_used,
                          'lr_next_epoch':[g['lr'] for g in optimizer.param_groups]})
            if epoch==0:
                # Continue the SECOND actual update from a fresh model/optimizer loaded from disk.
                del model,optimizer,scheduler,before,after
                model=classifier.RegionalDRModel(variant,pretrained=False,seed=cfg['seed']).to(device)
                optimizer,scheduler=make_optimizer(model,cfg)
                loaded=load_checkpoint(checkpoint,model,optimizer,scheduler,signature,'smoke',device)
                assert_tree_equal(model.state_dict(),payload['model'])
                assert_tree_equal(optimizer.state_dict(),payload['optimizer'])
                assert_tree_equal(scheduler.state_dict(),payload['scheduler'])
                assert_tree_equal(rng_state(),payload['rng'])
                model.eval()
                with torch.no_grad():
                    replay=model(probe['cnn'],probe['maps'],segmentation_method=model.superpixel_method)['logits'].cpu()
                torch.testing.assert_close(replay,reference,atol=5e-5,rtol=5e-5)
                restore_rng(loaded['rng'])
                del loaded,replay
            print(f'{variant} | optimizer step={epoch+1}/2 | all active modules updated=PASS | '
                  f'BN train=PASS | loss={stats["loss"]:.5f} (diagnostic)',flush=True)
            del probe,reference,payload
        results[variant]={'train_image_ids':ids,'optimizer_steps':2,'steps':steps,
            'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'checkpoint_file':checkpoint.name,'checkpoint_sha256':d.sha256(checkpoint),
            'checkpoint_restored_before_second_step':True,'classifier_output_shape':[cfg['batch_size'],5]}
        del model,optimizer,scheduler,dataset,before,after
    verify_readonly(project,provenance,ids)
    checks.update(all_four_variants_actual_updates=True, all_active_modules_updated=True,
                  BN_train_and_running_counters=True, AdamW_groups_and_step_state=True,
                  cosine_reference=True, model_optimizer_scheduler_RNG_checkpoint_roundtrip=True,
                  resumed_second_update=True, input_files_unchanged=True)
    summary={'status':'TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW','version':VERSION,'mode':'smoke',
        'core_fingerprint':provenance['core_fingerprint'],'checks':checks,'results':results,
        'unique_train_images':4,'training_image_exposures':32,'validation_images_read':0,'test_images_read':0,
        'optimizer_steps_total':8,'optimizer_steps_per_variant':2,'formal_training_started':False,
        'microbatch_size':cfg['batch_size'],'accumulation_steps':cfg['accumulation_steps'],'effective_batch_size':4,
        'precision':'float32','CUDA_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
        'CUDA_peak_reserved_mib':torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None,
        'elapsed_seconds':time.perf_counter()-STARTED,
        'limitation':'Two real parameter updates per variant, four train images; not accuracy/convergence evidence. '
            'Checkpoint-restored second step verified; CUDA scatter may prevent bitwise replay across runs. '
            'Smoke weights cannot seed formal experiments. Validation/test images remain unread.'}
    d.write_json(output/'smoke_summary_v1.json',summary)
    return summary


def fit(project,weights,provenance,cfg,variant,device,output,progress,resume=None):
    seed_runtime(cfg['seed'])
    signature=fingerprint({'core':provenance['core_fingerprint'],'variant':variant})
    model=classifier.RegionalDRModel(variant,pretrained=resume is None,seed=cfg['seed']).to(device)
    optimizer,scheduler=make_optimizer(model,cfg)
    state={'best':None,'best_epoch':None,'patience_anchor':None,'bad_epochs':0,'history':[],
           'best_checkpoint_file':None,'best_checkpoint_sha256':None}
    start_epoch=global_step=0
    if resume is not None:
        loaded=load_checkpoint(resume,model,optimizer,scheduler,signature,'train',device)
        start_epoch=loaded['epoch_completed']; global_step=loaded['global_step']; state=loaded['selection_state']
        require(0<start_epoch<cfg['epochs'],'Checkpoint already reached requested epoch limit.')
        require(state['bad_epochs']<cfg['patience'],'Checkpoint already triggered early stopping.')
        best_name=state['best_checkpoint_file']
        require(isinstance(best_name,str) and Path(best_name).name==best_name and
                (output/best_name).is_file() and d.sha256(output/best_name)==state['best_checkpoint_sha256'],
                'Best checkpoint missing/changed in resumed run.')
    train=d.AptosDataset(project,'train',seed=cfg['seed'],verify_file_hash=True)
    val=d.AptosDataset(project,'validation',seed=cfg['seed'],verify_file_hash=True)
    progress.updates=global_step
    for epoch in range(start_epoch,cfg['epochs']):
        train.set_epoch(epoch); model.train()
        progress.reset(f'TRAIN {variant} epoch {epoch+1}/{cfg["epochs"]}',len(train))
        numerator=denominator=0.; started=time.perf_counter(); lrs=[g['lr'] for g in optimizer.param_groups]
        for indices in index_groups(len(train),4,cfg['seed'],epoch):
            step=train_group(model,optimizer,train,indices,weights,cfg,device,progress)
            numerator+=step['numerator']; denominator+=step['denominator']; global_step+=1
            del step
        scheduler.step()
        progress.reset(f'VALIDATION {variant} epoch {epoch+1}/{cfg["epochs"]}',len(val))
        metrics,predictions=evaluate(model,val,weights,cfg['batch_size'],device,progress)
        require(metrics['qwk'] is not None and math.isfinite(metrics['qwk']),'Undefined/nonfinite validation QWK.')
        best_update=is_better(metrics,state['best'],cfg)
        if state['patience_anchor'] is None or metrics['qwk']>state['patience_anchor']+cfg['min_qwk_delta']:
            state['patience_anchor']=metrics['qwk']; state['bad_epochs']=0
        else: state['bad_epochs']+=1
        row={'epoch':epoch+1,'optimizer_steps':global_step,'train_weighted_loss':numerator/denominator,
             'validation':metrics,'lr_used':lrs,'elapsed_seconds':time.perf_counter()-started,
             'bad_epochs':state['bad_epochs']}
        state['history'].append(row)
        previous_best=state['best_checkpoint_file']
        if best_update:
            state['best']=metrics; state['best_epoch']=epoch+1
            state['best_checkpoint_file']=f'best_epoch_{epoch+1:04d}.pt'
            state['best_checkpoint_sha256']=None
            best_payload=checkpoint_payload(model,optimizer,scheduler,signature,'train',epoch+1,global_step,state)
            atomic_checkpoint(output/state['best_checkpoint_file'],best_payload)
            state['best_checkpoint_sha256']=d.sha256(output/state['best_checkpoint_file'])
        np.savez_compressed(output/f'validation_epoch_{epoch+1:04d}.npz',**predictions)
        atomic_checkpoint(output/'last.pt',checkpoint_payload(model,optimizer,scheduler,signature,'train',epoch+1,global_step,state))
        d.write_json(output/'training_history_v1.json',state['history'])
        # Keep the prior best until last.pt commits the new best reference; an interrupted
        # epoch can therefore resume safely without a mismatched best checkpoint.
        if best_update and previous_best and previous_best!=state['best_checkpoint_file']:
            old_best=output/previous_best
            if old_best.is_file(): old_best.unlink()
        print(f'Epoch {epoch+1}/{cfg["epochs"]} | train CE={row["train_weighted_loss"]:.5f} | '
              f'val CE={metrics["weighted_loss"]:.5f} | val QWK={metrics["qwk"]:.5f} | '
              f'val macro-F1={metrics["macro_f1"]:.5f} | best epoch={state["best_epoch"]}',flush=True)
        if state['bad_epochs']>=cfg['patience']: break
    verify_readonly(project,provenance,[])  # each train/validation PNG was checked on read
    summary={'status':'TRAINING_COMPLETE','version':VERSION,'mode':'train','variant':variant,
             'epochs_completed':len(state['history']),'optimizer_steps':global_step,
             'best_epoch':state['best_epoch'],'best_validation_metrics':state['best'],
             'stopped_early':state['bad_epochs']>=cfg['patience'], 'train_images':2453,'validation_images':527,
             'test_images_read':0,'best_checkpoint':state['best_checkpoint_file'],'last_checkpoint':'last.pt',
             'elapsed_seconds_this_process':time.perf_counter()-STARTED,
             'limitation':'Validation selects model; final held-out test/external evaluation and explanation remain separate.'}
    d.write_json(output/'training_summary_v1.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--mode',choices=['smoke','train'],default='smoke')
    parser.add_argument('--variant',choices=list(classifier.ABLATIONS))
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--classifier-run',type=Path); parser.add_argument('--smoke-run',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--batch-size',type=int,choices=[1,2,4],default=2)
    parser.add_argument('--epochs',type=int,default=30); parser.add_argument('--patience',type=int,default=7)
    parser.add_argument('--backbone-lr',type=float,default=1e-5); parser.add_argument('--new-lr',type=float,default=1e-4)
    parser.add_argument('--weight-decay',type=float,default=1e-4); parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args(); output=None
    try:
        require(args.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable.')
        device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
        require(args.epochs>=2 and args.patience>0 and 0<=args.seed<2**32,'Invalid epoch/patience/seed setting.')
        require(all(math.isfinite(v) for v in (args.backbone_lr,args.new_lr,args.weight_decay)) and
                0<args.backbone_lr<args.new_lr and args.weight_decay>=0,'Need 0 < backbone LR < new LR and decay>=0.')
        require((args.mode=='smoke' and args.variant is None and args.resume is None) or
                (args.mode=='train' and args.variant is not None),'Smoke checks all variants; train requires --variant.')
        cfg={**SETTINGS,'batch_size':args.batch_size,'accumulation_steps':4//args.batch_size,
             'epochs':args.epochs,'patience':args.patience,'backbone_lr':args.backbone_lr,
             'new_lr':args.new_lr,'weight_decay':args.weight_decay,'seed':args.seed}
        torch.set_num_threads(min(4,torch.get_num_threads())); seed_runtime(cfg['seed'])
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        root=args.root.resolve()
        project,weights,ids,provenance=load_context(root,args.classifier_run,cfg,device)
        resume=None
        if args.mode=='train':
            smoke_source=resolve_run(root,args.smoke_run,VERSION,'smoke_summary_v1.json')
            smoke=json.loads((smoke_source/'smoke_summary_v1.json').read_text(encoding='utf-8'))
            require(smoke.get('status')=='TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW' and
                    smoke.get('core_fingerprint')==provenance['core_fingerprint'],
                    'A matching smoke PASS is required for these settings/code/data/environment/device.')
            provenance['upstream_smoke_run']=str(smoke_source)
            if args.resume is not None:
                resume=(args.resume if args.resume.is_absolute() else root/args.resume).resolve()
                require(resume.is_file() and resume.name=='last.pt','Resume requires an existing last.pt (not best/smoke).')
                require(resume.parent.parent==(root/'environment/data_audit'/VERSION).resolve(),
                        'Resume checkpoint must belong to this project training_v1 run directory.')
                output=resume.parent
                existing=json.loads((output/'training_config_v1.json').read_text(encoding='utf-8'))
                require(existing['mode']=='train' and existing['variant']==args.variant and
                        existing['core_fingerprint']==provenance['core_fingerprint'],'Resume run configuration differs.')
        if output is None:
            output=root/'environment/data_audit'/VERSION/('run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
            output.mkdir(parents=True,exist_ok=False)
            d.write_json(output/'training_config_v1.json',{'version':VERSION,'mode':args.mode,
                'variant':args.variant,'settings':cfg,**provenance})
        with Progress() as progress:
            if args.mode=='smoke': summary=smoke_check(project,weights,ids,provenance,cfg,device,output,progress)
            else: summary=fit(project,weights,provenance,cfg,args.variant,device,output,progress,resume)
        if args.mode=='smoke':
            print('\nTRAINING SMOKE CHECK: PASS (technical updates only; formal training not started)')
            print(f'Device: {device} | Float32 | Microbatch: {cfg["batch_size"]} | Accumulation: {cfg["accumulation_steps"]} | Effective batch: 4')
            print(f'AdamW | Backbone LR: {cfg["backbone_lr"]} | New LR: {cfg["new_lr"]} | Weight decay: {cfg["weight_decay"]}')
            print('Weighted accumulation: PASS | All active module updates: PASS | CNN BN train: PASS')
            print('Cosine schedule: PASS | Model/optimizer/scheduler/RNG checkpoint restore: PASS')
            for variant,row in summary['results'].items():
                print(f'{variant}: optimizer steps=2 | step 2 after checkpoint restore=PASS | parameters={row["trainable_parameters"]}')
            print('Unique train images: 4 | Training exposures: 32 | Validation/test images read: 0')
            print('Total optimizer steps: 8 | Input files unchanged: PASS | Smoke checkpoints: diagnostic only')
            if device.type=='cuda':
                print(f'CUDA peak allocated: {summary["CUDA_peak_allocated_mib"]:.1f} MiB | reserved: {summary["CUDA_peak_reserved_mib"]:.1f} MiB (this check)')
            print('Return this summary before choosing the first formal training run. No image preview required.')
            print('After review passes: remember to commit train_v1.py.')
        else:
            print(f'\nTRAINING COMPLETE | {args.variant} | Best epoch: {summary["best_epoch"]} | Test images read: 0')
        print(f'Total elapsed: {time.perf_counter()-STARTED:.1f}s\nSaved: {output}')
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        message=f'{type(exc).__name__}: {exc}'
        print(f'\nTRAINING RUN STOPPED | {message}',flush=True)
        if output is not None:
            d.write_json(output/('interruption_'+uuid4().hex[:6]+'.json'),{'status':'STOPPED','error':message,
                'note':'Existing atomic epoch checkpoints remain intact. No silent batch-size/precision fallback.'})
        if isinstance(exc,torch.OutOfMemoryError):
            print('CUDA memory was insufficient. Return this output; the script did not silently change the experiment.')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
