r"""CP2 DenseNet121 denseblock2 features -> projection -> regional average pooling.

Run beside dataset_v1.py, boundary_map_v1.py and superpixels_v1.py:
    python regional_features_v1.py
First run downloads official ImageNet weights (~31 MB) to the PyTorch cache.
The CLI checks 4 TRAIN images, both superpixel variants, and ONE-image backward
propagation without an optimiser step. It does not train a classifier.

Training API (all label maps must match the CURRENT augmented RGB):
    model = RegionalFeatureExtractor().to(device)
    result = model(batch['cnn'].to(device), label_maps)
    # tokens: [B,Rmax,256]; key_padding_mask: True ONLY for padded slots.
    # geometry: [B,Rmax,3], ordered centroid_y, centroid_x, relative_area.
    # Geometry is returned separately; no positional embedding or GAT yet.

One token per actual region; K=256 is the initial target, not a hard sequence
length. Pooling equals bilinear upsampling of projected features to the ORIGINAL
label-map size (align_corners=False), followed by region mean. A precomputed
linear weight matrix avoids materialising [256,384,384] feature tensors.
Labels are never interpolated. This is NOT an eye/FOV mask. Parameters remain
trainable; never cache these preview tokens for later fine-tuning/augmentation.
"""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import threading
import time
from uuid import uuid4

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading regional-feature libraries...', flush=True)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import torchvision
from torchvision.models import DenseNet121_Weights, densenet121

VERSION = 'regional_features_v1'
SETTINGS = {
    'backbone': 'DenseNet121', 'weights': 'IMAGENET1K_V1',
    'feature_endpoint': 'features.denseblock2, before transition2',
    'input': 'existing ImageNet-normalised 384x384 CNN branch; no further resize/smoothing',
    'backbone_feature_shape': [512,48,48],
    'projection': 'learnable Conv2d(512,256,kernel_size=1,bias=True); no added activation',
    'projection_seed': 42, 'projection_initialisation': 'PyTorch Conv2d default; not trained',
    'pooling': 'mean over original region of bilinearly aligned projected features',
    'alignment': 'PyTorch interpolate bilinear align_corners=False convention; edge clamping',
    'implementation': 'CPU float64 accumulation of interpolation weights, then float32 matrix multiply',
    'pooling_accumulation': 'float32 even under outer autocast; gradients retained',
    'geometry': ['(mean_y+0.5)/H', '(mean_x+0.5)/W', 'area/(H*W)'],
    'batching': 'zero pad to actual max R in batch; key_padding_mask True means ignore',
    'canvas': 'all pixels and all regions, including black/grey background; no FOV filtering',
    'gradients': 'CNN and projection trainable; segmentation/geometry not differentiable',
    'preview': 'eval mode, four images one at a time; backward batch=1; no parameter update',
}
SOURCES = {
    'proposal': 'CP1 sections 3.5.2 and 3.6.1; Table 4.1 T4 (256-dimensional regional features)',
    'densenet': 'https://arxiv.org/abs/1608.06993 (Huang et al., CVPR 2017)',
    'weights': 'https://docs.pytorch.org/vision/0.27/models/generated/torchvision.models.densenet121.html',
    'regional_pooling': 'https://papers.miccai.org/miccai-2024/paper/3756_paper.pdf (Playout et al., 2024, section 2.1)',
    'interpolation': 'https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.interpolate.html',
    'scope': 'Playout supports feature interpolation and region averaging, not this exact '
             'DenseNet truncation, 1x1 projection, bilinear convention or implementation. '
             'No prototype matching, Leiden pooling or black-border merging is adopted.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class RegionPoolingPlan:
    weights: torch.Tensor       # CPU float32 [R,h*w]; each row sums to one
    geometry: torch.Tensor      # CPU float32 [R,3]
    areas: torch.Tensor         # CPU int64 [R]
    feature_hw: tuple
    image_hw: tuple


def make_pooling_plan(labels, feature_hw=(48,48)):
    """Build exact region means of a bilinear-upsampled feature grid, without RGB.

    A label map must use every integer ID 0..R-1. Region r remains token r.
    One-pixel regions are supported, even though SLIC cleanup normally avoids them.
    Regenerate the plan after ANY change to the label map/augmentation.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)
    require(labels.ndim == 2 and labels.size > 0, 'Expected a nonempty HxW label map.')
    require(np.issubdtype(labels.dtype,np.integer), 'Labels must be integers, not resized floats.')
    unique = np.unique(labels)
    require(np.array_equal(unique,np.arange(unique.size)), 'Labels must be contiguous IDs 0..R-1.')
    H,W = labels.shape
    h,w = map(int,feature_hw)
    require(h > 0 and w > 0, 'Feature grid must be positive.')
    R = unique.size
    ids = labels.astype(np.int64,copy=False).ravel()
    yy,xx = np.indices((H,W),dtype=np.float64)
    areas = np.bincount(ids,minlength=R)
    geometry = np.stack(((np.bincount(ids,weights=yy.ravel())/areas+.5)/H,
                         (np.bincount(ids,weights=xx.ravel())/areas+.5)/W,
                         areas/(H*W)),axis=1).astype(np.float32)
    # Same half-pixel geometry and border handling as interpolate(..., align_corners=False).
    sy = np.clip((yy+.5)*h/H-.5,0,h-1)
    sx = np.clip((xx+.5)*w/W-.5,0,w-1)
    y0,x0 = np.floor(sy).astype(np.int64),np.floor(sx).astype(np.int64)
    y1,x1 = np.minimum(y0+1,h-1),np.minimum(x0+1,w-1)
    dy,dx = sy-y0,sx-x0
    weights = np.zeros(R*h*w,dtype=np.float64)
    for y,x,a in ((y0,x0,(1-dy)*(1-dx)),(y0,x1,(1-dy)*dx),
                  (y1,x0,dy*(1-dx)),(y1,x1,dy*dx)):
        index = ids*(h*w)+(y*w+x).ravel()
        weights += np.bincount(index,weights=a.ravel(),minlength=weights.size)
    weights = (weights.reshape(R,h*w)/areas[:,None]).astype(np.float32)
    require(np.allclose(weights.sum(axis=1),1,atol=2e-6,rtol=0), 'Invalid pooling row sums.')
    return RegionPoolingPlan(torch.from_numpy(weights),torch.from_numpy(geometry),
                             torch.from_numpy(areas),(h,w),(H,W))


def pool_regions(projected, plan):
    """Differentiable weighted mean; no gradient to fixed segmentation labels."""
    require(projected.ndim == 3 and tuple(projected.shape[-2:]) == plan.feature_hw,
            'Projected feature shape does not match the pooling plan.')
    require(projected.is_floating_point(), 'Features must be floating point.')
    with torch.autocast(device_type=projected.device.type,enabled=False):
        return plan.weights.to(projected.device) @ projected.float().flatten(1).T


def pack_regions(tokens, plans):
    require(len(tokens) == len(plans) and len(tokens) > 0, 'One pooling plan per image required.')
    require(all(t.ndim == 2 and t.shape[0] == p.areas.numel() for t,p in zip(tokens,plans)),
            'Token count and region metadata differ.')
    padded = pad_sequence(tokens,batch_first=True,padding_value=0)
    counts = torch.tensor([t.shape[0] for t in tokens],device=padded.device,dtype=torch.long)
    mask = torch.arange(padded.shape[1],device=padded.device)[None,:] >= counts[:,None]
    geometry = pad_sequence([p.geometry.to(padded.device) for p in plans],batch_first=True)
    areas = pad_sequence([p.areas.to(padded.device) for p in plans],batch_first=True)
    return {'tokens':padded,'key_padding_mask':mask,'region_counts':counts,
            'geometry':geometry,'areas':areas}


class RegionalFeatureExtractor(nn.Module):
    def __init__(self, *, pretrained=True, seed=42):
        super().__init__()
        # CPU-only initialisation isolated from training RNG. No silent weight fallback.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            full = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
            children = OrderedDict()
            for name,module in full.features.named_children():
                children[name] = module
                if name == 'denseblock2':
                    break
            self.backbone = nn.Sequential(children)
            self.projection = nn.Conv2d(512,256,kernel_size=1,bias=True)
        self.pretrained = pretrained

    def extract(self, cnn):
        require(cnn.ndim == 4 and tuple(cnn.shape[1:]) == (3,384,384),
                'Expected existing normalised CNN input [B,3,384,384].')
        require(cnn.is_floating_point(), 'CNN input must be floating point.')
        spatial = self.backbone(cnn)
        require(tuple(spatial.shape[1:]) == (512,48,48), 'Unexpected denseblock2 feature shape.')
        return self.projection(spatial)

    def forward(self, cnn, label_maps):
        require(len(label_maps) == cnn.shape[0], 'One aligned label map per input image required.')
        plans = [make_pooling_plan(labels) for labels in label_maps]
        require(all(p.image_hw == (384,384) for p in plans), 'Label maps must remain 384x384.')
        projected = self.extract(cnn)
        return pack_regions([pool_regions(f,p) for f,p in zip(projected,plans)],plans)


def pooling_check(device):
    """Independent, explicit interpolation/region mean reference incl. one-pixel region."""
    labels = np.zeros((11,13),np.int32)
    labels[:,7:] = 1
    labels[2:9,3:6] = 2
    labels[0,0] = 3
    plan = make_pooling_plan(labels,(3,4))
    f = (torch.arange(60,device=device,dtype=torch.float32).reshape(5,3,4)/17).requires_grad_()
    actual = pool_regions(f,plan)
    up = F.interpolate(f[None],size=labels.shape,mode='bilinear',align_corners=False)[0]
    expected = torch.stack([up[:,torch.from_numpy(labels==r).to(device)].mean(1) for r in range(4)])
    error = float((actual-expected).abs().max().detach())
    require(torch.allclose(actual,expected,atol=2e-6,rtol=2e-6), 'Region pooling reference mismatch.')
    importance = torch.arange(20,device=device,dtype=torch.float32).reshape(4,5)/19
    ga = torch.autograd.grad((actual*importance).sum(),f,retain_graph=True)[0]
    ge = torch.autograd.grad((expected*importance).sum(),f)[0]
    require(torch.allclose(ga,ge,atol=2e-6,rtol=2e-6), 'Pooling backward reference mismatch.')
    constant = pool_regions(torch.ones_like(f)*3,plan)
    require(torch.allclose(constant,torch.full_like(constant,3),atol=2e-6,rtol=0),
            'Region averaging changed a constant feature.')
    plan2 = make_pooling_plan((labels>0).astype(np.int32),(3,4))
    packed = pack_regions([actual.detach(),pool_regions(f.detach(),plan2)],[plan,plan2])
    require(packed['region_counts'].tolist()==[4,2] and packed['key_padding_mask'].tolist()==
            [[False]*4,[False,False,True,True]], 'Variable-length batch mask incorrect.')
    require(torch.count_nonzero(packed['tokens'][1,2:]).item()==0, 'Padded tokens are not zero.')
    return {'bilinear_mean_reference':True,'pooling_gradient_reference':True,
            'one_pixel_region_retained':True,'constant_features_preserved':True,
            'variable_region_padding':True,'reference_max_abs_error':error}


class Progress:
    def __init__(self,total=11):
        self.total,self.done,self.stage,self.images = total,0,'starting',0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.refresh,daemon=True)
    def show(self):
        elapsed = time.perf_counter()-STARTED
        eta = f'{elapsed/self.done*(self.total-self.done):.1f}s' if self.done and self.done>=2 else '--'
        print(f'Regional checks {self.done}/{self.total} | Images {self.images}/4 | '
              f'Elapsed {elapsed:.1f}s | ETA {eta} | {self.stage}',flush=True)
    def refresh(self):
        while not self.stop.wait(1 if sys.stdout.isatty() else 5):
            self.show()
    def advance(self,stage):
        self.done += 1
        self.stage = stage
    def __enter__(self):
        self.show(); self.thread.start(); return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join(); self.show()


def state_sha256(model):
    digest = hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args()
    output = None
    try:
        import dataset_v1 as d
        import boundary_map_v1 as bm
        import superpixels_v1 as sp
        require((d.VERSION,bm.VERSION,sp.VERSION)==('input_pipeline_v1','boundary_map_v1','superpixels_v1'),
                'Expected existing dataset, boundary and superpixels v1 modules.')
        require(args.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable.')
        device = torch.device('cuda' if args.device=='cuda' or
                              (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
        torch.set_num_threads(min(4,torch.get_num_threads()))
        # CLI evidence is float32/repeatable on one backend; training controls its own settings.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        output = args.root.resolve()/'environment/data_audit'/VERSION/(
            'run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        output.mkdir(parents=True,exist_ok=False)
        results,all_tokens,all_plans = [],{'slic':[],'boundary_slic':[]},{'slic':[],'boundary_slic':[]}
        saved_labels,checks = {},{}
        with Progress() as progress:
            project = d.load_project(args.root)
            train = d.AptosDataset(project,'train',verify_file_hash=True)
            batch = next(iter(d.make_loader(train,epoch=0,batch_size=4)))
            d.validate_batch(batch,{r['id_code']:r for r in project['rows']['train']})
            rgb_before,cnn_before = batch['rgb'].clone(),batch['cnn'].clone()
            progress.advance('loading ImageNet weights (first run may download ~31 MB)')
            model = RegionalFeatureExtractor().to(device).eval()
            before_state = state_sha256(model)
            progress.advance('ImageNet backbone ready; projection initialised, untrained')
            checks.update(pooling_check(device))
            progress.advance('independent pooling and padding checks passed')
            for i,image_id in enumerate(batch['id_code']):
                progress.stage = f'image {i+1}/4 | feature extraction'
                with torch.no_grad():
                    projected = model.extract(batch['cnn'][i:i+1].to(device))[0]
                require(torch.isfinite(projected).all().item(),'Non-finite CNN features.')
                prepared = bm.boundary_map(batch['rgb'][i].to(device),return_details=True)
                for method,gamma in (('slic',0.0),('boundary_slic',2.0)):
                    def update_iteration(step,total):
                        progress.stage = f'image {i+1}/4 | {method} | SLIC iteration {step}/{total}'
                    regions = sp.segment_superpixels(prepared['lab'],prepared['boundary'][0],
                               boundary_weight=gamma,progress=update_iteration)
                    sp.validate_regions(regions)
                    labels = regions['labels']
                    plan = make_pooling_plan(labels)
                    with torch.no_grad():
                        tokens = pool_regions(projected,plan)
                    require(torch.isfinite(tokens).all().item(),'Non-finite regional tokens.')
                    require(np.array_equal(plan.areas.numpy(),regions['areas']),'Region area/ID mismatch.')
                    require(np.allclose(plan.geometry[:,:2].numpy(),(regions['centroids_yx']+.5)/d.SIZE,
                                        rtol=0,atol=1e-7),'Region centroid/ID mismatch.')
                    filename = f'{image_id}_{method}_tokens.npz'
                    np.savez_compressed(output/filename,tokens=tokens.cpu().numpy(),labels=labels,
                                        areas=plan.areas.numpy(),geometry=plan.geometry.numpy())
                    all_tokens[method].append(tokens.detach())
                    all_plans[method].append(plan)
                    if i == 0:
                        saved_labels[method] = labels.copy()
                    results.append({'id_code':image_id,'diagnosis':int(batch['label'][i]),'split':'train',
                                    'epoch':0,'method':method,'token_shape':list(tokens.shape),
                                    'input_png_sha256':project['processed'][image_id]['output_sha256'],
                                    'output_file':filename,'output_sha256':d.sha256(output/filename)})
                    print(f'{image_id} | {method} | tokens={list(tokens.shape)} | finite=PASS',flush=True)
                progress.images = i+1
                progress.advance(f'image {i+1}/4 regional features saved')
            packed_shapes = {}
            for method in all_tokens:
                packed = pack_regions(all_tokens[method],all_plans[method])
                require(int((~packed['key_padding_mask']).sum())==sum(t.shape[0] for t in all_tokens[method]),
                        'Real batch valid-token mask is inconsistent.')
                require(torch.count_nonzero(packed['tokens'][packed['key_padding_mask']]).item()==0,
                        'Real batch padding is not zero.')
                packed_shapes[method] = {'shape':list(packed['tokens'].shape),
                                        'counts':packed['region_counts'].tolist(),
                                        'padded_slots':int(packed['key_padding_mask'].sum())}
            checks['real_batch_padding'] = True
            progress.advance('both variable-length batches packed')
            # Exercise the same forward API future training uses. Evaluation replay is numerical,
            # not a claim of bit-identical results across CPU/CUDA or library versions.
            with torch.no_grad():
                replay = model(batch['cnn'][:1].to(device),[saved_labels['boundary_slic']])
            reference = all_tokens['boundary_slic'][0]
            replay_error = float((replay['tokens'][0]-reference).abs().max())
            require(torch.allclose(replay['tokens'][0],reference,atol=2e-5,rtol=2e-5),'Eval replay mismatch.')
            checks['eval_replay'] = True
            checks['eval_replay_max_abs_error'] = replay_error
            progress.advance('evaluation forward API replay checked')
            # eval() retains gradients, while preventing this diagnostic from changing BN buffers.
            # This loss only checks the graph; it is NOT the later DR classification objective.
            grad_input = batch['cnn'][:1].to(device).detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            out = model(grad_input,[saved_labels['boundary_slic']])
            diagnostic_loss = out['tokens'].square().mean()
            diagnostic_loss.backward()
            require(grad_input.grad is not None and torch.isfinite(grad_input.grad).all().item()
                    and grad_input.grad.abs().max().item()>0,'No finite nonzero input gradient.')
            for name,parameter in model.named_parameters():
                require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                        f'Missing/non-finite parameter gradient: {name}')
            require(model.backbone.conv0.weight.grad.abs().max().item()>0 and
                    model.projection.weight.grad.abs().max().item()>0,'Backbone/projection gradient is zero.')
            require(state_sha256(model)==before_state,'Check changed model parameters or BN buffers.')
            require(torch.equal(batch['rgb'],rgb_before) and torch.equal(batch['cnn'],cnn_before),
                    'Check changed RGB/CNN input.')
            checks.update(backward_finite=True,backbone_projection_gradients=True,
                          parameters_and_bn_unchanged=True,rgb_cnn_unchanged=True)
            model.zero_grad(set_to_none=True)
            progress.advance('one-image backward passed; no optimizer step')
            checkpoint = Path(torch.hub.get_dir())/'checkpoints'/Path(DenseNet121_Weights.IMAGENET1K_V1.url).name
            d.write_json(output/'regional_features_config_v1.json',{
                'version':VERSION,'settings':SETTINGS,'sources':SOURCES,'seed':d.SEED,'epoch':0,
                'input_settings':d.SETTINGS,'boundary_settings':bm.SETTINGS,'superpixel_settings':sp.SETTINGS,
                'code_sha256':{Path(m.__file__).name:d.sha256(Path(m.__file__)) for m in (d,bm,sp)},
                'regional_code_sha256':d.sha256(Path(__file__)),'source_metadata_sha256':project['source_sha256'],
                'versions':{'python':platform.python_version(),'torch':torch.__version__,
                            'torchvision':torchvision.__version__,'numpy':np.__version__,'pillow':d.PIL.__version__},
                'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                'weight_url':DenseNet121_Weights.IMAGENET1K_V1.url,
                'checkpoint_sha256':d.sha256(checkpoint) if checkpoint.is_file() else None,
                'initial_module_state_sha256':before_state,
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
            report = {'status':'TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW','checks':checks,
                      'train_images_read':4,'validation_images_read':0,'test_images_read':0,
                      'pretrained_weights_loaded':True,'projection_trained':False,'optimizer_steps':0,
                      'backbone_shape_per_image':[1,512,48,48],'projected_shape_per_image':[1,256,48,48],
                      'backward_batch_size':1,'backward_mode':'eval, float32, no optimizer step',
                      'device':str(device),'packed_batches':packed_shapes,'results':results,
                      'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                      'elapsed_seconds':time.perf_counter()-STARTED,
                      'limitation':'Small-batch engineering checks only; no classifier, graph, ViT, optimiser '
                                   'or training performance. Bilinear alignment cannot restore lost CNN spatial detail.'}
            d.write_json(output/'regional_features_summary_v1.json',report)
            progress.advance('8 token files, configuration and summary saved')
        print('\nREGIONAL FEATURE CHECK: PASS (small-batch technical checks only)')
        print(f'Device: {device} | ImageNet weights: loaded | Projection: initialised, untrained')
        print('DenseNet block2: [1, 512, 48, 48] | Projected: [1, 256, 48, 48]')
        for method,packed in packed_shapes.items():
            print(f'{method}: {packed}')
        print('Pooling/reference gradients: PASS | Variable token padding: PASS | Eval replay: PASS')
        print('Backward: PASS (batch=1, eval mode) | Optimizer steps: 0 | RGB/CNN unchanged: PASS')
        print('Train images: 4 | Validation: 0 | Test: 0')
        if report['cuda_peak_allocated_mib'] is not None:
            print(f"CUDA peak allocated: {report['cuda_peak_allocated_mib']:.1f} MiB (this check only)")
        print(f'Total elapsed: {time.perf_counter()-STARTED:.1f}s\nSaved: {output}')
        print('Return this summary. No visual feature-map review is required.')
        print('After review passes: remember to commit regional_features_v1.py.')
        return 0
    except Exception as exc:
        print(f'\nREGIONAL FEATURE CHECK: FAIL | {type(exc).__name__}: {exc}',flush=True)
        if output is not None:
            (output/'regional_features_failure_v1.json').write_text(json.dumps(
                {'status':'FAIL','error':f'{type(exc).__name__}: {exc}'},indent=2)+'\n',encoding='utf-8')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
