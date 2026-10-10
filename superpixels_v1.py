r"""CP2 SLIC / boundary-aware SLIC with one shared implementation.

Run beside dataset_v1.py and boundary_map_v1.py: python superpixels_v1.py
No new packages beyond the existing torch/torchvision/Pillow/NumPy environment.
Default: 4 augmented TRAIN images, 2 methods, 256 initial seeds, 10 iterations.
This entry point checks a small batch; it does NOT process all training epochs.

Later training: augment RGB first, compute boundary_map(return_details=True),
then call segment_superpixels(details['lab'][i], details['boundary'][i,0],
                            boundary_weight=0.0 or 2.0).
Both methods use ORIGINAL unsmoothed native Lab colour features. Only the
multiplicative path penalty changes. Outputs are CPU NumPy label maps and
region metadata, ready for later pooling. Masks are recovered as labels == id;
there is no FOV mask or background removal. Never reuse epoch-0 labels for
another epoch's augmented RGB. Actual region counts may differ from 256.
"""
import argparse
import csv
import heapq
import json
import math
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

STARTED = time.perf_counter()
if __name__ == '__main__':
    print('Loading superpixel libraries...', flush=True)
import numpy as np
from PIL import Image, ImageDraw
import torch
import boundary_map_v1 as bm

VERSION = 'superpixels_v1'
SETTINGS = {
    'initial_regions': 256, 'compactness': 10.0, 'iterations': 10,
    'boundary_weight': 2.0, 'minimum_area_fraction': 0.25,
    'colour': 'unsmoothed native CIELAB from boundary_map_v1',
    'distance': '(squared_Lab_distance + (m/S)^2 * squared_xy_distance) * (1 + gamma * path_max)',
    'S': 'image side / sqrt(initial_regions); 24 for 384x384 and 256 seeds',
    'initialisation': 'regular square grid, shift to minimum unsmoothed Lab central-difference energy in 3x3',
    'initial_ties': 'closest to original seed, then row-major offset',
    'candidate_window': '(2*S+1)^2 around round-half-up current centroid; clipped to canvas',
    'path': 'integer rounded DDA from rounded centroid to pixel; endpoints included; one step per dominant-axis pixel',
    'path_rounding': 'absolute displacement rounded half up, then apply displacement sign',
    'assignment_ties': 'lowest seed id',
    'mean_update': 'CPU NumPy float64 bincount in row-major order, cast centres to float32',
    'empty_seed': 'keep previous centre; unused labels absent from final output',
    'uncovered_pixel': 'evaluate same distance against all centres in chunks; count and report',
    'connectivity': 'split disconnected 4-neighbour components; merge small components by longest shared edge',
    'merge_ties': 'lowest current component id; process smallest area first',
    'final_ids': 'contiguous from zero, ordered by first row-major pixel',
    'fov_mask': False, 'canvas': 'all pixels including padding in BOTH variants',
    'status': 'initial implementation settings, not tuned or proven optimal for APTOS',
}
SOURCES = {
    'slic': 'https://doi.org/10.1109/TPAMI.2012.120',
    'slic_authors': 'https://www.epfl.ch/labs/ivrl/research/slic-superpixels/',
    'slic_paper_copy': 'https://www.cs.jhu.edu/~ayuille/JHUcourses/VisionAsBayesianInference2020/4/s/Achanta_SLIC_PAMI2012.pdf',
    'contour_path': 'https://doi.org/10.1016/j.cviu.2018.01.006',
    'contour_equations_7_8': 'https://arxiv.org/html/1903.07193v2',
    'scope': 'SLIC-style Lab/spatial baseline plus a SCALP-inspired path-maximum '
             'multiplier. Not full SCALP: no path-colour averaging, robust neighbourhood '
             'features, spectral embedding, learned contour detector or hard presegmentation. '
             'gamma=2, DDA sampling and merge rule are project choices.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def raster_offsets(dy, dx):
    """Integer line displacements, shape [steps+1, ...], including endpoints.

    Samples beyond a short line's endpoint repeat it; they do not change max.
    This explicitly defined DDA convention is not claimed bit-identical to
    every Bresenham implementation, especially when ties occur.
    """
    dy, dx = np.broadcast_arrays(np.asarray(dy, np.int64), np.asarray(dx, np.int64))
    n = np.maximum(np.abs(dy), np.abs(dx))
    t = np.arange(int(n.max())+1).reshape((-1,) + (1,)*n.ndim)
    t = np.minimum(t, n)
    denom = 2*np.maximum(n, 1)
    oy = np.sign(dy) * ((2*np.abs(dy)*t + n) // denom)
    ox = np.sign(dx) * ((2*np.abs(dx)*t + n) // denom)
    return oy, ox


def initial_centres(lab_numpy, grid):
    h, w, _ = lab_numpy.shape
    step = h // grid
    p = np.pad(lab_numpy.astype(np.float64), ((1,1),(1,1),(0,0)), mode='reflect')
    gx, gy = p[1:-1,2:]-p[1:-1,:-2], p[2:,1:-1]-p[:-2,1:-1]
    energy = (gx*gx+gy*gy).sum(axis=2)
    offsets = sorted([(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)],
                     key=lambda v:(v[0]*v[0]+v[1]*v[1],v[0],v[1]))
    centres = []
    for gy in range(grid):
        for gx in range(grid):
            y, x = int((gy+.5)*step), int((gx+.5)*step)
            options = [(y+dy,x+dx) for dy,dx in offsets
                       if 0 <= y+dy < h and 0 <= x+dx < w]
            yy, xx = min(options, key=lambda q:energy[q[0],q[1]])
            centres.append([*lab_numpy[yy,xx], yy, xx])
    return np.asarray(centres,np.float32)


def components4(labels):
    """Label equal-valued 4-connected components, in row-major order."""
    h,w = labels.shape
    src = labels.ravel()
    comp = np.full(src.size,-1,np.int32)
    count = 0
    for seed in range(src.size):
        if comp[seed] >= 0:
            continue
        value = src[seed]
        comp[seed] = count
        stack = [seed]
        while stack:
            q = stack.pop()
            y,x = divmod(q,w)
            neighbours = []
            if x: neighbours.append(q-1)
            if x+1<w: neighbours.append(q+1)
            if y: neighbours.append(q-w)
            if y+1<h: neighbours.append(q+w)
            for r in neighbours:
                if comp[r] == -1 and src[r] == value:
                    comp[r] = count
                    stack.append(r)
        count += 1
    return comp.reshape(h,w),count


def shared_edges(labels):
    pairs = []
    for a,b in ((labels[:-1,:],labels[1:,:]),(labels[:,:-1],labels[:,1:])):
        different = a != b
        pairs.append(np.sort(np.stack((a[different],b[different]),axis=1),axis=1))
    all_pairs = np.concatenate(pairs)
    return np.unique(all_pairs,axis=0,return_counts=True)


def enforce_connectivity(raw_labels, minimum_area):
    comp,n = components4(raw_labels)
    parent = np.arange(n)
    sizes = np.bincount(comp.ravel(),minlength=n)
    adjacency = [dict() for _ in range(n)]
    pairs,lengths = shared_edges(comp)
    for (a,b),length in zip(pairs,lengths):
        a,b,length = int(a),int(b),int(length)
        adjacency[a][b] = length
        adjacency[b][a] = length
    heap = [(int(sizes[i]),i) for i in range(n) if sizes[i] < minimum_area]
    heapq.heapify(heap)
    merges = 0
    while heap:
        area,a = heapq.heappop(heap)
        if parent[a] != a or sizes[a] != area or area >= minimum_area:
            continue
        if not adjacency[a]:
            continue  # The entire canvas is one connected component.
        b = min(adjacency[a],key=lambda c:(-adjacency[a][c],c))
        parent[a] = b
        sizes[b] += sizes[a]
        adjacency[b].pop(a,None)
        for c,length in list(adjacency[a].items()):
            if c == b: continue
            adjacency[c].pop(a)
            total = adjacency[b].get(c,0)+length
            adjacency[b][c] = total
            adjacency[c][b] = total
        adjacency[a].clear()
        merges += 1
        if sizes[b] < minimum_area:
            heapq.heappush(heap,(int(sizes[b]),b))
    for a in range(n):
        r = a
        while parent[r] != r: r = int(parent[r])
        parent[a] = r
    flat = parent[comp.ravel()]
    unique,first,inverse = np.unique(flat,return_index=True,return_inverse=True)
    remap = np.empty(len(unique),np.int32)
    remap[np.argsort(first)] = np.arange(len(unique),dtype=np.int32)
    labels = remap[inverse].reshape(comp.shape)
    return labels, {'raw_components':n,'small_component_merges':merges,
                    'minimum_area':minimum_area,'final_regions':len(unique)}


def region_metadata(labels,lab_numpy):
    h,w = labels.shape
    ids = labels.ravel()
    count = int(ids.max())+1
    areas = np.bincount(ids,minlength=count)
    yy,xx = np.indices((h,w))
    cy = np.bincount(ids,weights=yy.ravel(),minlength=count)/areas
    cx = np.bincount(ids,weights=xx.ravel(),minlength=count)/areas
    means = np.stack([np.bincount(ids,weights=lab_numpy[:,:,c].ravel(),minlength=count)/areas
                      for c in range(3)],axis=1)
    return {'areas':areas.astype(np.int64), 'centroids_yx':np.stack((cy,cx),axis=1),
            'mean_lab':means}


def _full_search(lab_flat,boundary,centres,missing,width,spatial_weight,gamma):
    """Rare hole recovery uses the SAME cost, not a spatial-only shortcut."""
    output = []
    anchors = np.floor(centres[:,3:].detach().cpu().numpy()+.5).astype(np.int64)
    for chunk in missing.split(32):
        py,px = chunk//width,chunk%width
        dc = (lab_flat[chunk][None,:,:]-centres[:,:3,None].transpose(1,2)).square().sum(2)
        ds = (py[None,:]-centres[:,3,None]).square()+(px[None,:]-centres[:,4,None]).square()
        cost = dc + spatial_weight*ds
        if gamma:
            pyy,pxx = py.cpu().numpy(),px.cpu().numpy()
            oy,ox = raster_offsets(pyy[None,:]-anchors[:,0,None],pxx[None,:]-anchors[:,1,None])
            path = (anchors[None,:,0,None]+oy)*width+anchors[None,:,1,None]+ox
            indices = torch.as_tensor(path,device=lab_flat.device)
            strength = boundary.flatten()[indices].amax(dim=0)
            cost = cost*(1+gamma*strength)
        output.append(cost.argmin(dim=0))
    return torch.cat(output)


@torch.no_grad()
def segment_superpixels(lab,boundary=None,*,boundary_weight=0.0,n_segments=256,
                        compactness=10.0,iterations=10,minimum_area_fraction=.25,
                        progress=None,return_raw=False):
    """One shared SLIC routine; gamma=0 disables only the boundary multiplier.

    v1 accepts square float32 Lab with square-grid seed counts dividing the side.
    Assignment uses CPU/CUDA; ordered mean reductions and connectivity use CPU.
    All output pixels, including padding, receive one contiguous region ID.
    """
    require(isinstance(lab,torch.Tensor) and lab.ndim==3 and lab.shape[0]==3
            and lab.dtype==torch.float32 and lab.device.type in ('cpu','cuda'),
            'Expected float32 Lab [3,H,W] on CPU/CUDA.')
    h,w = lab.shape[1:]
    require(isinstance(n_segments,int) and n_segments>=4,'Invalid seed count.')
    grid = math.isqrt(n_segments)
    require(h==w and grid*grid==n_segments and h%grid==0 and h//grid>=4,
            'v1 needs square images, square seed count, and grid spacing >=4 dividing the side.')
    require(math.isfinite(boundary_weight) and boundary_weight>=0
            and math.isfinite(compactness) and compactness>0
            and isinstance(iterations,int) and iterations>0
            and math.isfinite(minimum_area_fraction) and 0<minimum_area_fraction<=1,
            'Invalid SLIC parameters.')
    require(torch.isfinite(lab).all().item(),'Lab contains nonfinite values.')
    if boundary_weight:
        require(isinstance(boundary,torch.Tensor) and boundary.shape==(h,w)
                and boundary.dtype==torch.float32 and boundary.device==lab.device,
                'Boundary must be float32 [H,W] on the same device.')
        require(torch.isfinite(boundary).all().item() and boundary.min().item()>=0
                and boundary.max().item()<=1,'Boundary must be finite in [0,1].')
    start = time.perf_counter()
    step = h//grid
    lab_numpy = lab.detach().cpu().permute(1,2,0).numpy().copy()
    centre_numpy = initial_centres(lab_numpy,grid)
    features = torch.from_numpy(lab_numpy.reshape(-1,3).copy()).to(lab.device)
    yy,xx = np.indices((h,w))
    cpu_features = np.column_stack((lab_numpy.reshape(-1,3),yy.ravel(),xx.ravel()))
    dy,dx = np.meshgrid(np.arange(-step,step+1),np.arange(-step,step+1),indexing='ij')
    oy,ox = raster_offsets(dy.ravel(),dx.ravel())
    lookup = torch.as_tensor((oy+step)*(2*step+1)+ox+step,device=lab.device)
    dy = torch.as_tensor(dy.ravel(),device=lab.device)
    dx = torch.as_tensor(dx.ravel(),device=lab.device)
    k = n_segments
    seed_ids = torch.arange(k,device=lab.device)[:,None]
    holes,empty_counts = [],[]
    with torch.autocast(device_type=lab.device.type,enabled=False):
        for iteration in range(iterations):
            centres = torch.as_tensor(centre_numpy,device=lab.device)
            anchors = torch.floor(centres[:,3:]+.5).to(torch.int64)
            py,px = anchors[:,0,None]+dy,anchors[:,1,None]+dx
            valid = (py>=0)&(py<h)&(px>=0)&(px<w)
            index = py.clamp(0,h-1)*w+px.clamp(0,w-1)
            dc = (features[index]-centres[:,:3,None].transpose(1,2)).square().sum(dim=2)
            ds = (py-centres[:,3,None]).square()+(px-centres[:,4,None]).square()
            cost = dc+(compactness/step)**2*ds
            if boundary_weight:
                windows = boundary.flatten()[index]
                path_max = windows[:,lookup].amax(dim=1)
                cost = cost*(1+boundary_weight*path_max)
            cost = cost.masked_fill(~valid,float('inf'))
            best = torch.full((h*w,),float('inf'),device=lab.device)
            best.scatter_reduce_(0,index.flatten(),cost.flatten(),reduce='amin',include_self=True)
            winners = valid & (cost==best[index])
            candidate = torch.where(winners,seed_ids,torch.full_like(index,k))
            labels = torch.full((h*w,),k,dtype=torch.int64,device=lab.device)
            labels.scatter_reduce_(0,index.flatten(),candidate.flatten(),reduce='amin',include_self=True)
            missing = torch.nonzero(labels==k).flatten()
            holes.append(int(missing.numel()))
            if missing.numel():
                labels[missing] = _full_search(features,boundary,centres,missing,w,
                                               (compactness/step)**2,boundary_weight)
            raw = labels.cpu().numpy()
            counts = np.bincount(raw,minlength=k)
            nonempty = counts>0
            empty_counts.append(int((~nonempty).sum()))
            for channel in range(5):
                sums = np.bincount(raw,weights=cpu_features[:,channel],minlength=k)
                centre_numpy[nonempty,channel] = sums[nonempty]/counts[nonempty]
            if progress is not None: progress(iteration+1,iterations)
    raw = raw.reshape(h,w).astype(np.int32)
    minimum_area = max(1,int(minimum_area_fraction*h*w/k))
    labels,stats = enforce_connectivity(raw,minimum_area)
    metadata = region_metadata(labels,lab_numpy)
    stats.update({'initial_regions':k,'iterations':iterations,'boundary_weight':boundary_weight,
                  'uncovered_pixels_by_iteration':holes,'empty_seeds_by_iteration':empty_counts,
                  'area_min':int(metadata['areas'].min()),'area_max':int(metadata['areas'].max()),
                  'elapsed_seconds':time.perf_counter()-start,'device':str(lab.device)})
    result = {'labels':labels, **metadata, 'stats':stats}
    if return_raw: result['raw_labels'] = raw
    return result


def validate_regions(result):
    labels = result['labels']
    k = result['stats']['final_regions']
    require(labels.dtype==np.int32 and np.array_equal(np.unique(labels),np.arange(k)),
            'Region labels must be contiguous nonnegative int32.')
    require(int(result['areas'].sum())==labels.size and np.all(result['areas']>0),
            'Region areas do not cover the canvas exactly.')
    _,count = components4(labels)
    require(count==k,'A final region is disconnected.')
    require(k==1 or int(result['areas'].min())>=result['stats']['minimum_area'],
            'Small-component cleanup left a tiny region.')


def technical_check(device):
    y,x = torch.meshgrid(torch.arange(32,device=device),torch.arange(32,device=device),indexing='ij')
    lab = torch.stack((30+x.float()/3, y.float()/4, torch.zeros_like(x,dtype=torch.float32)))
    boundary = torch.zeros((32,32),device=device); boundary[:,13] = 1
    original = lab.clone()
    settings = dict(n_segments=16,iterations=3,return_raw=True)
    plain = segment_superpixels(lab,None,**settings)
    disabled = segment_superpixels(lab,boundary,boundary_weight=0,**settings)
    zeros = segment_superpixels(lab,torch.zeros_like(boundary),boundary_weight=2,**settings)
    guided = segment_superpixels(lab,boundary,boundary_weight=2,**settings)
    repeat = segment_superpixels(lab,boundary,boundary_weight=2,**settings)
    for key in ('labels','raw_labels'):
        require(np.array_equal(plain[key],disabled[key]) and np.array_equal(plain[key],zeros[key]),
                'Boundary off / zero-map equivalence failed.')
        require(np.array_equal(guided[key],repeat[key]),'Fixed-backend replay failed.')
    changed = int(np.count_nonzero(guided['raw_labels'] != plain['raw_labels']))
    require(changed > 0, 'Synthetic barrier did not change any assignment.')
    require(torch.equal(lab,original),'SLIC changed the Lab input.')
    for result in (plain,disabled,zeros,guided,repeat): validate_regions(result)
    return {'zero_weight_equivalence':True,'zero_map_equivalence':True,
            'synthetic_fixed_backend_replay':True,'synthetic_connectivity_and_coverage':True,
            'synthetic_barrier_changed_assignments':changed}


class Progress:
    def __init__(self,total=80):
        self.total,self.done,self.stage,self.images = total,0,'technical checks',0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.refresh,daemon=True)
    def display(self):
        elapsed = time.perf_counter()-STARTED
        eta = f'{elapsed/self.done*(self.total-self.done):.1f}s' if self.done else '--'
        print(f'Superpixel iterations {self.done}/{self.total} | Images {self.images}/4 | '
              f'Elapsed {elapsed:.1f}s | ETA {eta} | {self.stage}',flush=True)
    def refresh(self):
        while not self.stop.wait(1 if sys.stdout.isatty() else 5): self.display()
    def iteration(self,i,total):
        self.done += 1
        self.stage = self.current + f' | iteration {i}/{total}'
    def __enter__(self):
        self.display(); self.thread.start(); return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join(); self.display()


def overlay(rgb,labels):
    edges = np.zeros(labels.shape,bool)
    edges[:,1:] |= labels[:,1:] != labels[:,:-1]
    edges[1:,:] |= labels[1:,:] != labels[:-1,:]
    image = rgb.copy()
    image[edges] = [0,255,255]
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args()
    output = None
    try:
        import dataset_v1 as d
        require(d.VERSION=='input_pipeline_v1' and bm.VERSION=='boundary_map_v1',
                'Expected the existing dataset_v1.py and boundary_map_v1.py.')
        require(args.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable.')
        device = torch.device('cuda' if args.device=='cuda' or
                              (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
        torch.set_num_threads(min(4,torch.get_num_threads()))
        output = args.root.resolve()/'environment/data_audit'/VERSION/(
            'run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        output.mkdir(parents=True,exist_ok=False)
        evidence = []
        page = Image.new('RGB',(d.SIZE*3,(d.SIZE+48)*4),'white')
        draw = ImageDraw.Draw(page)
        with Progress() as progress:
            project = d.load_project(args.root)
            train = d.AptosDataset(project,'train',verify_file_hash=True)
            batch = next(iter(d.make_loader(train,epoch=0,batch_size=4)))
            d.validate_batch(batch,{r['id_code']:r for r in project['rows']['train']})
            rgb_before,cnn_before = batch['rgb'].clone(),batch['cnn'].clone()
            checks = technical_check(device)
            progress.stage = 'technical equivalence and connectivity checks passed'
            for i,image_id in enumerate(batch['id_code']):
                rgb = batch['rgb'][i].to(device)
                prepared = bm.boundary_map(rgb,return_details=True)
                lab_before,boundary_before = prepared['lab'].clone(),prepared['boundary'].clone()
                array = (batch['rgb'][i].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                y = i*(d.SIZE+48)
                draw.text((8,y+6),f'TRAIN {image_id} | grade {int(batch["label"][i])}',fill='black')
                draw.text((8,y+25),'Shared augmented RGB | epoch 0',fill='black')
                page.paste(Image.fromarray(array),(0,y+48))
                for col,(method,gamma) in enumerate([('slic',0.0),('boundary_slic',2.0)],start=1):
                    progress.current = f'image {i+1}/4 {image_id} | {method}'
                    result = segment_superpixels(prepared['lab'],prepared['boundary'][0],
                               boundary_weight=gamma,progress=progress.iteration)
                    validate_regions(result)
                    filename = f'{image_id}_{method}.npz'
                    path = output/filename
                    np.savez_compressed(path,labels=result['labels'],areas=result['areas'],
                                        centroids_yx=result['centroids_yx'],mean_lab=result['mean_lab'])
                    with np.load(path,allow_pickle=False) as saved:
                        require(np.array_equal(saved['labels'],result['labels']),'Saved labels differ.')
                    with (output/f'{image_id}_{method}_regions.csv').open('w',newline='',encoding='utf-8') as f:
                        writer = csv.writer(f);writer.writerow(['region_id','area','centroid_y','centroid_x','L','a','b'])
                        for region in range(len(result['areas'])):
                            writer.writerow([region,int(result['areas'][region]),
                                             *result['centroids_yx'][region],*result['mean_lab'][region]])
                    stats = result['stats']
                    evidence.append({'id_code':image_id,'diagnosis':int(batch['label'][i]),
                                     'split':'train','epoch':0,'method':method,**stats,
                                     'output_file':filename,'output_sha256':d.sha256(path),
                                     'input_png_sha256':project['processed'][image_id]['output_sha256']})
                    draw.text((col*d.SIZE+8,y+6),f'{method} | gamma={gamma:g} | regions={stats["final_regions"]}',fill='black')
                    draw.text((col*d.SIZE+8,y+25),'Same RGB, seeds, m=10, iterations=10',fill='black')
                    page.paste(overlay(array,result['labels']),(col*d.SIZE,y+48))
                    print(f'{image_id} | {method} | regions={stats["final_regions"]} | '
                          f'{stats["elapsed_seconds"]:.2f}s | uncovered={sum(stats["uncovered_pixels_by_iteration"])}',flush=True)
                require(torch.equal(prepared['lab'],lab_before) and torch.equal(prepared['boundary'],boundary_before),
                        'Superpixel extraction changed Lab or boundary input.')
                progress.images = i+1
            require(torch.equal(batch['rgb'],rgb_before) and torch.equal(batch['cnn'],cnn_before),
                    'RGB/CNN inputs changed.')
            page.save(output/'superpixels_preview_v1.jpg',quality=95)
            d.write_json(output/'superpixels_config_v1.json',{
                'version':VERSION,'settings':SETTINGS,'sources':SOURCES,'seed':d.SEED,'epoch':0,
                'input_pipeline_settings':d.SETTINGS,'boundary_settings':bm.SETTINGS,
                'code_sha256':{Path(m.__file__).name:d.sha256(Path(m.__file__)) for m in (d,bm)},
                'superpixels_sha256':d.sha256(Path(__file__)),
                'source_metadata_sha256':project['source_sha256'],
                'versions':{'python':platform.python_version(),'torch':torch.__version__,
                            'torchvision':d.torchvision.__version__,'numpy':np.__version__,
                            'pillow':d.PIL.__version__},
                'device':str(device),'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None})
            d.write_json(output/'superpixels_summary_v1.json',{
                'status':'TECHNICAL_PASS_AWAITING_SUPERPIXEL_PREVIEW_REVIEW',
                'scope':'four training samples, not performance validation or full-dataset processing',
                'train_images_read':4,'validation_images_read':0,'test_images_read':0,
                'split_counts':{name:len(rows) for name,rows in project['rows'].items()},
                'checks':checks,'rgb_cnn_unchanged':True,'all_pixels_assigned':True,
                'all_regions_4_connected':True,'device':str(device),'results':evidence,
                'elapsed_seconds':time.perf_counter()-STARTED})
            progress.stage = '8/8 segmentations checked and saved; preview ready'
        print('\nSUPERPIXEL CHECK: PASS (small-batch technical checks only)')
        print('Boundary off / zero-map equivalence: PASS | Connected regions: PASS')
        print('Train images: 4 | Validation: 0 | Test: 0 | RGB/CNN unchanged: PASS')
        print('SUPERPIXEL PREVIEW: AWAITING VISUAL REVIEW')
        print(f'Total elapsed: {time.perf_counter()-STARTED:.1f}s\nSaved: {output}')
        print('Return this summary and superpixels_preview_v1.jpg.')
        print('After review passes: remember to commit superpixels_v1.py.')
        return 0
    except Exception as exc:
        print(f'\nSUPERPIXEL CHECK: FAIL | {type(exc).__name__}: {exc}',flush=True)
        if output is not None:
            (output/'superpixels_failure_v1.json').write_text(json.dumps(
                {'status':'FAIL','error':f'{type(exc).__name__}: {exc}'},indent=2)+'\n',encoding='utf-8')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
