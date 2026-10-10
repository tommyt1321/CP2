r"""CP2 region adjacency only: one node per superpixel, shared-border edges.

Run: python region_adjacency_v1.py
Reads the latest regional_features_v1 run's eight label/token NPZs (4 TRAIN
images x 2 modes). No image decoding, CNN execution, downloads or GAT training.
To select an existing run explicitly: --features-run "C:\CP2\environment\..."

Reusable training API (use CURRENT augmented-image labels, never old previews):
    graphs = [build_region_graph(labels) for labels in current_label_maps]
    packed = pack_graph_batch(regional_result, graphs)
    # packed['x']: [sum(R),256], preserves autograd from regional_result['tokens'].
    # edge_index [2,2*E]: source, target; no links between different images.
    # After a future GAT: unpack_graph_features(updated_x, packed['ptr']).

Graph rule: four-neighbour pixel contacts only; touching at a corner is not an
edge. Each unordered pair is stored once, and twice in edge_index for directed
message passing. No self-loops in the physical graph. GAT self-connections,
attention, trainable weights, and feature updates are NOT implemented here.
Boundary lengths are evidence metadata, not attention or edge weights.
All canvas regions, including existing padding, remain nodes in both variants.
"""
import argparse
from collections import Counter
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
    print('Loading region-adjacency libraries...', flush=True)
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

VERSION = 'region_adjacency_v1'
SETTINGS = {
    'nodes': 'one per original contiguous superpixel ID; token row r is node r',
    'adjacency': 'shared horizontal/vertical pixel side; 4-neighbour contacts',
    'corner_only_contact': False, 'self_loops_in_physical_graph': False,
    'undirected_edges': 'unique pairs (u,v), u<v, lexicographically sorted',
    'edge_index': 'source,target; both directions; sorted by source then target; int64',
    'shared_boundary_length': 'count of unit pixel sides, saved as metadata only',
    'edge_weighting': 'none; no use of length, CNN similarity, distance or boundary strength',
    'batching': 'disjoint union of real nodes; remove padding slots, offset node IDs per image',
    'canvas': 'all regions including existing black/grey background; no FOV filtering',
    'gat': 'not implemented in this module',
}
SOURCES = {
    'proposal': 'CP1 section 3.6.2: one node per superpixel; connect physically shared boundaries',
    'adjacency': 'https://papers.miccai.org/miccai-2024/paper/3756_paper.pdf',
    'citation': 'Playout et al. (2024), A Region-Based Approach to Diabetic Retinopathy '
                'Classification with Superpixel Tokenization, MICCAI, section 2.3',
    'scope': 'Horizontal/vertical region contacts are supported by section 2.3. '
             'This implementation uses array slices, not the paper convolution/scatter code. '
             'It does not adopt diagonal area entries, adjacency normalisation, prototypes, '
             'Leiden pooling, or black-border merging.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def as_labels(labels):
    if isinstance(labels,torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = np.asarray(labels)
    require(labels.ndim==2 and labels.size>0 and np.issubdtype(labels.dtype,np.integer),
            'Expected a nonempty integer HxW label map.')
    ids = np.unique(labels)
    require(np.array_equal(ids,np.arange(len(ids))), 'Label IDs must be contiguous from zero.')
    return labels.astype(np.int64,copy=False),len(ids)


def build_region_graph(labels):
    """Topology uses label contacts only, not RGB, features, diagnoses or centroids."""
    labels,n = as_labels(labels)
    left = np.concatenate((labels[:,:-1].ravel(),labels[:-1,:].ravel()))
    right = np.concatenate((labels[:,1:].ravel(),labels[1:,:].ravel()))
    different = left!=right
    pairs = np.stack((np.minimum(left[different],right[different]),
                      np.maximum(left[different],right[different])),axis=1)
    pairs,lengths = np.unique(pairs,axis=0,return_counts=True)
    directed = np.concatenate((pairs,pairs[:,::-1]),axis=0)
    order = np.lexsort((directed[:,1],directed[:,0]))
    edge_index = directed[order].T.copy()
    degree = np.bincount(edge_index[0],minlength=n)
    return {'num_nodes':n,'undirected_edges':pairs,'boundary_lengths':lengths.astype(np.int64),
            'edge_index':edge_index,'directed_boundary_lengths':np.tile(lengths,2)[order].astype(np.int64),
            'degree':degree,'areas':np.bincount(labels.ravel(),minlength=n)}


def validate_graph(graph):
    n = graph['num_nodes']; pairs = graph['undirected_edges']; edge = graph['edge_index']
    require(n>0 and pairs.ndim==2 and pairs.shape[1]==2 and edge.shape==(2,2*len(pairs)),
            'Invalid graph shape.')
    require(pairs.dtype==np.int64 and edge.dtype==np.int64, 'Graph indices must be int64.')
    require(np.all(pairs[:,0]<pairs[:,1]) and np.all(pairs>=0) and np.all(pairs<n),
            'Self-loop or invalid node ID in physical graph.')
    require(len(np.unique(pairs,axis=0))==len(pairs),'Duplicate unordered edges.')
    expected = {(int(u),int(v)) for u,v in pairs} | {(int(v),int(u)) for u,v in pairs}
    require(set(map(tuple,edge.T.tolist()))==expected, 'Directed edges are not symmetric/exact.')
    lengths = graph['boundary_lengths']; directed_lengths = graph['directed_boundary_lengths']
    require(lengths.shape==(len(pairs),) and np.all(lengths>0), 'Invalid boundary lengths.')
    lookup = {tuple(p):int(length) for p,length in zip(pairs,lengths)}
    require(directed_lengths.shape==(edge.shape[1],) and
            all(lookup[tuple(sorted((int(u),int(v))))]==int(length)
                for (u,v),length in zip(edge.T,directed_lengths)), 'Directed boundary metadata mismatch.')
    neighbours = [[] for _ in range(n)]
    for u,v in pairs:
        neighbours[u].append(int(v));neighbours[v].append(int(u))
    visited={0};todo=[0]
    while todo:
        for v in neighbours[todo.pop()]:
            if v not in visited:
                visited.add(v);todo.append(v)
    require(len(visited)==n,'Full-canvas region graph is disconnected.')
    require(np.array_equal(graph['degree'],np.array([len(x) for x in neighbours])), 'Degree mismatch.')
    return {'nodes':n,'undirected_edges':len(pairs),'directed_edges':edge.shape[1],
            'degree_min':int(graph['degree'].min()),'degree_max':int(graph['degree'].max()),
            'shared_pixel_sides':int(lengths.sum()),'connected':True}


def pack_graph_batch(regional, graphs):
    """Convert padded regional tensors to a disjoint graph batch without detaching x.

    Input is the dict returned by regional_features_v1. Real IDs stay in region
    order within each image. No padded token can become a graph node.
    """
    tokens = regional['tokens']; counts = regional['region_counts']
    mask = regional['key_padding_mask']; geometry = regional['geometry']; areas = regional['areas']
    require(tokens.ndim==3 and len(graphs)==tokens.shape[0]>0,'Expected one graph per image.')
    B,M,C = tokens.shape
    require(counts.dtype==torch.long and counts.shape==(B,), 'Invalid region counts.')
    require(mask.dtype==torch.bool and mask.shape==(B,M) and geometry.shape==(B,M,3)
            and areas.shape==(B,M),'Regional metadata shape mismatch.')
    require(all(t.device==tokens.device for t in (counts,mask,geometry,areas)), 'Mixed tensor devices.')
    sizes = counts.detach().cpu().tolist()
    require(all(0<n<=M for n in sizes), 'Invalid valid-token count.')
    expected_mask = torch.arange(M,device=tokens.device)[None,:]>=counts[:,None]
    require(torch.equal(mask,expected_mask), 'Padding mask is not a valid prefix mask.')
    offsets = np.concatenate(([0],np.cumsum(sizes,dtype=np.int64)))
    edges=[];lengths=[]
    for i,(graph,n) in enumerate(zip(graphs,sizes)):
        require(graph['num_nodes']==n, 'Graph nodes do not match token rows.')
        require(np.array_equal(areas[i,:n].detach().cpu().numpy(),graph['areas']),
                'Graph labels and token region areas differ.')
        edges.append(graph['edge_index']+offsets[i])
        lengths.append(graph['directed_boundary_lengths'])
    device=tokens.device
    return {'x':torch.cat([tokens[i,:n] for i,n in enumerate(sizes)]),
            'edge_index':torch.from_numpy(np.concatenate(edges,axis=1)).to(device),
            'boundary_lengths':torch.from_numpy(np.concatenate(lengths)).to(device),
            'ptr':torch.from_numpy(offsets).to(device),
            'batch':torch.repeat_interleave(torch.arange(B,device=device),counts),
            'local_region_id':torch.cat([torch.arange(n,device=device) for n in sizes]),
            'geometry':torch.cat([geometry[i,:n] for i,n in enumerate(sizes)]),
            'areas':torch.cat([areas[i,:n] for i,n in enumerate(sizes)])}


def unpack_graph_features(x,ptr):
    """Return variable graph features to padded tokens for a later Transformer."""
    require(x.ndim==2 and ptr.ndim==1 and ptr.dtype==torch.long and len(ptr)>=2,
            'Expected node features [N,C] and int64 graph pointers.')
    points=ptr.detach().cpu().tolist()
    require(points[0]==0 and points[-1]==len(x) and all(a<b for a,b in zip(points,points[1:])),
            'Graph pointers do not partition the nodes.')
    pieces=[x[a:b] for a,b in zip(points,points[1:])]
    tokens=pad_sequence(pieces,batch_first=True)
    counts=torch.tensor([len(p) for p in pieces],dtype=torch.long,device=x.device)
    mask=torch.arange(tokens.shape[1],device=x.device)[None,:]>=counts[:,None]
    return {'tokens':tokens,'key_padding_mask':mask,'region_counts':counts}


def scalar_reference(labels):
    """Independent check: visit each pixel and its right/down neighbours once."""
    contacts=Counter();H,W=labels.shape
    for y in range(H):
        for x in range(W):
            a=int(labels[y,x])
            for yy,xx in ((y,x+1),(y+1,x)):
                if yy<H and xx<W:
                    b=int(labels[yy,xx])
                    if a!=b:contacts[tuple(sorted((a,b)))]+=1
    return contacts


def check_batch(regional,graphs):
    packed=pack_graph_batch(regional,graphs)
    edge=packed['edge_index']
    require(torch.equal(packed['batch'][edge[0]],packed['batch'][edge[1]]),
            'A graph edge crosses between different images.')
    recovered=unpack_graph_features(packed['x'],packed['ptr'])
    require(torch.equal(recovered['tokens'],regional['tokens']) and
            torch.equal(recovered['key_padding_mask'],regional['key_padding_mask']),
            'Pack/unpack changed valid features or zero padding.')
    return packed


def technical_check(device):
    q=np.array([[0,0,1,1],[0,0,1,1],[2,2,3,3],[2,2,3,3]],np.int32)
    graph=build_region_graph(q)
    assert_expected={(0,1):2,(0,2):2,(1,3):2,(2,3):2}
    actual={tuple(p):int(v) for p,v in zip(graph['undirected_edges'],graph['boundary_lengths'])}
    require(actual==assert_expected,'Four-quadrant graph includes diagonal contact or wrong boundary length.')
    single=build_region_graph(np.zeros((3,5),np.int32))
    require(single['edge_index'].shape==(2,0),'Single-region graph should have no physical edges.')
    strip=np.array([[0,1,2],[0,1,2]],np.int32)
    for labels in (q,strip,np.zeros((3,5),np.int32)):
        g=build_region_graph(labels);validate_graph(g)
        require(dict(zip(map(tuple,g['undirected_edges']),g['boundary_lengths']))==scalar_reference(labels),
                'Scalar contact reference mismatch.')
    # Two graphs of sizes 4 and 1. Padding must have no node and receive zero gradient.
    tokens=torch.arange(24,device=device,dtype=torch.float32).reshape(2,4,3)
    tokens[1,1:]=0;tokens.requires_grad_()
    areas=torch.zeros((2,4),device=device,dtype=torch.long)
    areas[0]=torch.from_numpy(graph['areas']).to(device);areas[1,0]=15
    regional={'tokens':tokens,'region_counts':torch.tensor([4,1],device=device),
              'key_padding_mask':torch.tensor([[False]*4,[False,True,True,True]],device=device),
              'geometry':torch.zeros((2,4,3),device=device),'areas':areas}
    packed=check_batch(regional,[graph,single])
    unpack_graph_features(packed['x'],packed['ptr'])['tokens'].sum().backward()
    require(torch.equal(tokens.grad,(~regional['key_padding_mask'])[:,:,None].expand_as(tokens).float()),
            'Packing detached features or allowed gradient to padding.')
    return {'four_neighbour_rule':True,'corner_only_pairs_excluded':True,'single_node_supported':True,
            'scalar_reference':True,'disjoint_batch_roundtrip':True,'packing_gradients':True,
            'padding_not_nodes':True,'test_device':str(device)}


class Progress:
    def __init__(self):
        self.done,self.graphs,self.stage=0,0,'starting'
        self.stop=threading.Event();self.thread=threading.Thread(target=self.refresh,daemon=True)
    def show(self):
        elapsed=time.perf_counter()-STARTED
        eta=f'{elapsed/self.done*(12-self.done):.1f}s' if self.done else '--'
        print(f'Graph checks {self.done}/12 | Graphs {self.graphs}/8 | Elapsed {elapsed:.1f}s | '
              f'ETA {eta} | {self.stage}',flush=True)
    def refresh(self):
        while not self.stop.wait(1 if sys.stdout.isatty() else 5):self.show()
    def advance(self,stage):
        self.done+=1;self.stage=stage
    def __enter__(self):
        self.show();self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.show()


def resolve_run(root,chosen):
    if chosen is not None:
        path=chosen if chosen.is_absolute() else root/chosen
    else:
        runs=sorted((p for p in (root/'environment/data_audit/regional_features_v1').glob('run_*')
                     if p.is_dir()),reverse=True)
        require(bool(runs),'No regional_features_v1 run found; run that module first.')
        path=runs[0]
    require((path/'regional_features_summary_v1.json').is_file(),
            'Selected/latest regional-feature run is incomplete. Use --features-run to select the accepted run.')
    return path.resolve()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--features-run',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto',help='batch tensor checks only')
    args=parser.parse_args();output=None
    try:
        import dataset_v1 as d
        require(d.VERSION=='input_pipeline_v1','Expected existing dataset_v1.py.')
        require(args.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable.')
        device=torch.device('cuda' if args.device=='cuda' or
                            (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
        torch.set_num_threads(min(4,torch.get_num_threads()))
        root=args.root.resolve();source=resolve_run(root,args.features_run)
        output=root/'environment/data_audit'/VERSION/(
            'run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:6])
        output.mkdir(parents=True,exist_ok=False)
        methods=('slic','boundary_slic');groups={m:[] for m in methods};results=[];input_hashes={}
        with Progress() as progress:
            project=d.load_project(root)
            summary_path=source/'regional_features_summary_v1.json'
            config_path=source/'regional_features_config_v1.json'
            summary=json.loads(summary_path.read_text());config=json.loads(config_path.read_text())
            require(summary.get('status')=='TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW',
                    'Upstream regional-feature run did not pass its technical checks.')
            require(config.get('version')=='regional_features_v1' and
                    config.get('source_metadata_sha256')==project['source_sha256'],
                    'Source run and current split/preprocessing metadata differ.')
            evidence=summary['results'];train={r['id_code']:r for r in project['rows']['train']}
            ids=list(dict.fromkeys(r['id_code'] for r in evidence))
            require(len(ids)==4 and len(evidence)==8 and
                    Counter((r['id_code'],r['method']) for r in evidence)==
                    Counter((image_id,m) for image_id in ids for m in methods),
                    'Expected four train images with both methods exactly once.')
            input_hashes={str(summary_path):d.sha256(summary_path),str(config_path):d.sha256(config_path)}
            progress.advance('accepted upstream run and fixed split metadata matched')
            checks=technical_check(device)
            progress.advance('known-contact, padding and differentiable batch checks passed')
            for row in evidence:
                image_id,method=row['id_code'],row['method']
                require(image_id in train and row['split']=='train' and row['epoch']==config['epoch']
                        and int(row['diagnosis'])==int(train[image_id]['diagnosis']), 'Invalid source sample metadata.')
                require(row['input_png_sha256']==project['processed'][image_id]['output_sha256'],
                        'Source RGB record differs from accepted preprocessing metadata.')
                filename=f'{image_id}_{method}_tokens.npz'
                require(row['output_file']==filename,'Unexpected upstream filename.')
                path=source/filename
                require(d.sha256(path)==row['output_sha256'],'Saved token/label file hash differs from upstream summary.')
                input_hashes[str(path)]=row['output_sha256']
                with np.load(path,allow_pickle=False) as data:
                    labels,tokens,areas,geometry=[data[k].copy() for k in ('labels','tokens','areas','geometry')]
                require(labels.shape==(384,384),'Expected original 384x384 label map.')
                graph=build_region_graph(labels);stats=validate_graph(graph);n=graph['num_nodes']
                require(tokens.dtype==np.float32 and tokens.shape==(n,256) and list(tokens.shape)==row['token_shape']
                        and np.isfinite(tokens).all(),'Token shape/dtype/values do not match graph nodes.')
                require(areas.dtype==np.int64 and np.array_equal(areas,graph['areas'])
                        and geometry.shape==(n,3) and np.isfinite(geometry).all(),'Region metadata mismatch.')
                yy,xx=np.indices(labels.shape);flat=labels.ravel()
                expected_geometry=np.stack(((np.bincount(flat,weights=yy.ravel())/areas+.5)/384,
                                            (np.bincount(flat,weights=xx.ravel())/areas+.5)/384,
                                            areas/(384*384)),axis=1)
                require(np.allclose(geometry,expected_geometry,atol=1e-7,rtol=0),'Region geometry/ID mismatch.')
                reference=scalar_reference(labels)
                require(dict(zip(map(tuple,graph['undirected_edges']),graph['boundary_lengths']))==reference,
                        'Graph differs from independent pixel contact counts.')
                graph_file=output/f'{image_id}_{method}_graph.npz'
                np.savez_compressed(graph_file,tokens=tokens,labels=labels,geometry=geometry,**graph)
                with np.load(graph_file,allow_pickle=False) as saved:
                    require(np.array_equal(saved['edge_index'],graph['edge_index']) and
                            np.array_equal(saved['tokens'],tokens),'Graph save/readback mismatch.')
                groups[method].append((graph,torch.from_numpy(tokens).to(device),
                                       torch.from_numpy(geometry).to(device),torch.from_numpy(areas).to(device)))
                results.append({'id_code':image_id,'method':method,'split':'train','epoch':row['epoch'],
                                **stats,'source_file':filename,'source_sha256':row['output_sha256'],
                                'output_file':graph_file.name,'output_sha256':d.sha256(graph_file)})
                print(f'{image_id} | {method} | nodes={n} | edges={stats["undirected_edges"]} '
                      f'(undirected) / {stats["directed_edges"]} (directed) | connected=PASS',flush=True)
                progress.graphs+=1;progress.advance(f'{progress.graphs}/8 graphs built and checked')
            batches={}
            for method,items in groups.items():
                graphs,xs,geos,ars=zip(*items)
                counts=torch.tensor([len(x) for x in xs],device=device,dtype=torch.long)
                tokens=pad_sequence(xs,batch_first=True)
                regional={'tokens':tokens,'geometry':pad_sequence(geos,batch_first=True),
                          'areas':pad_sequence(ars,batch_first=True),'region_counts':counts,
                          'key_padding_mask':torch.arange(tokens.shape[1],device=device)[None,:]>=counts[:,None]}
                packed=check_batch(regional,graphs)
                batches[method]={'images':len(items),'nodes':len(packed['x']),
                                 'directed_edges':packed['edge_index'].shape[1],'ptr':packed['ptr'].tolist(),
                                 'x_shape':list(packed['x'].shape),'cross_image_edges':0,'padding_nodes':0}
            checks.update(all_eight_scalar_contact_references=True,all_nodes_match_token_rows=True,
                          source_npz_hashes_verified=True,graph_roundtrip=True,real_disjoint_batches=True)
            progress.advance('both graph batches preserve tokens and image isolation')
            require(all(d.sha256(Path(p))==h for p,h in input_hashes.items()),'Upstream evidence changed during check.')
            checks['upstream_files_unchanged']=True
            d.write_json(output/'region_adjacency_config_v1.json',{
                'version':VERSION,'settings':SETTINGS,'sources':SOURCES,'upstream_run':str(source),
                'source_files_sha256':input_hashes,'source_metadata_sha256':project['source_sha256'],
                'code_sha256':{Path(__file__).name:d.sha256(Path(__file__)),Path(d.__file__).name:d.sha256(Path(d.__file__))},
                'upstream_regional_config':config,'graph_construction_device':'cpu','batch_check_device':str(device),
                'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}})
            d.write_json(output/'region_adjacency_summary_v1.json',{
                'status':'TECHNICAL_PASS_AWAITING_USER_SUMMARY_REVIEW','checks':checks,'results':results,'batches':batches,
                'train_samples_reused':4,'graphs_built':8,'image_files_read':0,'validation_test_samples_used':0,
                'cnn_executed':False,'gat_implemented':False,'optimizer_steps':0,
                'graph_construction_device':'cpu','batch_check_device':str(device),
                'elapsed_seconds':time.perf_counter()-STARTED,
                'limitation':'Four cached epoch-zero train samples only. No GAT or training performance validation. '
                             'Future training must build graphs from current labels and use live CNN tokens.'})
            progress.advance('graph files, configuration and summary saved')
        print('\nREGION ADJACENCY CHECK: PASS (small-batch technical checks only)')
        print(f'Graph construction: CPU | Batch tensor/gradient checks: {device}')
        print('Shared-border rule: PASS | Diagonal-only exclusion: PASS | Token/node alignment: PASS')
        for method,stats in batches.items():print(f'{method}: {stats}')
        print('Cross-image edges: 0 | Padding nodes: 0 | Feature roundtrip/packing gradients: PASS')
        print('Reused train samples: 4 | Graphs: 8 | Image files read: 0 | Validation/test samples used: 0')
        print('Upstream files unchanged: PASS | CNN rerun: No | GAT: not implemented')
        print(f'Total elapsed: {time.perf_counter()-STARTED:.1f}s\nSaved: {output}')
        print('Return this summary. No new image preview is required.')
        print('After review passes: remember to commit region_adjacency_v1.py.')
        return 0
    except Exception as exc:
        print(f'\nREGION ADJACENCY CHECK: FAIL | {type(exc).__name__}: {exc}',flush=True)
        if output is not None:
            (output/'region_adjacency_failure_v1.json').write_text(json.dumps(
                {'status':'FAIL','error':f'{type(exc).__name__}: {exc}'},indent=2)+'\n',encoding='utf-8')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
