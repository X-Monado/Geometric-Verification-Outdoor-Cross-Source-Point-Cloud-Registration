"""Bounded deterministic data, label-correct augmentation, native graph operators."""
import numpy as np
import torch
import os
from pathlib import Path
from scipy.spatial.transform import Rotation
from . import bootstrap
from .data import Cross3DReg,collate as native_collate


def collate(sample,cfg,limits):
    data=native_collate(sample,cfg,limits)
    # Native capped neighbors are views into larger uncapped storage. Compact
    # their storage before IPC/pinning without changing a single index.
    for key in ('neighbors','subsampling','upsampling'):
        data[key]=[tensor.contiguous() for tensor in data[key]]
    return data


def augment(sample,seed,noise=.01):
    """Independent rigid coordinate changes + shared scale; exact updated SE(3).

    p_ref'=a R_ref p_ref+b_ref, p_src'=a R_src p_src+b_src.
    R'=R_ref R R_src^T, t'=a R_ref t+b_ref-R' b_src.
    Small independent positional noise is added after this exact change.
    """
    rng=np.random.RandomState(seed)
    result=dict(sample)
    r_ref=Rotation.random(random_state=rng).as_matrix()
    r_src=Rotation.random(random_state=rng).as_matrix()
    scale=rng.uniform(.8,1.2)
    b_ref=rng.uniform(-2.,2.,3);b_src=rng.uniform(-2.,2.,3)
    old=sample['transform'].astype(np.float64)
    transform=np.eye(4)
    transform[:3,:3]=r_ref@old[:3,:3]@r_src.T
    transform[:3,3]=scale*r_ref@old[:3,3]+b_ref-transform[:3,:3]@b_src
    for side,rotation,shift in [('ref',r_ref,b_ref),('src',r_src,b_src)]:
        points=scale*sample[side+'_points']@rotation.T+shift
        if noise:points+=rng.uniform(-noise/2,noise/2,points.shape)
        result[side+'_points']=np.ascontiguousarray(points,dtype=np.float32)
    result['transform']=transform.astype(np.float32)
    result['augmentation_scale']=float(scale)
    return result


class BoundedPairs(Cross3DReg):
    def __getitem__(self,index):
        result=super().__getitem__(index)
        # Bounded FIFO raw/voxel cache per worker, never an all-dataset graph cache.
        while len(self.cache)>8:self.cache.pop(next(iter(self.cache)))
        return result


class TrainingSteps:
    def __init__(self,root,per_sequence,steps,seed,cfg,limits,augmentation,graph_cache_dir=None):
        self.pairs=BoundedPairs(root,'train',per_sequence,seed=7351)
        self.steps,self.seed,self.cfg,self.limits,self.augmentation=steps,seed,cfg,limits,augmentation
        self.graph_cache_dir=Path(graph_cache_dir) if graph_cache_dir and not augmentation else None
        if self.graph_cache_dir:self.graph_cache_dir.mkdir(parents=True,exist_ok=True)
        rng=np.random.RandomState(seed)
        self.order=[]
        while len(self.order)<steps:self.order.extend(rng.permutation(len(self.pairs)).tolist())

    def __len__(self):return self.steps

    def __getitem__(self,step):
        cache=self.graph_cache_dir/('%08d.pt'%self.pairs.indices[self.order[step]]) if self.graph_cache_dir else None
        hit=bool(cache and cache.exists())
        if hit:data=torch.load(cache,map_location='cpu',weights_only=True)
        else:
            sample=self.pairs[self.order[step]]
            if self.augmentation:sample=augment(sample,(self.seed+1000003*(step+1)) % 2**32)
            data=collate(sample,self.cfg,self.limits)
            if cache:
                temporary=cache.with_suffix('.%d.tmp'%os.getpid())
                torch.save(data,temporary);os.replace(temporary,cache)
        data['training_step']=step+1
        data['graph_cache_hit']=hit
        return data


class PreparedPairs:
    def __init__(self,pairs,indices,cfg,limits):
        self.pairs,self.indices,self.cfg,self.limits=pairs,indices,cfg,limits

    def __len__(self):return len(self.indices)

    def __getitem__(self,i):
        sample=self.pairs[self.indices[i]]
        return collate(sample,self.cfg,self.limits)


def single_item(item):return item


def worker_init(worker_id):
    torch.set_num_threads(1)
