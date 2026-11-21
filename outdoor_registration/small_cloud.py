"""Boundary-safe extensions; normal-size inputs delegate to upstream behavior.

No extra trainable parameters. Missing patch entries are masked, never counted
as repeated observations. Single-node geometry has no angular information.
"""
import torch
from torch import nn
from geotransformer.modules.ops import point_to_node_partition as original_partition,pairwise_distance
from geotransformer.modules.geotransformer.geotransformer import GeometricStructureEmbedding,GeometricTransformer
from geotransformer.modules.transformer import RPEConditionalTransformer
from geotransformer.modules.loss import WeightedCircleLoss


@torch.no_grad()
def point_to_node_partition(points,nodes,point_limit,return_count=False):
    if not len(points) or not len(nodes):raise ValueError('Empty cloud cannot be partitioned')
    k=min(point_limit,len(points))
    result=list(original_partition(points,nodes,k,return_count=return_count))
    if k<point_limit:
        ids,masks=result[-2:]
        result[-2]=torch.cat([ids,ids.new_full((len(nodes),point_limit-k),len(points))],dim=1)
        result[-1]=torch.cat([masks,masks.new_zeros((len(nodes),point_limit-k))],dim=1)
    return tuple(result)


class SmallCloudStructureEmbedding(GeometricStructureEmbedding):
    @torch.no_grad()
    def get_embedding_indices(self,points):
        b,n,_=points.shape
        if n>self.angle_k:return super().get_embedding_indices(points)
        distance=torch.sqrt(pairwise_distance(points,points))
        if n<2:return distance/self.sigma_d,points.new_zeros((b,n,n,1))
        k=n-1
        ids=distance.topk(k=n,dim=2,largest=False)[1][:,:,1:]
        neighbors=torch.gather(points[:,None,:,:].expand(b,n,n,3),2,ids[:,:,:,None].expand(b,n,k,3))
        reference=(neighbors-points[:,:,None,:])[:,:,None,:,:].expand(b,n,n,k,3)
        anchors=(points[:,None,:,:]-points[:,:,None,:])[:,:,:,None,:].expand(b,n,n,k,3)
        sine=torch.linalg.norm(torch.cross(reference,anchors,dim=-1),dim=-1)
        cosine=(reference*anchors).sum(dim=-1)
        return distance/self.sigma_d,torch.atan2(sine,cosine)*self.factor_a

    def forward(self,points):
        if points.shape[1]==0:raise ValueError('Empty coarse cloud')
        if points.shape[1]==1:
            # No artificial zero-angle observation when no neighbor exists.
            d=torch.sqrt(pairwise_distance(points,points))/self.sigma_d
            return self.proj_d(self.embedding(d))
        return super().forward(points)


class SmallCloudGeoTransformer(GeometricTransformer):
    def __init__(self,input_dim,output_dim,hidden_dim,num_heads,blocks,sigma_d,sigma_a,angle_k,
                 dropout=None,activation_fn='ReLU',reduction_a='max'):
        # Same module names, parameter dimensions and initialization order as upstream.
        nn.Module.__init__(self)
        self.embedding=SmallCloudStructureEmbedding(hidden_dim,sigma_d,sigma_a,angle_k,reduction_a=reduction_a)
        self.in_proj=nn.Linear(input_dim,hidden_dim)
        self.transformer=RPEConditionalTransformer(blocks,hidden_dim,num_heads,dropout=dropout,activation_fn=activation_fn)
        self.out_proj=nn.Linear(hidden_dim,output_dim)


class SmallCloudCircleLoss(WeightedCircleLoss):
    def forward(self,pos_masks,neg_masks,feat_dists,pos_scales=None,neg_scales=None):
        rows=(pos_masks.any(-1)&neg_masks.any(-1)).detach()
        cols=(pos_masks.any(-2)&neg_masks.any(-2)).detach()
        if rows.any() and cols.any():
            return super().forward(pos_masks,neg_masks,feat_dists,pos_scales,neg_scales)
        if not rows.any() and not cols.any():return feat_dists.sum()*0
        # Only directions with both positive and negative anchors are defined.
        pos=(feat_dists-1e5*(~pos_masks).float()-self.pos_optimal).clamp_min(0)
        neg=(self.neg_optimal-feat_dists-1e5*(~neg_masks).float()).clamp_min(0)
        if pos_scales is not None:pos=pos*pos_scales
        if neg_scales is not None:neg=neg*neg_scales
        p=self.log_scale*(feat_dists-self.pos_margin)*pos.detach()
        n=self.log_scale*(self.neg_margin-feat_dists)*neg.detach()
        losses=[]
        if rows.any():losses.append((torch.nn.functional.softplus(p.logsumexp(-1)+n.logsumexp(-1))/self.log_scale)[rows].mean())
        if cols.any():losses.append((torch.nn.functional.softplus(p.logsumexp(-2)+n.logsumexp(-2))/self.log_scale)[cols].mean())
        return torch.stack(losses).mean()
