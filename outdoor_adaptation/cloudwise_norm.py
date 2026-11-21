"""Experimental per-cloud GroupNorm with unchanged learned affine parameters.

Stacked ref/src neighborhoods stay disjoint. This option also keeps their
normalization statistics separate, without any sensor/domain labels.
"""
import torch
from torch import nn
from geotransformer.modules.kpconv.modules import GroupNorm


class CloudwiseGroupNorm(nn.Module):
    def __init__(self,original,context):
        super().__init__();self.norm=original.norm;self.context=context

    def forward(self,x):
        lengths=self.context['by_total'][len(x)]
        pieces=[];start=0
        for n in lengths:
            assert n>0
            y=x[start:start+n].T.unsqueeze(0)
            pieces.append(self.norm(y).squeeze(0).T);start+=n
        assert start==len(x)
        return torch.cat(pieces,dim=0)


def install_cloudwise_norm(model):
    assert not getattr(model,'cloudwise_norm_installed',False)
    context={};count=0
    def replace(module):
        nonlocal count
        for name,child in list(module.named_children()):
            if isinstance(child,GroupNorm):
                setattr(module,name,CloudwiseGroupNorm(child,context));count+=1
            else:replace(child)
    keys=set(model.state_dict());replace(model.backbone)
    assert set(model.state_dict())==keys and count>0
    def prepare(module,args):
        data=args[1];mapping={}
        for lengths in data['lengths']:
            values=tuple(int(x) for x in lengths);total=sum(values)
            assert total not in mapping or mapping[total]==values,'Ambiguous stage length mapping'
            mapping[total]=values
        context['by_total']=mapping
    model.backbone.register_forward_pre_hook(prepare)
    model.cloudwise_norm_installed=True
    return count
