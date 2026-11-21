"""Zero-initialized, point-dependent fine interaction; no GT in forward."""
import math
import torch
from torch import nn


class FineInteraction(nn.Module):
    def __init__(self,kind='geometry',hidden=64):
        super().__init__();self.kind=kind;self.hidden=hidden
        self.input=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,hidden),nn.GELU())
        if kind=='geometry':
            self.self_q=nn.Linear(hidden,hidden,bias=False);self.self_k=nn.Linear(hidden,hidden,bias=False)
            self.self_v=nn.Linear(hidden,hidden,bias=False);self.self_norm=nn.LayerNorm(hidden)
            self.cross_q=nn.Linear(hidden,hidden,bias=False);self.cross_k=nn.Linear(hidden,hidden,bias=False)
            self.cross_v=nn.Linear(hidden,hidden,bias=False);self.cross_norm=nn.LayerNorm(hidden)
            self.null=nn.Parameter(torch.zeros(1,1,hidden))
            self.register_buffer('centers',torch.tensor([0.,.6,1.2,2.4,4.8,9.6]))
            self.distance=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,1))
            self.output=nn.Linear(3*hidden,256)
        else:self.output=nn.Linear(hidden,256)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def self_context(self,x,points,mask):
        dist=(points[:,:,None]-points[:,None]).norm(dim=-1)
        rbf=torch.exp(-.5*((dist[:,:,:,None]-self.centers)/self.centers.clamp_min(.6)).square())
        scores=self.self_q(x)@self.self_k(x).transpose(1,2)/math.sqrt(self.hidden)
        scores=scores+self.distance(rbf).squeeze(-1)
        scores=scores.masked_fill(~mask[:,None],-1e4)
        return self.self_norm(x+scores.softmax(-1)@self.self_v(x))

    def cross_context(self,x,y,ymask,prior):
        memory=torch.cat([y,self.null.expand(len(y),-1,-1)],1)
        valid=torch.cat([ymask,torch.ones_like(ymask[:,:1])],1)
        scores=self.cross_q(x)@self.cross_k(memory).transpose(1,2)/math.sqrt(self.hidden)
        # Frozen descriptor affinity supplies a soft prior, never a hard anchor.
        scores=scores+torch.cat([prior,torch.zeros_like(prior[:,:,:1])],-1)
        return self.cross_norm(x+scores.masked_fill(~valid[:,None],-1e4).softmax(-1)@self.cross_v(memory))

    def forward(self,rf,sf,rp,sp,rm,sm):
        x,y=self.input(rf),self.input(sf)
        if self.kind=='geometry':
            a,b=self.self_context(x,rp,rm),self.self_context(y,sp,sm)
            prior=rf@sf.transpose(1,2)/16.
            c,d=self.cross_context(a,b,sm,prior),self.cross_context(b,a,rm,prior.transpose(1,2))
            dx,dy=self.output(torch.cat([x,a,c],-1)),self.output(torch.cat([y,b,d],-1))
        else:dx,dy=self.output(x),self.output(y)
        return rf+dx*rm[:,:,None],sf+dy*sm[:,:,None]
