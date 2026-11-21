"""Inference-only coarse pose verification and spatially conditioned fine matching.

Inputs use the original metric SE(3) convention. This module neither reads labels
nor trains the backbone. The verifier and overlap checkpoints are explicit inputs.
The research candidate is evaluated separately from the preserved B0 baseline.
"""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from outdoor_registration import bootstrap
from outdoor_registration.small_cloud import point_to_node_partition

def fit(a,b,w=None):
    if w is None:w=torch.ones(a.shape[:-1],device=a.device,dtype=a.dtype)
    w=w/w.sum(-1,keepdim=True).clamp_min(1e-8);ac=(a*w[...,None]).sum(-2);bc=(b*w[...,None]).sum(-2)
    h=(a-ac[...,None,:]).transpose(-1,-2)@((b-bc[...,None,:])*w[...,None])
    u,s,vh=torch.linalg.svd(h);v=vh.transpose(-1,-2);d=torch.ones_like(s);d[...,-1]=torch.linalg.det(v@u.transpose(-1,-2))
    r=(v*d[...,None,:])@u.transpose(-1,-2);t=bc-(r@ac[...,None]).squeeze(-1)
    result=torch.eye(4,device=a.device).expand(*r.shape[:-2],4,4).clone();result[...,:3,:3]=r;result[...,:3,3]=t
    return result


def hypotheses(ref,src,base,seed):
    if len(src)<3:return base[None],dict(draws=0,compatible=0,fitted=0)
    g=torch.Generator(device=src.device).manual_seed(seed);ids=torch.randint(len(src),(200000,3),generator=g,device=src.device)
    a,b=src[ids],ref[ids]
    da=torch.stack([(a[:,i]-a[:,j]).norm(dim=-1) for i,j in ((0,1),(0,2),(1,2))],1)
    db=torch.stack([(b[:,i]-b[:,j]).norm(dim=-1) for i,j in ((0,1),(0,2),(1,2))],1)
    valid=((da-db).abs()<1.2).all(1)&(da>2).all(1)&(db>2).all(1)
    valid &= torch.cross(a[:,1]-a[:,0],a[:,2]-a[:,0],dim=-1).norm(dim=-1)>1
    valid &= torch.cross(b[:,1]-b[:,0],b[:,2]-b[:,0],dim=-1).norm(dim=-1)>1
    keep=valid.nonzero(as_tuple=True)[0];count=len(keep)
    if count>4096:keep=keep[torch.linspace(0,count-1,4096,device=src.device).long()]
    pose=fit(a[keep],b[keep]) if len(keep) else base.new_empty((0,4,4))
    return torch.cat([base[None],pose]),dict(draws=200000,compatible=count,fitted=len(keep))


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def cloud_info(out,coarse_features,head,seed):
    rng=np.random.RandomState(seed);clouds={}
    for side in ('ref','src'):
        pc=out[side+'_points_c'].numpy().astype(float);pf=out[side+'_points_f'].numpy().astype(float)
        fc=coarse_features[side+'_features'].numpy();ff=out[side+'_feats_f'].numpy()
        with torch.no_grad():prob=head(coarse_features[side+'_features']).sigmoid().numpy()
        fp=prob[cKDTree(pc).query(pf)[1]]
        q=(fp+.05)/(fp+.05).sum();fine_ids=rng.choice(len(pf),min(128,len(pf)),replace=False,p=q)
        coarse_ids=np.argsort(-prob,kind='stable')[:64]
        clouds[side]=dict(fine=dict(points=pf,features=unit(ff),prob=fp,ids=fine_ids,tree=cKDTree(pf)),
                          coarse=dict(points=pc,features=unit(fc),prob=prob,ids=coarse_ids,tree=cKDTree(pc)))
    return clouds


def directional_features(rot,trans,query,memory,radii,k):
    ids=query['ids'];p=query['points'][ids];f=query['features'][ids];qp=query['prob'][ids]
    moved=p[None]@rot.transpose(0,2,1)+trans[:,None]
    dist,index=memory['tree'].query(moved.reshape(-1,3),k=min(k,len(memory['points'])))
    if dist.ndim==1:dist=dist[:,None];index=index[:,None]
    dist=dist.reshape(len(rot),len(ids),-1);index=index.reshape(len(rot),len(ids),-1)
    cosine=(memory['features'][index]*f[None,:,None]).sum(-1)
    mp=memory['prob'][index];weight=qp[None,:,None]*mp
    result=[]
    for radius in radii:
        near=dist[:,:,0]<radius;soft=np.exp(-.5*(dist/radius)**2);feat=np.maximum(cosine,0)*soft
        result += [near.mean(1),soft[:,:,0].mean(1),(cosine[:,:,0]*near).sum(1)/np.maximum(near.sum(1),1),
                   feat.max(2).mean(1),(feat*weight).sum((1,2))/np.maximum(weight.sum((1,2)),1e-8)]
    return np.column_stack(result)


@torch.no_grad()
def features_for_poses(out,coarse_features,head,poses,seed):
    # This function deliberately has no transform/GT-label argument.
    clouds=cloud_info(out,coarse_features,head,seed);matrix=poses.cpu().numpy().astype(float);chunks=[]
    ref=out['ref_corr_points'].numpy().astype(float);src=out['src_corr_points'].numpy().astype(float);confidence=out['corr_scores'].numpy().astype(float)
    for start in range(0,len(matrix),64):
        p=matrix[start:start+64];r,t=p[:,:3,:3],p[:,:3,3];inv_r=r.transpose(0,2,1);inv_t=-(inv_r@t[:,:,None])[:,:,0]
        columns=[]
        for level,radii,k in [('fine',(.3,.6,1.2,2.4),1),('coarse',(2.4,4.8,9.6),3)]:
            columns.append(directional_features(r,t,clouds['src'][level],clouds['ref'][level],radii,k))
            columns.append(directional_features(inv_r,inv_t,clouds['ref'][level],clouds['src'][level],radii,k))
        residual=np.linalg.norm(src[None]@r.transpose(0,2,1)+t[:,None]-ref[None],axis=-1)
        support=[]
        for radius in (.3,.6,1.2,2.4):
            good=residual<radius;support.extend([good.mean(1),(good*confidence[None]).sum(1)/max(confidence.sum(),1e-8)])
        columns.append(np.column_stack(support))
        context=[]
        for side in ('ref','src'):
            prob=clouds[side]['coarse']['prob'];context += [float(prob.mean()),float(prob.std())]
        context += [np.log(len(clouds['src']['fine']['points'])/len(clouds['ref']['fine']['points']))]
        columns.append(np.tile(context,(len(p),1)))
        chunks.append(np.column_stack(columns))
    result=np.concatenate(chunks).astype(np.float32);assert np.isfinite(result).all() and result.shape[1]==83
    return torch.from_numpy(result)


def spatial_bias(raw,d,poses,sigma):
    valid=d['rm'][:,:,None]&d['sm'][:,None];pieces=[]
    for start in range(0,len(raw),16):
        rp,sp=d['rp'][start:start+16],d['sp'][start:start+16]
        # Union of pose-conditioned neighborhoods; all poses selected without GT.
        moved=torch.einsum('hij,bsj->bhsi',poses[:,:3,:3],sp)+poses[None,:,None,:3,3]
        distance=(rp[:,None,:,None]-moved[:,:,None]).square().sum(-1)
        compat=torch.exp(-.5*distance/(sigma*sigma)).amax(1)
        mask=valid[start:start+16];n=mask.sum((1,2)).clamp_min(1)
        mean=(compat*mask).sum((1,2))/n
        std=(((compat-mean[:,None,None]).square()*mask).sum((1,2))/n).sqrt().clamp_min(1e-4)
        score=raw[start:start+16];sm=(score*mask).sum((1,2))/n
        scale=(((score-sm[:,None,None]).square()*mask).sum((1,2))/n).sqrt()
        bias=((compat-mean[:,None,None])/std[:,None,None]).clamp(-3,3)*scale[:,None,None]*mask
        pieces.append(bias)
    return torch.cat(pieces)


class OverlapHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,features):return self.net(features).squeeze(-1)


class VerificationHead(nn.Module):
    """Exact input subset and normalization restored from a trained verifier."""
    def __init__(self, state):
        super().__init__()
        for key in ('mean', 'std', 'indices'):
            self.register_buffer(key, state[key].clone())
        width = len(self.indices)
        if state['net.0.weight'].shape[0] == 1:
            self.net = nn.Sequential(nn.Linear(width, 1))
        else:
            self.net = nn.Sequential(nn.Linear(width, 64), nn.GELU(),
                nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.load_state_dict(state, strict=True)

    def forward(self, features):
        x = ((features - self.mean) / self.std).clamp(-6, 6)
        return self.net(x[:, self.indices]).squeeze(-1)


class PoseGuidedRefiner:
    """A frozen verifier proposes a pose, then native fine matching runs once.

    Sampling seeds control reproducibility only. They must not be chosen using
    evaluation labels. The reference/source points and features determine all
    proposals, verifier inputs, and fine matching scores.
    """
    def __init__(self, verifier_state, overlap_state, device='cuda', sigma_m=1.2,
                 feedback_weight=1.):
        if not np.isfinite(sigma_m) or sigma_m <= 0:
            raise ValueError('sigma_m must be positive')
        if not np.isfinite(feedback_weight) or feedback_weight < 0:
            raise ValueError('feedback_weight must be finite and nonnegative')
        self.verifier = VerificationHead(verifier_state).to(device).eval()
        self.overlap = OverlapHead().cpu().eval()
        self.overlap.load_state_dict(overlap_state, strict=True)
        self.sigma_m = float(sigma_m)
        self.feedback_weight = float(feedback_weight)

    @classmethod
    def from_checkpoints(cls, verifier_path, overlap_path, **kwargs):
        # These are the project's own trusted training checkpoints.
        verifier = torch.load(verifier_path, map_location='cpu', weights_only=False)
        overlap = torch.load(overlap_path, map_location='cpu', weights_only=False)
        return cls(verifier['model'], overlap['model'], **kwargs)

    @staticmethod
    def _patches(out):
        parts = []
        for side in ('ref', 'src'):
            p, f = out[side + '_points_f'], out[side + '_feats_f']
            _, _, index, mask = point_to_node_partition(p, out[side + '_points_c'], 128)
            ids = out[side + '_node_corr_indices']
            parts.append((torch.cat([f, torch.zeros_like(f[:1])])[index][ids],
                torch.cat([p, torch.zeros_like(p[:1])])[index][ids], mask[ids]))
        (rf, rp, rm), (sf, sp, sm) = parts
        return dict(rf=rf, sf=sf, rp=rp, sp=sp, rm=rm, sm=sm)

    @torch.no_grad()
    def refine(self, out, optimal_transport, matcher, *, hypothesis_seed, sampling_seed):
        if matcher.use_global_score:
            raise ValueError('This candidate requires the native use_global_score=False setting')
        poses, stats = hypotheses(out['ref_corr_points'], out['src_corr_points'],
            out['estimated_transform'], hypothesis_seed)
        keys = ['ref_corr_points', 'src_corr_points', 'corr_scores']
        keys += [side + '_' + key for side in ('ref', 'src')
            for key in ('points_f', 'points_c', 'feats_f', 'feats_c')]
        cpu_out = {k: out[k].detach().cpu() for k in keys}
        coarse = {side + '_features': cpu_out[side + '_feats_c'] for side in ('ref', 'src')}
        x = features_for_poses(cpu_out, coarse, self.overlap, poses, sampling_seed)
        scores = self.verifier(x.to(out['ref_points_f'].device))
        index = int(scores.argmax())
        prior = poses[index:index + 1]
        data = self._patches(out)
        # Joint models expose the affinity after their learned fine interaction.
        # Preserve that branch when using a jointly trained backbone.
        raw = out.get('interaction_logits')
        if raw is None:
            raw = data['rf'] @ data['sf'].transpose(1, 2) / 16.
        logits = raw + self.feedback_weight * spatial_bias(raw, data, prior, self.sigma_m)
        log = optimal_transport(logits, data['rm'], data['sm'])
        ref, src, confidence, pose = matcher(data['rp'], data['sp'], data['rm'],
            data['sm'], log[:, :-1, :-1], log.new_ones(len(log)))
        result = dict(out)
        result.update(initial_transform=out['estimated_transform'],
            pose_prior=prior[0], pose_prior_score=scores[index], hypothesis_stats=stats,
            ref_corr_points=ref, src_corr_points=src, corr_scores=confidence,
            estimated_transform=pose, matching_scores=log, guided_fine_logits=logits)
        return result

    @torch.no_grad()
    def register(self, model, data, *, hypothesis_seed, sampling_seed):
        if model.training:
            raise ValueError('Call model.eval() before inference')
        # The native model must never see a supervision transform in this path.
        observed = {k: v for k, v in data.items() if k != 'transform'}
        out = model(observed)
        return self.refine(out, model.optimal_transport, model.fine_matching,
            hypothesis_seed=hypothesis_seed, sampling_seed=sampling_seed)
