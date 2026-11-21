import torch
import numpy as np
from outdoor_registration.small_cloud import point_to_node_partition
def prepare(row,ids=None):
    out={k:v.cuda() for k,v in row['out'].items()};parts=[]
    for side in ('ref','src'):
        p,f=out[side+'_points_f'],out[side+'_feats_f']
        _,_,index,mask=point_to_node_partition(p,out[side+'_points_c'],128)
        parts.append(dict(points=torch.cat([p,torch.zeros_like(p[:1])])[index],
            features=torch.cat([f,torch.zeros_like(f[:1])])[index],mask=mask))
    if ids is None:ids=torch.stack([out['ref_node_corr_indices'],out['src_node_corr_indices']],-1)
    else:ids=ids.cuda()
    r,s=parts;ri,si=ids[:,0],ids[:,1]
    data=dict(rf=r['features'][ri],sf=s['features'][si],rp=r['points'][ri],sp=s['points'][si],rm=r['mask'][ri],sm=s['mask'][si])
    return out,data

def labels(data,truth):
    aligned=data['sp'].double()@truth.double()[:3,:3].T+truth.double()[:3,3]
    return ((data['rp'].double()[:,:,None]-aligned[:,None]).square().sum(-1)<.36)&data['rm'][:,:,None]&data['sm'][:,None]

def affinity(module,data):
    rf,sf=module(**data)
    return rf@sf.transpose(1,2)/16.,((rf-data['rf']).square().mean()+(sf-data['sf']).square().mean())*.5
def train_update(model,optimizer,coarse,batch,step,seed):
    optimizer.zero_grad(set_to_none=True);out=model(batch);truth=batch['transform']
    first={short:out['%s_node_corr_knn_%s'%(side,suffix)] for short,side,suffix in
        [('rp','ref','points'),('sp','src','points'),('rm','ref','masks'),('sm','src','masks')]}
    target=labels(first,truth);logs=[out['matching_scores']];targets=[target];rms=[first['rm']];sms=[first['sm']]
    ids=torch.stack([out['ref_node_corr_indices'],out['src_node_corr_indices']],1)
    rng=np.random.RandomState(seed+step*1009);chosen=torch.as_tensor(rng.choice(len(ids),min(16,len(ids)),replace=False),device='cuda')
    _,data=prepare(dict(out=out),ids[chosen]);logits,reg=affinity(model.interaction,data)
    logs.append(model.optimal_transport(logits,data['rm'],data['sm']));targets.append(labels(data,truth));rms.append(data['rm']);sms.append(data['sm'])
    log=torch.cat(logs);target=torch.cat(targets);rm=torch.cat(rms);sm=torch.cat(sms)
    positive=-log[:,:-1,:-1][target].mean()
    unmatched=torch.cat([log[:,:-1,-1][rm&~target.any(2)],log[:,-1,:-1][sm&~target.any(1)]])
    negative=-unmatched.mean() if len(unmatched) else positive*0
    closs=coarse(out);loss=closs+positive+.5*negative+1e-4*reg
    assert bool(torch.isfinite(loss));loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
    assert bool(torch.isfinite(gn));optimizer.step()
    return dict(loss=float(loss),coarse_loss=float(closs),positive_loss=float(positive),unmatched_loss=float(negative),
        gradient_norm=float(gn),positive_edges=int(target.sum()),pair_id=batch['pair_id'],fine_regions=len(log))