"""Adaptive round 3: matched joint updates and training-data coverage controls."""
import argparse,copy,gzip,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from outdoor_registration import bootstrap
from outdoor_registration.config import make_cfg
from outdoor_registration.data import sha256
from outdoor_registration.runtime import cuda
from outdoor_registration.loss import CoarseMatchingLoss
from outdoor_registration.diagnostics import evaluate
from outdoor_adaptation.joint_fine_interaction import JointFineGeoTransformer
from scripts.rotation_stage1_runtime import BASE,graph_path,save,rng_state,restore_rng
from scripts.autodiag_interaction_runtime import ROOT,prepare,labels,affinity
from scripts.fine_anchor_runtime import predicted_anchors,anchor_stats


def load_graph(item):
    if item['compressed']:
        with gzip.open(item['path'],'rb') as stream:row=torch.load(stream,weights_only=True)
        for key in ('neighbors','subsampling','upsampling'):row[key]=[x.long() for x in row[key]]
        return row
    return torch.load(item['path'],weights_only=True)


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


@torch.no_grad()
def evaluate_graphs(model,cfg,items,path):
    model.eval();records=[]
    for i,item in enumerate(items):
        batch=cuda(load_graph(item));truth=batch.pop('transform');out=model(batch);m=evaluate(out,truth,cfg)
        d={short:out['%s_node_corr_knn_%s'%(side,suffix)] for short,side,suffix in [('rp','ref','points'),('sp','src','points'),('rm','ref','masks'),('sm','src','masks')]}
        target=labels(d,truth);valid=d['rm'][:,:,None]&d['sm'][:,None];logits=out['interaction_logits'];eligible=target.any(1)
        ids=logits.masked_fill(~valid,-torch.inf).argmax(1,keepdim=True)
        m.update(pair_id=batch['pair_id'],sequence=batch['sequence'],top1=float((target.gather(1,ids).squeeze(1)&eligible).sum()/eligible.sum().clamp_min(1)),
            anchor=anchor_stats(predicted_anchors(logits,valid,16),target))
        m['medium']=bool(m['valid_pose'] and m['rre_deg']<5 and m['rte_m']<1)
        m['relaxed']=bool(m['valid_pose'] and m['rre_deg']<5 and m['rte_m']<2)
        records.append(m)
    summary=dict(pairs=len(records),strict=sum(x['success'] for x in records),medium=sum(x['medium'] for x in records),relaxed=sum(x['relaxed'] for x in records),
        source_recall=float(np.mean([x['inlier_spatial_support']['src']['fraction_of_visible_fine_points'] for x in records])),
        fine_ir=float(np.mean([x['fine_inlier_ratio']['0.6'] for x in records])),top1=float(np.mean([x['top1'] for x in records])),
        anchor_positive_precision=float(np.mean([x['anchor']['positive_patch_correct']/max(x['anchor']['positive_patch_anchors'],1) for x in records])))
    save(path,dict(summary=summary,records=records));print('EVAL',path.name,json.dumps(summary),flush=True);model.train();return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--kind',choices=('point','geometry'),required=True)
    p.add_argument('--data-size',type=int,choices=(192,703),required=True);p.add_argument('--steps',type=int,default=1406)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--backbone-lr',type=float,default=1e-5);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;a.output.mkdir(parents=True,exist_ok=True)
    cfg=copy.deepcopy(make_cfg());cfg.coarse_matching.num_targets=16
    cp=BASE/('baseline_seed%d'%a.seed)/'step1920.pth.tar';mp=cp.parent/'manifest.json';manifest=json.loads(mp.read_text())
    hp=ROOT/('round2_%s_seed%d'%(a.kind,a.seed))/'step0768.pth.tar'
    graph_manifest=ROOT/'expanded_graphs/manifest.json'
    if a.data_size==703:train=[x for x in json.loads(graph_manifest.read_text())['records'] if x['split']=='train']
    else:train=[dict(path=str(graph_path(manifest,x,'train')),compressed=False,metadata_index=x['metadata_index']) for x in manifest['train']['pairs']]
    val=[dict(path=str(graph_path(manifest,x,'val')),compressed=False,metadata_index=x['metadata_index']) for x in manifest['final']['pairs']]
    paths=[Path(__file__),Path('outdoor_adaptation/joint_fine_interaction.py'),Path('outdoor_adaptation/fine_interaction.py'),Path('scripts/autodiag_interaction_runtime.py')]
    paths+=list(Path('outdoor_registration').rglob('*.py'))
    sources={str(x):sha256(x) for x in paths};inputs={str(x):sha256(x) for x in (cp,mp,hp)}
    for item in train+val:inputs[item['path']]=sha256(item['path'])
    lock=dict(source_hashes=sources,input_hashes=inputs,seed=a.seed,kind=a.kind,data_size=a.data_size,steps=a.steps,
        original_lr=a.backbone_lr,interaction_ot_lr=1e-4,optimizer='new Adam, identical reset policy for all arms',
        training='16 original GT strong-positive patches plus 16 random predicted patches; coarse loss + balanced OT loss; all original parameters trainable',test_access=False)
    lp=a.output/'lock.json'
    if lp.exists():assert json.loads(lp.read_text())==lock
    else:save(lp,lock)
    torch.manual_seed(a.seed+31000);np.random.seed(a.seed+31000)
    model=JointFineGeoTransformer(cfg,a.kind).cuda();base=torch.load(cp,map_location='cpu',weights_only=False)['model']
    missing,extra=model.load_state_dict(base,strict=False);assert not extra and all(k.startswith('interaction.') for k in missing)
    head=torch.load(hp,map_location='cpu',weights_only=False);model.interaction.load_state_dict(head['module']);model.optimal_transport.load_state_dict(head['ot'])
    model.skip_training_registration=True
    other=[v for k,v in model.named_parameters() if not k.startswith(('interaction.','optimal_transport.'))]
    optimizer=torch.optim.Adam([dict(params=other,lr=a.backbone_lr),dict(params=list(model.interaction.parameters())+list(model.optimal_transport.parameters()),lr=1e-4)],weight_decay=1e-6)
    coarse=CoarseMatchingLoss(cfg).cuda();latest=a.output/'latest.pth.tar';done=0;records=[]
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False);assert state['lock_sha256']==sha256(lp)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);done=state['step'];records=state['records'];restore_rng(state['rng'])
    rng=np.random.RandomState(a.seed+31000);order=[]
    while len(order)<a.steps:order.extend(rng.permutation(len(train)).tolist())
    start=time.time();model.train()
    def checkpoint():
        tmp=latest.with_suffix('.partial');torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),step=done,records=records,lock_sha256=sha256(lp)),tmp);os.replace(tmp,latest)
        if done%703==0 or done==a.steps:
            final=a.output/('step%05d.pth.tar'%done)
            if not final.exists():os.link(latest,final)
    for step in range(done,a.steps):
        batch=cuda(load_graph(train[order[step]]));metrics=train_update(model,optimizer,coarse,batch,step,a.seed);done=step+1
        records.append(dict(step=done,metadata_index=int(batch['metadata_index']),**metrics))
        if done%32==0 or done==a.steps:
            checkpoint();save(a.output/'status.json',dict(status='training',step=done,total=a.steps,elapsed_seconds=time.time()-start,loss=metrics['loss']))
            print('JOINT',a.seed,a.kind,a.data_size,done,round(metrics['loss'],3),round(time.time()-start,1),flush=True)
        if done%703==0 or done==a.steps:
            checkpoint();evaluate_graphs(model,cfg,val,a.output/('step%05d_val122.json'%done))
    if not (a.output/('step%05d_val122.json'%a.steps)).exists():evaluate_graphs(model,cfg,val,a.output/('step%05d_val122.json'%a.steps))
    (a.output/'train.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    assert all(sha256(x)==h for x,h in sources.items()) and all(sha256(x)==h for x,h in inputs.items())
    save(a.output/'completion.json',dict(status='complete',steps=done,seed=a.seed,kind=a.kind,data_size=a.data_size,
        source_hashes=sources,input_hashes=inputs,checkpoint_sha256=sha256(a.output/('step%05d.pth.tar'%done)),test_access=False))
    print('COMPLETE JOINT',a.output,flush=True)


if __name__=='__main__':main()
