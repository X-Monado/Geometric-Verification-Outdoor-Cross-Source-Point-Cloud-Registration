"""Matched target exposure, with optional Cross3DReg replay and fixed readouts."""
import argparse,datetime,gc,json,os,random,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.kitti_crossdomain_runtime_v2 import ROOT,CROSS,RELEASE,load_model,load_refiner,datasets,evaluate_panel
from scripts.train_kitti_crosssource_pilot import save,rng_state,restore
from scripts.autodiag_train_joint import train_update
from outdoor_registration.kitti_crosssource import KITTICrossSource
from outdoor_registration.expanded_data import BoundedPairs,collate
from outdoor_registration.runtime import cuda
from outdoor_registration.data import sha256
from outdoor_registration.loss import CoarseMatchingLoss
from outdoor_registration.rigid_augmentation import rotate_pair


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-updates',type=int,default=384);p.add_argument('--replay-every',type=int,default=0)
    p.add_argument('--readout-every',type=int,default=128);p.add_argument('--seed',type=int,default=7351)
    p.add_argument('--limits',default='57,69,84,98,93');p.add_argument('--backbone-lr',type=float,default=1e-5)
    p.add_argument('--stop-utc',default='2026-10-31T20:10:28+00:00');p.add_argument('--final-recovery',action='store_true')
    p.add_argument('--target-offset',type=int,default=0)
    p.add_argument('--cloudwise-norm',action='store_true');p.add_argument('--rotation-augmentation',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    assert not (a.output/'completion.json').exists()
    assert a.target_updates>0 and a.replay_every>=0
    stop=datetime.datetime.fromisoformat(a.stop_utc).timestamp();limits=[int(x) for x in a.limits.split(',')]
    assert len(limits)==5
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.55);torch.backends.cudnn.benchmark=False
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    target=KITTICrossSource(ROOT/'data/KITTI-CrossSource','train',None)
    replay=BoundedPairs(CROSS,'train',64,seed=7351)
    panels=datasets();model,cfg=load_model(a.checkpoint,cloudwise=a.cloudwise_norm)
    normalization='cloudwise' if getattr(model,'cloudwise_norm_installed',False) else 'stacked'
    other=[v for k,v in model.named_parameters() if not k.startswith(('interaction.','optimal_transport.'))]
    opt=torch.optim.Adam([dict(params=other,lr=a.backbone_lr),dict(params=list(model.interaction.parameters())+list(model.optimal_transport.parameters()),lr=1e-4)],weight_decay=1e-6)
    coarse=CoarseMatchingLoss(cfg).cuda()
    rng=np.random.RandomState(a.seed);target_order=[]
    while len(target_order)<a.target_updates+a.target_offset:target_order.extend(rng.permutation(len(target)).tolist())
    target_order=target_order[a.target_offset:a.target_offset+a.target_updates]
    rng=np.random.RandomState(a.seed+1);replay_order=[]
    while len(replay_order)<a.target_updates:replay_order.extend(rng.permutation(len(replay)).tolist())
    schedule=[];j=0
    for i,index in enumerate(target_order):
        schedule.append(dict(domain='kitti',index=index,target_count=i+1))
        if a.replay_every and (i+1)%a.replay_every==0:
            schedule.append(dict(domain='cross3dreg',index=replay_order[j],target_count=i+1));j+=1
    protocol=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha256(a.checkpoint),seed=a.seed,
                  target_updates=a.target_updates,replay_updates=j,total_updates=len(schedule),target_offset=a.target_offset,
                  limits=limits,backbone_lr=a.backbone_lr,interaction_ot_lr=1e-4,weight_decay=1e-6,
                  target_training_manifest=target.manifest(),cross_replay_manifest=replay.manifest(),schedule=schedule,
                  validation_panels={k:v.manifest() for k,v in panels.items()},test_access=False,
                  readout_every_target_updates=a.readout_every,augmentation='independentSO3_on_KITTI_only' if a.rotation_augmentation else 'none',normalization=normalization,all_parameters_trainable=True,
                  comparison='Same initialization and target pair/update order; replayadds extraCross3DReg updates. Compute differs and is reported.',
                  sampling='Per-domain logical update seeds; the KITTI GT/predicted patch sampling RNG is paired across replay arms.',
                  frozen_release_modified=False,stop_utc=a.stop_utc,
                  source_hashes={str(f):sha256(f) for f in [Path(__file__),ROOT/'scripts/kitti_crossdomain_runtime_v2.py',ROOT/'scripts/autodiag_train_joint.py',ROOT/'outdoor_adaptation/cloudwise_norm.py',ROOT/'outdoor_registration/rigid_augmentation.py']},
                  metadata_hashes={str(d.metadata_path):sha256(d.metadata_path) for d in (target,replay,*panels.values())})
    pp=a.output/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:save(pp,protocol)
    done=0;records=[];latest=a.output/'latest.pth.tar';summaries={};target_done=0
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False);assert state['protocol_sha256']==sha256(pp)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore(state['rng'])
        done=state['step'];target_done=state['target_done'];records=state['records'];summaries=state['summaries'];del state
    (a.output/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    start=time.time();model.train()
    def checkpoint():
        assert shutil.disk_usage(a.output).free>8*1024**3
        tmp=latest.with_suffix('.partial')
        torch.save(dict(model=model.state_dict(),normalization=normalization,optimizer=opt.state_dict(),rng=rng_state(),step=done,target_done=target_done,
                        records=records,summaries=summaries,protocol_sha256=sha256(pp)),tmp);os.replace(tmp,latest)
    def readout():
        path=a.output/('target%04d'%target_done);state=rng_state()
        if not (path/'completion.json').exists():
            save(a.output/'status.json',dict(status='evaluating',step=done,target_done=target_done,total_target=a.target_updates))
            summaries[str(target_done)]=evaluate_panel(model,cfg,panels,path,limits)
        else:summaries[str(target_done)]=json.loads((path/'completion.json').read_text())['summary']
        restore(state)
        weights=a.output/('target%04d_model.pt'%target_done)
        if not weights.exists():torch.save(dict(model=model.state_dict(),normalization=normalization,step=done,target_done=target_done,protocol_sha256=sha256(pp)),weights)
        checkpoint()
    # A readout follows the entire target/replay block, never between the two.
    if done and (done==len(schedule) or schedule[done]['target_count']!=target_done) and (target_done%a.readout_every==0 or target_done==a.target_updates):readout()
    for step in range(done,len(schedule)):
        if time.time()>=stop:break
        item=schedule[step];data=target if item['domain']=='kitti' else replay
        assert shutil.disk_usage(a.output).free>8*1024**3
        begin=time.time();sample=data[item['index']]
        if a.rotation_augmentation and item['domain']=='kitti':
            sample=rotate_pair(sample,(a.seed+500000+item['target_count']*1009)%2**32)
        b=cuda(collate(sample,cfg,limits))
        logical=item['target_count']+(1000000 if item['domain']=='cross3dreg' else 0)
        sample_seed=(a.seed+logical*1009)%2**32;torch.manual_seed(sample_seed);np.random.seed(sample_seed)
        metrics=train_update(model,opt,coarse,b,logical,a.seed);assert metrics['positive_edges']>0
        done=step+1;target_done=item['target_count']
        record=dict(step=done,target_done=target_done,domain=item['domain'],sequence=str(sample['sequence']),
                    metadata_index=int(sample['metadata_index']),seconds=time.time()-begin,**metrics)
        records.append(record)
        with (a.output/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        del b;gc.collect()
        save(a.output/'status.json',dict(status='training',step=done,total_steps=len(schedule),target_done=target_done,total_target=a.target_updates,
                                       last_domain=item['domain'],loss=metrics['loss'],seconds=time.time()-start,
                                       gpu_peak_gib=torch.cuda.max_memory_allocated()/1024**3,free_disk_gib=shutil.disk_usage(a.output).free/1024**3))
        print('TRAIN',a.output.name,done,len(schedule),item['domain'],target_done,round(metrics['loss'],4),round(record['seconds'],2),flush=True)
        if done%32==0:checkpoint()
        block_end=done==len(schedule) or schedule[done]['target_count']!=target_done
        if block_end and (target_done%a.readout_every==0 or target_done==a.target_updates):readout()
    checkpoint()
    final=a.output/'final_model.pt'
    if not final.exists():torch.save(dict(model=model.state_dict(),normalization=normalization,step=done,target_done=target_done,protocol_sha256=sha256(pp)),final)
    if a.final_recovery and done==len(schedule):
        evaluate_panel(model,cfg,panels,a.output/'final_recovery',limits,load_refiner())
    assert all(sha256(f)==h for f,h in protocol['source_hashes'].items())
    assert all(sha256(f)==h for f,h in protocol['metadata_hashes'].items())
    status='complete' if done==len(schedule) else 'time_budget_stopped'
    save(a.output/'completion.json',dict(status=status,steps=done,target_updates=target_done,replay_updates=sum(r['domain']=='cross3dreg' for r in records),
                                      unique_target_pairs=len({r['metadata_index'] for r in records if r['domain']=='kitti'}),
                                      unique_replay_pairs=len({r['metadata_index'] for r in records if r['domain']=='cross3dreg'}),
                                      summaries=summaries,final_model_sha256=sha256(final),seconds=time.time()-start,test_access=False))
    save(a.output/'status.json',dict(status=status,step=done,target_done=target_done))


if __name__=='__main__':main()
