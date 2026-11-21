"""Replay the documented final joint-adaptation stage from its bundled initialization."""
import argparse,json,random
from pathlib import Path
import torch,numpy as np
from crossreg.runtime import ROOT,settings,dataset,load_model,collate,cuda,check_digest,metadata_path
from crossreg.training_loss import train_update
from outdoor_registration.loss import CoarseMatchingLoss

def main():
 p=argparse.ArgumentParser();p.add_argument('--paths',default=str(ROOT/'configs/paths.local.json'));p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 assert not (a.output/'final_model.pt').exists(),'Use a new output directory'
 paths,ev=settings(a.paths);protocol=json.loads((ROOT/'training/final_stage_protocol.json').read_text());torch.set_num_threads(2);torch.manual_seed(7351);np.random.seed(7351);random.seed(7351);torch.backends.cudnn.benchmark=False
 if not 0 <= a.steps <= len(protocol['schedule']):p.error('--steps must be zero or within the recorded schedule')
 check_digest(ROOT/'weights/training_init.pt',protocol['checkpoint_sha256'])
 for domain in ('kitti','cross3dreg'):
  meta=metadata_path(domain,'train',paths);check_digest(meta,protocol['metadata_hashes'][meta.name])
 data={d:dataset(d,'train',paths,64 if d=='cross3dreg' else None) for d in ['kitti','cross3dreg']}
 assert data['kitti'].manifest()==protocol['target_training_manifest']
 assert data['cross3dreg'].manifest()==protocol['cross_replay_manifest']
 model,cfg=load_model(ROOT/'weights/training_init.pt');model.train();other=[v for k,v in model.named_parameters() if not k.startswith(('interaction.','optimal_transport.'))]
 opt=torch.optim.Adam([dict(params=other,lr=1e-5),dict(params=list(model.interaction.parameters())+list(model.optimal_transport.parameters()),lr=1e-4)],weight_decay=1e-6);coarse=CoarseMatchingLoss(cfg).cuda()
 schedule=protocol['schedule'][:a.steps or None]
 for step,item in enumerate(schedule):
  batch=cuda(collate(data[item['domain']][item['index']],cfg,ev['limits']));logical=item['target_count']+(1000000 if item['domain']=='cross3dreg' else 0);seed=(7351+logical*1009)%2**32;torch.manual_seed(seed);np.random.seed(seed)
  metrics=train_update(model,opt,coarse,batch,logical,7351)
  with (a.output/'train.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,**metrics))+'\n')
  print(step+1,len(schedule),metrics['loss'],flush=True);del batch
 torch.save(dict(model=model.state_dict(),step=len(schedule),normalization='stacked'),a.output/'final_model.pt');(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
