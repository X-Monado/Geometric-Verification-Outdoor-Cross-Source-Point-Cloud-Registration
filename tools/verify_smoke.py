import json,sys,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];expected=json.loads((R/'evidence/smoke_expected.json').read_text());results=[]
for dom,folder in [('kitti','smoke_kitti_compiled'),('cross3dreg','smoke_cross_compiled')]:
 rows=json.loads((R/'outputs'/folder/'000000.json').read_text());full=next(r for r in rows if r['arm']=='full');old=next(r for r in expected if r['domain']==dom and r['arm']=='candidate');delta=float(np.max(np.abs(np.array(full['pose'])-np.array(old['pose']))));assert delta<1e-4,(dom,delta);results.append(dict(domain=dom,pair_id=full['pair_id'],full_max_abs_pose_difference=delta))
results.append(dict(joint_training_one_step=json.loads((R/'outputs/train_smoke/train.jsonl').read_text()),verifier_one_step=(R/'outputs/verifier_smoke/verifier.pt').exists()))
(R/'evidence/verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
