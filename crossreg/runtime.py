import json,pickle,hashlib,copy
from pathlib import Path
import numpy as np,joblib,torch
from outdoor_registration import bootstrap
from outdoor_registration.config import make_cfg
from outdoor_registration.expanded_data import BoundedPairs,collate
from outdoor_registration.kitti_crosssource import KITTICrossSource
from outdoor_registration.runtime import cuda
from outdoor_adaptation.joint_fine_interaction import JointFineGeoTransformer
from outdoor_adaptation.prior_refinement import PriorRefitRefiner
from outdoor_adaptation.consistency_guard import guarded_pose
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def settings(paths):
    path = Path(paths)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: copy configs/paths.example.json and set your dataset roots")
    return json.loads(path.read_text()), json.loads((ROOT/'configs/evaluation.json').read_text())


def check_digest(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    if sha(path) != expected:
        with path.open("rb") as stream:
            lfs_pointer = stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
        hint = " Run git lfs pull to fetch the file contents." if lfs_pointer else ""
        raise ValueError(f"SHA-256 does not match the recorded artifact: {path}.{hint}")


def metadata_path(domain, split, paths):
    root = Path(paths[domain])
    return root / (f"Split/{split}_cs2.pkl" if domain == "kitti" else f"metadata/{split}.pkl")
def dataset(domain,split,paths,per_sequence=None):
 root=Path(paths[domain])
 if split!='test':return KITTICrossSource(root,split,per_sequence) if domain=='kitti' else BoundedPairs(root,split,per_sequence or 10**9,seed=7351)
 cls=KITTICrossSource if domain=='kitti' else BoundedPairs
 ds=object.__new__(cls);ds.root=root.resolve();ds.split=split;ds.seed=7351;ds.point_limit=30000;ds.voxel_size=.3;ds.cache={}
 ds.metadata_path=root/('Split/test_cs2.pkl' if domain=='kitti' else 'metadata/test.pkl')
 ds.rows=pickle.load(open(ds.metadata_path,'rb')) if domain=='kitti' else joblib.load(ds.metadata_path);ds.indices=list(range(len(ds.rows)));return ds

def load_model(path=None):
 cfg=copy.deepcopy(make_cfg());cfg.coarse_matching.num_targets=16
 model=JointFineGeoTransformer(cfg,'geometry').cuda();state=torch.load(path or ROOT/'weights/native.pt',map_location='cpu',weights_only=False)
 model.load_state_dict(state['model'],strict=True);model.skip_training_registration=True
 if state.get('normalization')=='cloudwise':
  from outdoor_adaptation.cloudwise_norm import install_cloudwise_norm
  install_cloudwise_norm(model)
 return model,cfg

def load_refiner():return PriorRefitRefiner.from_checkpoints(ROOT/'weights/verifier.pt',ROOT/'weights/overlap.pt',device='cuda',sigma_m=1.2,feedback_weight=.5,refit_radius=4.8)
@torch.no_grad()
def infer_outputs(model,refiner,batch,index):
 assert 'transform' not in batch
 native=model(batch);c=refiner.refine(native,model.optimal_transport,model.fine_matching,hypothesis_seed=7352+index*1009,sampling_seed=index+75001)
 full,guard=consistency_output(native,c)
 pre=c['pre_refit_transform'];angle=torch.rad2deg(torch.acos(torch.clamp(((full[:3,:3]*pre[:3,:3]).sum()-1)/2,-1,1)));shift=torch.linalg.norm(full[:3,3]-pre[:3,3]);gate=pre if angle>2 or shift>.4 else full
 return dict(full=(full,c),update_gate=(gate,c),native=(native['estimated_transform'],native),
             without_refit=(pre,c),without_guard=(c['estimated_transform'],c)),dict(rotation_change_deg=float(angle),translation_change_m=float(shift),**guard)


def consistency_output(native, candidate):
    return guarded_pose(native['estimated_transform'], candidate['pose_prior'],
                        candidate['pre_refit_transform'], candidate['estimated_transform'], native['src_points'])
