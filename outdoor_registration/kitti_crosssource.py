"""Official KITTI-CrossSource NPY pairs: pcd1 reconstruction -> pcd0 LiDAR."""
import pickle
from pathlib import Path
import numpy as np


class KITTICrossSource:
    def __init__(self, root, split, per_sequence=None, point_limit=30000, seed=7351):
        if split not in ('train', 'val'):
            raise ValueError('Development loader permits train/val only')
        self.root = Path(root).resolve()
        self.split, self.point_limit, self.seed = split, point_limit, seed
        self.metadata_path = self.root/'Split'/(split+'_cs2.pkl')
        with self.metadata_path.open('rb') as f:
            self.rows = pickle.load(f)
        self.indices = []
        for seq in sorted({r['seq_id'] for r in self.rows}):
            ids = [i for i,r in enumerate(self.rows) if r['seq_id'] == seq]
            offsets = np.arange(len(ids)) if per_sequence is None else np.linspace(0,len(ids)-1,min(per_sequence,len(ids)),dtype=int)
            self.indices.extend(ids[int(i)] for i in offsets)

    def __len__(self):
        return len(self.indices)

    def path(self, name):
        p = (self.root/name).resolve()
        if self.root not in p.parents:
            raise ValueError('Metadata path escapes root')
        return p

    def manifest(self):
        return [dict(metadata_index=i, sequence=int(self.rows[i]['seq_id']),
                     ref=self.rows[i]['pcd0'], src=self.rows[i]['pcd1']) for i in self.indices]

    def __getitem__(self, j):
        i = self.indices[j]
        r = self.rows[i]
        t = np.asarray(r['transform'], dtype=np.float32)
        assert t.shape == (4,4) and np.isfinite(t).all() and np.allclose(t[3],[0,0,0,1])
        assert np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) and abs(np.linalg.det(t[:3,:3])-1)<1e-5
        sample = dict(transform=t, sequence=int(r['seq_id']), metadata_index=i,
                      pair_id='%s_%02d_%06d_%06d' % (self.split,r['seq_id'],r['frame0'],r['frame1']),raw_counts={})
        for side, field in [('ref','pcd0'),('src','pcd1')]:
            xyz = np.load(self.path(r[field]), allow_pickle=False)
            assert xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz)>128 and np.isfinite(xyz).all()
            sample['raw_counts'][side] = len(xyz)
            if self.point_limit and len(xyz)>self.point_limit:
                rng = np.random.RandomState(self.seed+i+(100000 if side=='src' else 0))
                xyz = xyz[np.sort(rng.choice(len(xyz),self.point_limit,replace=False))]
            sample[side+'_points'] = np.ascontiguousarray(xyz,dtype=np.float32)
            sample[side+'_feats'] = np.ones((len(xyz),1),dtype=np.float32)
        return sample
