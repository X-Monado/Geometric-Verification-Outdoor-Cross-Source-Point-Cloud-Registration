"""Cross3DReg: source=array, reference=lidar, T maps source to reference.

Development preprocessing: centroid voxelization at 0.3 m in each input's
own frame. Repeated points contribute to centroids; no explicit deduplication.
No RGB, GT-dependent cropping, normalization, or augmentation is performed.
"""
from pathlib import Path
import hashlib
import joblib
import numpy as np
import torch
from scipy.spatial import cKDTree

from . import bootstrap
from geotransformer.modules.ops import grid_subsample
from geotransformer.utils.data import registration_collate_fn_stack_mode


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_xyz(path):
    with Path(path).open('rb') as stream:
        lines = []
        for _ in range(100):
            line = stream.readline().decode('ascii').strip()
            lines.append(line)
            if line == 'end_header':
                break
        else:
            raise ValueError('Missing PLY end_header')
        if 'format binary_little_endian 1.0' not in lines:
            raise ValueError('Unsupported PLY encoding')
        props = [line for line in lines if line.startswith('property')]
        if props != ['property double x', 'property double y', 'property double z']:
            raise ValueError('Expected XYZ float64 PLY')
        n = int(next(line.split()[-1] for line in lines if line.startswith('element vertex')))
        points = np.fromfile(stream, dtype='<f8', count=3*n).reshape(n, 3)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Empty or nonfinite cloud')
    return points.astype(np.float32)


def voxelize(points, size):
    points, _ = grid_subsample(torch.from_numpy(points), torch.LongTensor([len(points)]), size)
    return points.numpy()


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def pose_errors(estimate, truth):
    cosine = (np.trace(estimate[:3, :3].T @ truth[:3, :3])-1)/2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))), float(np.linalg.norm(estimate[:3, 3]-truth[:3, 3]))


class Cross3DReg:
    def __init__(self, root, split, per_sequence=1, voxel_size=.3, point_limit=30000, seed=7351):
        if split not in ('train', 'val'):
            raise ValueError('Development runner intentionally permits only train/val')
        self.root = Path(root).resolve()
        self.split, self.voxel_size, self.point_limit, self.seed = split, voxel_size, point_limit, seed
        self.metadata_path = self.root/'metadata'/(split+'.pkl')
        self.rows = joblib.load(self.metadata_path)
        self.indices = []
        for seq in sorted(set(row['seq_id'] for row in self.rows)):
            ids = [i for i, row in enumerate(self.rows) if row['seq_id'] == seq]
            offsets = [len(ids)//2] if per_sequence == 1 else np.linspace(0, len(ids)-1, min(per_sequence, len(ids)), dtype=int)
            self.indices.extend(ids[int(offset)] for offset in offsets)
        self.cache = {}

    def __len__(self):
        return len(self.indices)

    def path(self, value):
        path = (self.root/value.replace('\\', '/')).resolve()
        if self.root not in path.parents:
            raise ValueError('Metadata path escapes dataset root')
        return path

    def manifest(self):
        return dict(split=self.split, metadata_sha256=sha256(self.metadata_path), selection='equally spaced metadata offsets within each sequence; midpoint if one',
                    pairs=[dict(metadata_index=i, sequence=self.rows[i]['seq_id'], frame=self.rows[i]['frame0'],
                                source=self.rows[i]['pcd0'], reference=self.rows[i]['pcd1']) for i in self.indices])

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        metadata_index = self.indices[index]
        row = self.rows[metadata_index]
        transform = np.asarray(row['transform'], dtype=np.float32)
        if not (np.allclose(transform[3], [0, 0, 0, 1]) and
                np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5) and
                np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-5)):
            raise ValueError('Invalid SE(3) ground truth')
        clouds, raw_counts, voxel_counts = {}, {}, {}
        for modality, field in [('src', 'pcd0'), ('ref', 'pcd1')]:
            points = read_xyz(self.path(row[field]))
            raw_counts[modality] = len(points)
            points = voxelize(points, self.voxel_size)
            voxel_counts[modality] = len(points)
            if self.point_limit and len(points) > self.point_limit:
                rng = np.random.RandomState(self.seed+metadata_index+(0 if modality == 'src' else 100000))
                points = points[np.sort(rng.choice(len(points), self.point_limit, replace=False))]
            clouds[modality+'_points'] = np.ascontiguousarray(points)
            clouds[modality+'_feats'] = np.ones((len(points), 1), dtype=np.float32)
        clouds.update(transform=transform, pair_id='%s_%s_%s' % (self.split, row['seq_id'], row['frame0']),
                      sequence=row['seq_id'], metadata_index=metadata_index,
                      raw_counts=raw_counts, voxel_counts=voxel_counts)
        self.cache[index] = clouds
        return clouds


def collate(sample, cfg, limits):
    return registration_collate_fn_stack_mode([sample], cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size, cfg.backbone.init_radius, limits)


def calibrate(dataset, cfg, keep_ratio=.8):
    """Exact CPU neighbor counts, only train samples, without histogram overflow."""
    if dataset.split != 'train':
        raise ValueError('Neighbor calibration requires training data')
    counts = [[[], []] for _ in range(cfg.backbone.num_stages)]
    lengths = []
    for j in range(len(dataset)):
        sample = dataset[j]
        sides = [sample['ref_points'], sample['src_points']]
        sizes = []
        for stage in range(cfg.backbone.num_stages):
            if stage:
                sides = [voxelize(p, cfg.backbone.init_voxel_size*2**stage) for p in sides]
            radius = cfg.backbone.init_radius*2**stage
            for side, points in enumerate(sides):
                counts[stage][side].append(cKDTree(points).query_ball_point(points, radius, return_length=True))
            sizes.append([len(points) for points in sides])
        lengths.append(dict(pair_id=sample['pair_id'], stage_lengths_ref_src=sizes))
    limits, statistics = [], []
    for stage_counts in counts:
        sides = [np.concatenate(side) for side in stage_counts]
        both = np.concatenate(sides)
        limit = max(1, int(np.quantile(both, keep_ratio, method='higher')))
        limits.append(limit)
        statistics.append({name: dict(p50=float(np.median(c)), p95=float(np.percentile(c,95)),
            max=int(c.max()), query_truncation_fraction=float(np.mean(c > limit)),
            neighbor_discard_fraction=float(np.maximum(c-limit,0).sum()/c.sum()))
            for name,c in zip(('ref_lidar','src_array'),sides)})
    return dict(neighbor_limits=limits, keep_ratio=keep_ratio, samples=dataset.manifest(),
                statistics=statistics, stage_lengths=lengths, note='Same-level convolutions; does not claim cross-level query truncation statistics.')
