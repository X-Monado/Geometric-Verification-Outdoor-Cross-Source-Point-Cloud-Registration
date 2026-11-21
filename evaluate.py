"""Evaluate the frozen shared checkpoint on either manuscript dataset."""

import argparse
import json
from pathlib import Path
from statistics import mean
import time

import numpy as np
import torch

from crossreg.interventions import ABLATION_STATES, STATES, infer
from crossreg.metrics import correspondence_metrics, pose_metrics
from crossreg.runtime import ROOT, check_digest, collate, cuda, dataset, load_model, load_refiner, metadata_path, settings


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def summarize(rows, arms, criteria, expected):
    summary = []
    for arm in arms:
        group = [r for r in rows if r["arm"] == arm]
        for label, (angle, distance) in criteria.items():
            good = [r for r in group if r["rre_deg"] < angle and r["rte_m"] < distance]
            summary.append(dict(
                arm=arm, protocol=label, n=len(group), expected=expected,
                complete=len(group) == expected, success=len(good), RR_pct=100 * len(good) / len(group),
                RRE_success=mean(r["rre_deg"] for r in good) if good else None,
                RTE_success=mean(r["rte_m"] for r in good) if good else None,
                IR1_pct=100 * mean(r["correspondences"]["IR10"] for r in group),
            ))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default=str(ROOT / "configs/paths.local.json"))
    parser.add_argument("--domain", choices=["kitti", "cross3dreg"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 evaluates the complete split")
    parser.add_argument("--ablation", action="store_true", help="Evaluate all ten Table 4 configurations")
    args = parser.parse_args()
    if args.max_pairs < 0:
        parser.error("--max-pairs must be zero or positive")

    paths, cfglock = settings(args.paths)
    torch.set_num_threads(2)
    torch.manual_seed(7351)
    np.random.seed(7351)
    torch.backends.cudnn.benchmark = False
    for name in ("native.pt", "verifier.pt", "overlap.pt"):
        check_digest(ROOT / "weights" / name, cfglock["weights"][name])
    metadata = metadata_path(args.domain, args.split, paths)
    metadata_digest = cfglock["metadata_hashes"][f"{args.domain}/{metadata.name}"]
    check_digest(metadata, metadata_digest)
    ds = dataset(args.domain, args.split, paths)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(domain=args.domain, split=args.split, max_pairs=args.max_pairs,
                    metadata_sha=metadata_digest, evaluation=cfglock)
    if args.ablation:
        protocol["ablation"] = list(ABLATION_STATES)
    protocol_file = args.output / "protocol.json"
    if protocol_file.exists():
        if json.loads(protocol_file.read_text()) != protocol:
            parser.error("output directory belongs to a different evaluation protocol")
    else:
        write_json(protocol_file, protocol)

    model, cfg = load_model()
    model.eval()
    refiner = load_refiner()
    n = min(len(ds), args.max_pairs or len(ds))
    rows = []
    arms = ABLATION_STATES if args.ablation else STATES
    for i in range(n):
        target = args.output / f"{i:06d}.json"
        if target.exists():
            cached = json.loads(target.read_text())
            if len(cached) != len(arms) or {row["arm"] for row in cached} != set(arms) or any(
                row["metadata_index"] != ds.indices[i] for row in cached
            ):
                raise ValueError(f"Existing result has an unexpected sample or configuration: {target}")
            rows.extend(cached)
            continue
        sample = dict(ds[i])
        truth = sample.pop("transform")
        batch = cuda(collate(sample, cfg, cfglock["limits"]))
        start = time.monotonic()
        records = []
        for arm, (pose, value) in infer(model, refiner, batch, sample["metadata_index"], args.ablation):
            matrix = pose.cpu().numpy()
            matches = [value[k].cpu().numpy() for k in ("ref_corr_points", "src_corr_points", "corr_scores")]
            records.append(dict(
                pair_id=sample["pair_id"], metadata_index=sample["metadata_index"],
                sequence=str(sample["sequence"]), arm=arm, pose=matrix.tolist(),
                **pose_metrics(matrix, truth), correspondences=correspondence_metrics(*matches, truth, matrix),
            ))
        write_json(target, records)
        rows.extend(records)
        print(i + 1, n, "seconds", round(time.monotonic() - start, 2), flush=True)
        del batch, value, pose

    criteria = dict(primary=cfglock["primary"][args.domain], strict=cfglock["strict"])
    summary = summarize(rows, arms, criteria, len(ds))
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
