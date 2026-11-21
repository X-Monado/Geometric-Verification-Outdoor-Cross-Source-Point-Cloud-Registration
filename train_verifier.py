"""Replay verifier training from the archived candidate features."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from crossreg.listwise import fit
from crossreg.runtime import ROOT, check_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "training/verifier_cache")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if (args.output / "verifier.pt").exists():
        parser.error("choose a fresh output directory")
    torch.set_num_threads(2)
    protocol = json.loads((ROOT / "training/verifier_protocol.json").read_text())
    manifest_path = ROOT / "training/verifier_cache/records.json"
    check_digest(manifest_path, protocol["release_records_sha256"])
    manifest = json.loads(manifest_path.read_text())
    angle, distance = protocol["primary"]
    records = []
    for record in manifest:
        path = args.cache_dir / record["path"]
        check_digest(path, record["sha256"])
        with np.load(path, allow_pickle=False) as cache:
            labels = (cache["rre"] < angle) & (cache["rte"] < distance)
            records.append(dict(x=torch.from_numpy(cache["features"].astype(np.float32)),
                                y=torch.from_numpy(labels.astype(np.float32))))
    verifier = ROOT / "weights/verifier.pt"
    expected = json.loads((ROOT / "configs/evaluation.json").read_text())["weights"][verifier.name]
    check_digest(verifier, expected)
    indices = torch.load(verifier, map_location="cpu", weights_only=True)["model"]["indices"]
    model, trace = fit(records, indices, args.steps, 7351)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), loss_trace=trace), args.output / "verifier.pt")
    print("Saved", args.output / "verifier.pt")


if __name__ == "__main__":
    main()
