"""Verify all split manifests and check that their point-cloud paths exist."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crossreg.runtime import ROOT, check_digest, dataset, metadata_path, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default=str(ROOT / "configs/paths.local.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths, evaluation = settings(args.paths)
    training = json.loads((ROOT / "training/final_stage_protocol.json").read_text())
    result = []
    for domain in ("kitti", "cross3dreg"):
        for split in ("train", "val", "test"):
            path = metadata_path(domain, split, paths)
            expected = (training["metadata_hashes"][path.name] if split == "train" else
                        evaluation["metadata_hashes"][f"{domain}/{path.name}"])
            check_digest(path, expected)
            data = dataset(domain, split, paths)
            missing = [row[k] for row in data.rows for k in ("pcd0", "pcd1")
                       if not data.path(row[k]).is_file()]
            if missing:
                raise FileNotFoundError(f"{domain}/{split}: missing {len(missing)} point clouds, e.g. {missing[:3]}")
            result.append(dict(domain=domain, split=split, pairs=len(data), missing_files=0,
                               metadata_sha256=expected))
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
