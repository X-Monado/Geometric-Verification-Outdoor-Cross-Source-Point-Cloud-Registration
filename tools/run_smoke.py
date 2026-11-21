"""Exercise the ordinary evaluation, Table 4 evaluation, and training entry points."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def verify_pair(target, expected):
    actual = {row["arm"]: row for row in read(target / "000000.json")}
    assert set(actual) == set(expected["records"])
    assert all(row["pair_id"] == expected["pair_id"] for row in actual.values())
    delta = max(
        float(np.max(np.abs(np.asarray(actual[arm]["pose"]) - row["pose"])))
        for arm, row in expected["records"].items()
    )
    ir_delta = max(
        abs(actual[arm]["correspondences"]["IR10"] - row["IR10"])
        for arm, row in expected["records"].items()
    )
    assert delta < 1e-4 and ir_delta < 1e-6, (target.name, delta, ir_delta)
    assert all(row["n"] == 1 and not row["complete"] for row in read(target / "summary.json"))
    return dict(pair_id=expected["pair_id"], arms=len(actual), max_pose_difference=delta,
                max_IR1_difference=ir_delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier-cache", type=Path, help="Optional local directory for verifier training")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def run(name, *cmd):
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run([sys.executable, "-u", *map(str, cmd)], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} exited with {result.returncode}. See {out / (name + '.log')}")
        print(name, "passed", flush=True)

    paths = args.paths.resolve()
    run("data", "tools/check_data.py", "--paths", paths, "--output", out / "data.json")
    report = dict(datasets=read(out / "data.json"), validation={}, full_test={}, training={})
    val = {row["domain"]: row for row in read(ROOT / "evidence/smoke_validation_expected.json")}
    for reference in read(ROOT / "evidence/smoke_test_expected.json"):
        domain = reference["domain"]
        val_dir = out / f"{domain}_val"
        run(val_dir.name, "evaluate.py", "--paths", paths, "--domain", domain, "--split", "val",
            "--max-pairs", 1, "--output", val_dir)
        report["validation"][domain] = verify_pair(val_dir, val[domain])
        test_dir = out / f"{domain}_ablation"
        run(test_dir.name, "evaluate.py", "--paths", paths, "--domain", domain, "--split", "test",
            "--ablation", "--max-pairs", 1, "--output", test_dir)
        report["full_test"][domain] = verify_pair(test_dir, reference)

    # The first Cross3DReg update is step three in the recorded schedule.
    train_dir = out / "joint_training"
    run(train_dir.name, "train.py", "--paths", paths, "--steps", 3, "--output", train_dir)
    training = [json.loads(row) for row in (train_dir / "train.jsonl").read_text().splitlines()]
    assert len(training) == 3 and all(math.isfinite(row["loss"]) and row["positive_edges"] > 0 for row in training)
    assert [len(row["pair_id"].split("_")) for row in training] == [4, 4, 3]
    model = torch.load(train_dir / "final_model.pt", map_location="cpu", weights_only=True)
    assert model["step"] == 3 and model["model"]
    report["training"]["joint"] = [{"pair_id": r["pair_id"], "loss": r["loss"]} for r in training]
    if args.verifier_cache:
        verifier_dir = out / "verifier_training"
        run(verifier_dir.name, "train_verifier.py", "--cache-dir", args.verifier_cache.resolve(),
            "--steps", 1, "--output", verifier_dir)
        verifier = torch.load(verifier_dir / "verifier.pt", map_location="cpu", weights_only=True)
        assert len(verifier["loss_trace"]) == 1 and math.isfinite(verifier["loss_trace"][0]["loss"])
        report["training"]["verifier"] = verifier["loss_trace"]
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified smoke results:", out / "report.json")


if __name__ == "__main__":
    main()
