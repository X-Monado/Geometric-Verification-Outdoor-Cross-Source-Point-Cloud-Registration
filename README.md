# CrossSourceReg — manuscript reproduction package

Reproduction code for *Geometric Verification and Pose Feedback for Outdoor Cross-Source Point Cloud Registration*. KITTI-CrossSource and Cross3DReg share the same backbone, fine-interaction, overlap and verifier weights. A [Chinese quick start](README_zh.md) is available.

## Install

```bash
conda env create -f environment.yml
conda activate crosssource-reg
cd outdoor_registration/vendor
MAX_JOBS=2 python setup.py build_ext --inplace
cd ../..
```
Build on Linux with an NVIDIA GPU, CUDA 12.4 and a compatible host C++ compiler. If your system has multiple CUDA installations, point `CUDA_HOME` at the 12.4 toolkit before compiling. The build was tested from source on a V100 with Python 3.8 and PyTorch 2.4.0+cu124. See [the smoke command](#smoke-test) for a fresh checkout.

## Data and weights

See [data/README.md](data/README.md) for official downloads and layouts. Copy `configs/paths.example.json` to `configs/paths.local.json` and set both roots, or pass `--paths PATH`. Use metadata from the dataset providers. The entry points verify that each manifest matches the recorded paper split before loading it.

```bash
python tools/check_data.py
```
Inference uses `weights/native.pt`, `weights/verifier.pt`, and `weights/overlap.pt`. Final-stage training starts from `weights/training_init.pt`. After cloning, run `git lfs pull` to retrieve the checkpoint contents. Each entry point verifies the required checkpoint hashes.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --domain kitti --split test --output outputs/kitti_test
CUDA_VISIBLE_DEVICES=0 python evaluate.py --domain cross3dreg --split test --output outputs/cross_test
# Table 4, all ten fixed-weight configurations:
CUDA_VISIBLE_DEVICES=0 python evaluate.py --domain kitti --split test --ablation --output outputs/kitti_ablation
CUDA_VISIBLE_DEVICES=0 python evaluate.py --domain cross3dreg --split test --ablation --output outputs/cross_ablation
```
The default run emits `full`, the frozen 4.8 m refit with the consistency guard, and `update_gate`, which returns the feedback pose if the subsequent update exceeds 2° or 0.4 m. `--ablation` adds the native pose, feedback pose, and fixed-weight module interventions. Each pair is saved atomically. Resume with the same command and output directory. Use a new directory for a different configuration.

| Dataset | Variant | Primary RR | Strict RR |
|---|---|---:|---:|
| KITTI544 | full |93.1985%|47.7941%|
| KITTI544 | update_gate |93.1985%|56.9853%|
| KITTI544 | without_refit |93.1985%|64.7059%|
| Cross4145 | full |92.4729%|70.8082%|
| Cross4145 | update_gate |94.1134%|79.3245%|
| Cross4145 | without_refit |94.1616%|86.5139%|

A pair succeeds when both errors are below the specified limits. Primary limits are 5°/2 m for KITTI and 5°/1 m for Cross3DReg. Strict limits are 2°/0.5 m. RRE and RTE average successful pairs, and IR@1m averages all pairs. The archived [complete test metrics](evidence/full_test_interventions.csv) and [paired transitions](evidence/full_test_paired_transitions.csv) come from the frozen experiment. The release entry point was checked against archived smoke pairs.

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_smoke.py --paths configs/paths.local.json --output outputs/smoke
```

The smoke checks all six dataset manifests and their files, compares one validation pair and one ten-configuration test pair per dataset against archived poses and inlier ratios, and executes three recorded joint training steps to reach both domains. Pass `--verifier-cache /path/to/candidate-cache` to include one verifier update. See the [release smoke report](evidence/release_smoke_20261121.json) and [training scope](docs/TRAINING.md).

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --steps 1 --output outputs/train_smoke
CUDA_VISIBLE_DEVICES=0 python train.py --output outputs/final_stage_replay
```
`train.py` uses the original joint loss and recorded 1941-update final-stage schedule (1294 KITTI and 647 Cross3DReg replay updates), starting from the bundled `training_init.pt`. The verifier was trained with a success target of 15°/2 m. Evaluation thresholds apply at reporting time. See [docs/TRAINING.md](docs/TRAINING.md) for scope and head fitting. Supplied checkpoints and historical source cover the preceding backbone and overlap stages. Final-stage numerical identity after retraining depends on CUDA and extension determinism and remains unverified over a complete training rerun.

## Attribution and release scope

GeoTransformer-derived operators are in `outdoor_registration/vendor`. See the [third-party license](THIRD_PARTY_LICENSE.GeoTransformer) and [publication handoff](docs/PUBLISHING.md) for release scope and attribution. Historical scripts in `training/historical` record the original experiment provenance. The reproduction commands above use the portable entry points.
