# Training provenance and supported replay

The paper's shared network is frozen release crossdomain_candidate_20261031, native SHA2568d51512fb2be226cfd36374aa3940ae5db52d10fb70b6897d2d030e0fbc1f270. The release was selected as round2_replay_control. The executable training entrypoint replays that final joint-adaptation stage from the previous round checkpoint, included as training_init.pt. It uses the recorded schedule, identical fine/coarse/dustbin loss functions, logical update seeds, Adam learning rates1e-5(backbone) and1e-4(interaction/transport), and weight decay1e-6. No test samples are loaded by training.

Verifier reproduction:
```
python train_verifier.py --cache-dir /path/to/candidate-cache --steps 1 --output outputs/verifier_smoke
python train_verifier.py --cache-dir /path/to/candidate-cache --output outputs/verifier_replay
```
The candidate cache consists of 273 training-pair feature and pose-error files. The repository includes their filenames and checksums in `training/verifier_cache/records.json`. Obtain the cache separately from the authors when distribution is authorized. Local copies under `training/verifier_cache` are excluded from default Git staging. The original listwise objective uses positives at 15°/2 m, 800 updates, seed 7351, and the preserved feature subset. Historical source records the earlier held-sequence diagnostics.

The overlap head and preceding backbone pretraining are supplied as checkpoints. Their entire preceding historical multistage training chain has not been repackaged into a fresh-from-scratch reproduction recipe. Do not claim that a one-command training run recreates all initial training stages. The original relevant training sources are kept under historical/ for inspection; supported commands are train.py and train_verifier.py.

A one-step smoke verifies execution only. Full training replay, multiseed stability and a clean environment installation remain distinct verification tasks, not prerequisites for replaying the supplied paper checkpoints.
