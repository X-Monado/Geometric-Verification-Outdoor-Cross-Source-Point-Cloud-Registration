"""The fixed-weight interventions from the manuscript's complete test table."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from outdoor_adaptation import pose_guided_registration as pg
from crossreg.runtime import consistency_output, infer_outputs

STATES = ("full", "update_gate")
ABLATION_STATES = (
    "native", "full", "without_refit", "without_guard", "without_feedback",
    "geometry_rank", "neutral_overlap", "without_native_candidate",
    "without_fine_interaction", "update_gate",
)


class GeometryRank(torch.nn.Module):
    def forward(self, evidence):
        return evidence[:, 73]  # OT-confidence weighted support at 0.6 m.


class NeutralOverlap(torch.nn.Module):
    def forward(self, features):
        return features.new_zeros(len(features))  # sigmoid gives 0.5.


@contextmanager
def intervention(refiner, arm):
    if arm == "geometry_rank":
        change = patch.object(refiner, "verifier", GeometryRank())
    elif arm == "neutral_overlap":
        change = patch.object(refiner, "overlap", NeutralOverlap())
    elif arm == "without_feedback":
        change = patch.object(refiner, "feedback_weight", 0.0)
    else:
        assert arm == "without_native_candidate"
        original = pg.hypotheses

        def remove_native(*args, **kwargs):
            poses, stats = original(*args, **kwargs)
            return (poses[1:] if len(poses) > 1 else poses), dict(stats, native_fallback=len(poses) == 1)

        change = patch.object(pg, "hypotheses", remove_native)
    with change:
        yield


@torch.no_grad()
def infer(model, refiner, batch, index, with_ablation=False):
    initial, _ = infer_outputs(model, refiner, batch, index)
    if not with_ablation:
        for arm in STATES:
            yield arm, initial[arm]
        return

    native = initial["native"][1]
    for arm in ABLATION_STATES:
        if arm in initial:
            yield arm, initial[arm]
            continue
        if arm == "without_fine_interaction":
            with patch.object(model.interaction, "forward", lambda ref, src, *args: (ref, src)):
                base = model(batch)
            candidate = refiner.refine(
                base, model.optimal_transport, model.fine_matching,
                hypothesis_seed=7352 + index * 1009, sampling_seed=index + 75001,
            )
        else:
            base = native
            with intervention(refiner, arm):
                candidate = refiner.refine(
                    base, model.optimal_transport, model.fine_matching,
                    hypothesis_seed=7352 + index * 1009, sampling_seed=index + 75001,
                )
        pose, _ = consistency_output(base, candidate)
        yield arm, (pose, candidate)
        del base, candidate, pose
