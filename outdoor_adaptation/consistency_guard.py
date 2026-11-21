"""Conservative native/feedback agreement check without labels or domain IDs."""
import torch


@torch.no_grad()
def guarded_pose(native,prior,feedback,candidate,source):
    assert len(source)>0
    ids=torch.linspace(0,len(source)-1,min(1024,len(source)),device=source.device).long()
    points=source[ids]
    def moved(pose):return points@pose[:3,:3].T+pose[:3,3]
    native_feedback=float((moved(native)-moved(feedback)).norm(dim=1).median())
    prior_feedback=float((moved(prior)-moved(feedback)).norm(dim=1).median())
    # Both constants already belong to the registration pipeline: native
    # inlier radius and widest prior-refit radius. No threshold search here.
    fallback=native_feedback<=.6 and prior_feedback>4.8
    return (feedback if fallback else candidate),dict(
        native_feedback_median_displacement_m=native_feedback,
        prior_feedback_median_displacement_m=prior_feedback,
        fallback_to_feedback=bool(fallback),agreement_radius_m=.6,disagreement_radius_m=4.8)
