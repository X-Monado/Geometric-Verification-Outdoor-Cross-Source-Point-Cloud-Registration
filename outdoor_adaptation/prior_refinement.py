"""Retain a verified initialization during robust fitting of rematched correspondences."""
import math
import torch
from .pose_guided_registration import PoseGuidedRefiner,fit


def refit_from_prior(prior,ref,src,weights,radius):
    pose=prior.clone();used=0
    for _ in range(5):
        mask=(src@pose[:3,:3].T+pose[:3,3]-ref).norm(dim=1)<radius
        used=int(mask.sum())
        if used<3:break
        pose=fit(src[mask],ref[mask],weights[mask])
    return pose,used


class PriorRefitRefiner(PoseGuidedRefiner):
    def __init__(self,*args,refit_radius=4.8,refit_weights='ot',**kwargs):
        super().__init__(*args,**kwargs)
        if not math.isfinite(refit_radius) or refit_radius<=0:raise ValueError('refit_radius must be positive')
        if refit_weights not in ('ot','uniform'):raise ValueError('refit_weights must be ot or uniform')
        self.refit_radius=float(refit_radius);self.refit_weights=refit_weights

    @torch.no_grad()
    def refine(self,out,optimal_transport,matcher,*,hypothesis_seed,sampling_seed):
        result=super().refine(out,optimal_transport,matcher,hypothesis_seed=hypothesis_seed,sampling_seed=sampling_seed)
        weights=result['corr_scores'] if self.refit_weights=='ot' else torch.ones_like(result['corr_scores'])
        pose,support=refit_from_prior(result['pose_prior'],result['ref_corr_points'],result['src_corr_points'],weights,self.refit_radius)
        result.update(pre_refit_transform=result['estimated_transform'],estimated_transform=pose,refit_last_iteration_support=support)
        return result
