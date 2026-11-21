import numpy as np
from scipy.spatial import cKDTree
from outdoor_registration.data import pose_errors
def stats(values):
    x=np.asarray(values,dtype=float)
    if len(x)==0:return dict(n=0,mean=None,median=None,p90=None)
    assert np.isfinite(x).all()
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p90=float(np.percentile(x,90)))

def correspondence_metrics(ref,src,weights,truth,estimate):
    ref=np.asarray(ref,dtype=np.float64);src=np.asarray(src,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    gt_dist=np.linalg.norm(src@truth[:3,:3].T+truth[:3,3]-ref,axis=1)
    self_dist=np.linalg.norm(src@estimate[:3,:3].T+estimate[:3,3]-ref,axis=1)
    near=gt_dist<.6;points=src[near];span=0.;ratios=[0.,0.]
    if len(points)>=3:
        centered=points-points.mean(0);sv=np.linalg.svd(centered,compute_uv=False)
        span=float(np.sqrt(np.mean(np.sum(centered**2,axis=1))))
        ratios=[float(sv[1]/max(sv[0],1e-8)),float(sv[2]/max(sv[1],1e-8))]
    result=dict(count=len(src),inlier_src_span_m=span,sigma2_over_sigma1=ratios[0],sigma3_over_sigma2=ratios[1],gt_residual=stats(gt_dist),self_residual=stats(self_dist))
    for radius,name in ((.3,'03'),(.6,'06'),(1.,'10'),(1.2,'12')):
        mask=gt_dist<radius;result['inliers'+name]=int(mask.sum())
        result['IR'+name]=float(mask.mean()) if len(mask) else 0.
        result['weighted_IR'+name]=float(w[mask].sum()/max(w.sum(),1e-12))
    result['self_IR06']=float((self_dist<.6).mean()) if len(src) else 0.
    return result

def pose_metrics(pose,truth):
    rr,tt=pose_errors(np.asarray(pose,dtype=np.float64),np.asarray(truth,dtype=np.float64))
    return dict(rre_deg=rr,rte_m=tt,success=rr<15 and tt<2,success5=rr<5 and tt<2,success_strict=rr<2 and tt<.5)