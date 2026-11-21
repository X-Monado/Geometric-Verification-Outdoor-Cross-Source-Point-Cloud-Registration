"""Use the unchanged GeoTransformer forward, inserting a fine interaction before OT."""
import torch
from outdoor_registration import bootstrap
from outdoor_registration.model import GeoTransformer
from outdoor_registration.small_cloud import point_to_node_partition
from .fine_interaction import FineInteraction


class JointFineGeoTransformer(GeoTransformer):
    def __init__(self,cfg,kind='geometry'):
        super().__init__(cfg);self.interaction=FineInteraction(kind);self._active=False
        self.backbone.register_forward_hook(self._capture_features)
        self.coarse_matching.register_forward_hook(self._capture_ids)
        self.coarse_target.register_forward_hook(self._capture_ids)
        self.optimal_transport.register_forward_pre_hook(self._before_transport)

    def _capture_features(self,module,args,output):
        if self._active:self._features=output[0]

    def _capture_ids(self,module,args,output):
        if self._active:self._ids=torch.stack([output[0],output[1]],1)

    def _before_transport(self,module,args):
        if not self._active:return None
        batch=self._batch;nf=int(batch['lengths'][1][0]);nc=int(batch['lengths'][-1][0]);points=batch['points'][1];nodes=batch['points'][-1]
        pieces=[]
        for p,n,f,ids in ((points[:nf],nodes[:nc],self._features[:nf],self._ids[:,0]),
                           (points[nf:],nodes[nc:],self._features[nf:],self._ids[:,1])):
            _,_,knn,mask=point_to_node_partition(p,n,128);knn,mask=knn[ids],mask[ids]
            pieces.append((torch.cat([f,torch.zeros_like(f[:1])])[knn],torch.cat([p,torch.zeros_like(p[:1])])[knn],mask))
        (rf,rp,rm),(sf,sp,sm)=pieces
        assert torch.equal(rm,args[1]) and torch.equal(sm,args[2])
        # Bound activation memory during evaluation without changing patch independence.
        if self.training:
            r,s=self.interaction(rf,sf,rp,sp,rm,sm);logits=r@s.transpose(1,2)/16.
        else:
            chunks=[]
            for start in range(0,len(rf),32):
                sl=slice(start,start+32);r,s=self.interaction(rf[sl],sf[sl],rp[sl],sp[sl],rm[sl],sm[sl]);chunks.append(r@s.transpose(1,2)/16.)
            logits=torch.cat(chunks)
        self._last_logits=logits
        return logits,args[1],args[2]

    def forward(self,data):
        self._active=True;self._batch=data
        try:
            result=super().forward(data);result['interaction_logits']=self._last_logits
            return result
        finally:
            self._active=False;self._batch=None;self._features=None;self._ids=None;self._last_logits=None
