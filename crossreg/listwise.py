import torch
import numpy as np
from torch import nn
class TrainHead(nn.Module):
    def __init__(self, mean, std, indices):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.register_buffer('indices', indices.clone())
        self.net = nn.Sequential(nn.Linear(len(indices), 64), nn.GELU(),
                                 nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(((x-self.mean)/self.std).clamp(-6, 6)[:, self.indices]).squeeze(-1)

def fit(records, indices, steps, seed):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    mean = torch.stack([r['x'].mean(0) for r in records]).mean(0)
    second = torch.stack([r['x'].square().mean(0) for r in records]).mean(0)
    std = (second-mean.square()).clamp_min(1e-6).sqrt()
    model = TrainHead(mean, std, indices)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_trace = []
    for step in range(steps):
        batch = [records[i] for i in rng.choice(len(records), 8, replace=False)]
        terms = []
        for r in batch:
            score = model(r['x']); good = r['y'].bool(); bce = []
            if good.any(): bce.append(nn.functional.softplus(-score[good]).mean())
            if (~good).any(): bce.append(nn.functional.softplus(score[~good]).mean())
            rank = score.logsumexp(0)-score[good].logsumexp(0) if good.any() else score.sum()*0
            terms.append(rank+.5*torch.stack(bce).mean())
        loss = torch.stack(terms).mean()
        assert torch.isfinite(loss), 'nonfinite verifier loss'
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        if step % 100 == 0 or step == steps-1:
            loss_trace.append(dict(step=step+1, loss=float(loss)))
            print('LISTWISE_TRAIN', seed, step+1, float(loss), flush=True)
    return model.eval(), loss_trace