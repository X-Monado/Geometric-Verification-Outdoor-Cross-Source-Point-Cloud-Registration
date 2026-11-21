"""Fit a shared verifier on training candidates; sequence-held-out head checks.

The feature backbone has seen all training sequences. The six folds validate
only this newly initialized head, not unseen-scene generalization of a pipeline.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from outdoor_registration.data import sha256
from outdoor_adaptation.pose_guided_registration import VerificationHead
from scripts.train_kitti_crosssource_pilot import save


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


@torch.no_grad()
def evaluate(model, records):
    rows = []
    for r in records:
        scores = model(r['x']).numpy()
        selected, old = int(scores.argmax()), int(r['old'].argmax())
        row = {k:r[k] for k in ('domain', 'sequence', 'pair_id', 'metadata_index')}
        for arm, index in [('new', selected), ('old', old), ('native', 0)]:
            rr, tt = float(r['rr'][index]), float(r['tt'][index])
            row[arm] = dict(index=index, rre_deg=rr, rte_m=tt,
                            success=rr<15 and tt<2, strict=rr<2 and tt<.5)
        row['oracle_success'] = bool(r['y'].any())
        rows.append(row)
    return rows


def summarize(rows):
    result = {}
    for domain in sorted({r['domain'] for r in rows}):
        rs = [r for r in rows if r['domain']==domain]
        result[domain] = dict(pairs=len(rs), oracle=sum(r['oracle_success'] for r in rs),
            arms={a:dict(success=sum(r[a]['success'] for r in rs),
                         strict=sum(r[a]['strict'] for r in rs)) for a in ['new','old','native']})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fold-steps', type=int, default=400); p.add_argument('--full-steps', type=int, default=800)
    p.add_argument('--indices-from', type=Path, default=Path('releases/cross3dreg_best_20261031/weights/7352/verifier.pt'))
    a = p.parse_args(); assert not a.output.exists(); a.output.mkdir(parents=True)
    torch.set_num_threads(2); start = time.time()
    assert json.loads((a.data/'completion.json').read_text())['status']=='complete'
    source_records = json.loads((a.data/'records.json').read_text())
    records = []
    for r in source_records:
        assert sha256(r['path'])==r['sha256']
        with np.load(r['path']) as z:
            x = z['features'].astype(np.float32); rr, tt, old = z['rre'].copy(), z['rte'].copy(), z['old_logits'].copy()
        y = (rr<15)&(tt<2); positive = np.flatnonzero(y); negative = np.flatnonzero(~y)
        hard = negative[np.argsort(old[negative])[-128:]]
        records.append(dict(r, x=torch.from_numpy(x), y=torch.from_numpy(y.astype(np.float32)),
                            rr=rr, tt=tt, old=old, positive=positive, negative=negative, hard_negative=hard))
    indices = torch.load(a.indices_from, map_location='cpu', weights_only=False)['model']['indices']
    protocol = dict(data=str(a.data.resolve()), data_protocol_sha256=sha256(a.data/'protocol.json'),
                    records_sha256=sha256(a.data/'records.json'), source_sha256=sha256(__file__),
                    primary=[15,2], strict=[2,.5], fold_steps=a.fold_steps, full_steps=a.full_steps,
                    objective='Full-pool positive-mass NLL plus0.5 pair-balanced BCE;8 pairs/update',
                    indices_from_sha256=sha256(a.indices_from), fresh_initialization=True,
                    fold_scope='Head-only sequence-held-out; backbone has seen these train sequences.',
                    training_splits_only=True, validation_access=False, test_access=False,
                    interpretation='Ranking logits are not calibrated success probabilities.')
    save(a.output/'protocol.json', protocol)
    oof = []; folds = []
    for sequence in range(6):
        train = [r for r in records if r['sequence']!=sequence]
        held = [r for r in records if r['sequence']==sequence]
        assert train and held and not ({r['sequence'] for r in train}&{r['sequence'] for r in held})
        model, trace = fit(train, indices, a.fold_steps, 7351+sequence)
        rows = evaluate(model, held); oof.extend(rows)
        folds.append(dict(sequence=sequence, train_pairs=len(train), held_pairs=len(held),
                          summary=summarize(rows), loss_trace=trace))
        save(a.output/'head_oof.json', dict(summary=summarize(oof), folds=folds, records=oof))
        print('HEAD_FOLD',sequence,json.dumps(folds[-1]['summary']),flush=True)
    model, trace = fit(records, indices, a.full_steps, 7351)
    checkpoint = a.output/'verifier.pt'
    torch.save(dict(model=model.state_dict(), protocol=protocol, loss_trace=trace), checkpoint)
    # Check compatibility with the actual deployed verifier loader.
    restored = VerificationHead(torch.load(checkpoint, map_location='cpu', weights_only=False)['model']).eval()
    with torch.no_grad():
        assert torch.equal(model(records[0]['x'][:100]), restored(records[0]['x'][:100]))
    save(a.output/'completion.json', dict(status='complete', seconds=time.time()-start,
         head_oof=summarize(oof), training_fit=summarize(evaluate(model, records)),
         checkpoint_sha256=sha256(checkpoint), validation_access=False, test_access=False))
    print('HEAD_COMPLETE', json.dumps(summarize(oof)), flush=True)


if __name__=='__main__':
    main()
