"""Neural sequence baselines for anytime VO2max prediction (causal GRU and causal Transformer).

They read the raw test signals up to each decision time and are evaluated on exactly the same decision points and
subject-grouped splits as the gradient-boosting predictor in run.py.
Usage: python certistop/seqnn.py --ds malaga --arch gru --reps 20 --gpu 0
"""
import argparse, os, sys, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import load, split_ids


def build_malaga(P, raw='data_raw/malaga'):
    from prep_malaga import clean_test
    m = pd.read_csv(f'{raw}/test_measure.csv'); s = pd.read_csv(f'{raw}/subject-info.csv').set_index('ID_test')
    keep = set(P.ID_test)
    seqs, stat = {}, {}
    for tid, d in m.groupby('ID_test'):
        if tid not in keep:
            continue
        info = s.loc[tid]; d = clean_test(d); w = info['Weight']
        b = (d.time // 10).astype(int)
        g = pd.DataFrame(dict(speed=d.Speed, hr=d.HRs, vo2=d.VO2s / w, vco2=d.VCO2s / w, ve=d.VEs, rr=d.RRs, b=b)).groupby('b').mean()
        g = g.reindex(range(0, int(b.max()) + 1)).ffill()  # causal: leading gaps stay missing
        g['vo2max_sofar'] = g.vo2.cummax()
        seqs[tid] = g.values.astype(np.float32)   # columns: speed hr vo2 vco2 ve rr vo2max_sofar
        stat[tid] = np.array([info['Age'], info['Sex'], info['Height'], info['Weight'], info['Weight'] / (info['Height'] / 100) ** 2], np.float32)
    chans = {'full': [0, 1, 2, 3, 4, 5, 6], 'hr': [0, 1]}
    # decision row r of test uses bins 0..tsec/10-1
    idx = (P.tsec // 10).astype(int).values
    return seqs, stat, chans, idx


def build_dublin(P, info='data/dublin_info.csv', stages='data/dublin_stages.csv'):
    I = pd.read_csv(info).set_index('file'); S = pd.read_csv(stages)
    S.loc[(S.hr < 35) | (S.hr > 230), 'hr'] = np.nan
    keep = set(P.ID_test); seqs, stat = {}, {}
    sp = {'Cycling': 0, 'Running': 1, 'Rowing': 2, 'Kayak': 3}
    for f, st in S.groupby('file'):
        if f not in keep:
            continue
        st = st.sort_values('k'); st = st[st.load.notna()]
        r = I.loc[f]
        vo2 = st.vo2.where((st.vo2 > 2) & (st.vo2 < 95))
        x = pd.DataFrame(dict(load=st.load.values, hr=st.hr.values, vo2=vo2.values, bla=st.bla.values)).ffill()
        x['vo2max_sofar'] = x.vo2.cummax()
        seqs[f] = x.values.astype(np.float32)
        oh = np.eye(4, dtype=np.float32)[sp[r.sport]]
        age = P.loc[P.ID_test == f, 'age'].iloc[0]
        stat[f] = np.concatenate([[age, float(r.sex == 'M'), r.height, r.weight, r.bmi], oh]).astype(np.float32)
    chans = {'full': [0, 1, 2, 3, 4], 'hr': [0, 1]}
    idx = (P.step.values + 1).astype(int)   # stages 0..k inclusive
    return seqs, stat, chans, idx


class Net(nn.Module):
    def __init__(self, cin, cs, arch, h=64):
        super().__init__()
        self.arch = arch
        self.inp = nn.Linear(cin, h); self.st = nn.Sequential(nn.Linear(cs, h), nn.ReLU(), nn.Linear(h, h))
        if arch == 'gru':
            self.core = nn.GRU(h, h, num_layers=2, batch_first=True, dropout=0.1)
        else:
            layer = nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=0.1)
            self.core = nn.TransformerEncoder(layer, 2)
            self.pos = nn.Parameter(torch.zeros(1, 400, h)); nn.init.normal_(self.pos, std=0.02)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, s):
        z = self.inp(x); L = z.shape[1]
        if self.arch == 'gru':
            o, _ = self.core(z)
        else:
            z = z + self.pos[:, :L]
            mask = torch.triu(torch.full((L, L), float('-inf'), device=z.device), 1)
            o = self.core(z.transpose(0, 1), mask=mask).transpose(0, 1)
        e = self.st(s).unsqueeze(1).expand(-1, L, -1)
        return self.head(torch.cat([o, e], -1)).squeeze(-1)


def run(ds, arch, reps, gpu, out):
    P, _ = load(ds)
    seqs, stat, chans, idx = (build_malaga if ds == 'malaga' else build_dublin)(P)
    P = P.assign(_idx=idx)
    dev = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    rows = []
    for rep in reps:
        tr, ca, te = split_ids(P, rep)
        for inp, ch in chans.items():
            t0 = time.time(); torch.manual_seed(rep); np.random.seed(rep)
            dt = inp == 'full'
            trt = sorted(set(P.ID_test[P.ID.isin(tr)])); tet = sorted(set(P.ID_test[P.ID.isin(te)]))
            rng = np.random.default_rng(rep); val = set(rng.choice(trt, size=max(1, len(trt) // 10), replace=False)); fit = [t for t in trt if t not in val]
            X = np.concatenate([seqs[t][:, ch] for t in fit]); mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-6
            Sx = np.stack([stat[t] for t in fit]); smu, ssd = np.nanmean(Sx, 0), np.nanstd(Sx, 0) + 1e-6
            ytr = P[P.ID_test.isin(fit)].groupby('ID_test').y.first(); ymu, ysd = ytr.mean(), ytr.std()

            def batch(tids):
                L = max(len(seqs[t]) for t in tids)
                x = np.zeros((len(tids), L, len(ch)), np.float32); s = np.stack([(stat[t] - smu) / ssd for t in tids])
                base = np.zeros((len(tids), L), np.float32)
                for i, t in enumerate(tids):
                    q = seqs[t]; x[i, :len(q)] = (q[:, ch] - mu) / sd
                    if dt:
                        base[i, :len(q)] = np.nan_to_num(q[:, -1])
                x = np.nan_to_num(x); s = np.nan_to_num(s)
                return torch.tensor(x, device=dev), torch.tensor(s, device=dev), torch.tensor(base, device=dev)

            def targets(tids):
                rr = P[P.ID_test.isin(tids)]
                pos = {t: i for i, t in enumerate(tids)}
                bi = torch.tensor([pos[t] for t in rr.ID_test], device=dev); ti = torch.tensor(rr._idx.values - 1, device=dev)
                return bi, ti, torch.tensor(rr.y.values, dtype=torch.float32, device=dev), rr
            net = Net(len(ch), len(next(iter(stat.values()))), arch).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            vx, vs, vb = batch(sorted(val)); vbi, vti, vy, _ = targets(sorted(val))
            best, bstate, bad = 1e9, None, 0
            for ep in range(200):
                net.train(); perm = np.random.permutation(fit)
                for k in range(0, len(perm), 32):
                    tids = list(perm[k:k + 32]); x, s, b = batch(tids); bi, ti, y, _ = targets(tids)
                    pr = net(x, s)[bi, ti] * ysd + b[bi, ti] + (0 if dt else ymu)
                    loss = nn.functional.smooth_l1_loss(pr, y, beta=2.0)
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                net.eval()
                with torch.no_grad():
                    pv = net(vx, vs)[vbi, vti] * ysd + vb[vbi, vti] + (0 if dt else ymu)
                    v = (pv - vy).abs().mean().item()
                if v < best - 1e-3:
                    best, bstate, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 20:
                        break
            net.load_state_dict(bstate); net.eval()
            os.makedirs(f'weights/{ds}/seqnn', exist_ok=True)
            torch.save(bstate, f'weights/{ds}/seqnn/{arch}_{inp}_rep{rep:02d}.pt')
            with torch.no_grad():
                preds = []
                for k in range(0, len(tet), 64):
                    tids = tet[k:k + 64]; x, s, b = batch(tids); bi, ti, y, rr = targets(tids)
                    pr = net(x, s)[bi, ti] * ysd + b[bi, ti] + (0 if dt else ymu)
                    preds.append(pd.DataFrame(dict(rep=rep, input=inp, model=arch, ID_test=rr.ID_test.values, tsec=rr.tsec.values,
                                                   Tsec=rr.Tsec.values, y=rr.y.values, hr_frac=rr.hr_frac.values, yhat=pr.cpu().numpy())))
            pdf = pd.concat(preds); rows.append(pdf)
            print(ds, arch, rep, inp, f'ep={ep} val={best:.2f} test MAE={np.abs(pdf.yhat - pdf.y).mean():.2f} {time.time() - t0:.0f}s', flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.concat(rows).to_parquet(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True); ap.add_argument('--arch', default='gru'); ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    run(a.ds, a.arch, range(a.start, a.start + a.reps), a.gpu, f'results/{a.ds}/seqnn_{a.arch}.parquet')
