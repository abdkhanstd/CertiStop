"""Predictor hyper-parameter study on development splits (seeds 90-99, never used for reported results)."""
import os, sys, itertools
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import load, split_ids
from models import Predictor
import models

GRID = list(itertools.product([30, 100, 300], [150, 300], [15, 31], [1.0, 10.0]))


def one(ds, rep):
    P, FEATS = load(ds); tr, ca, te = split_ids(P, rep); out = []
    for inp, feats in FEATS.items():
        TR = P[P.ID.isin(tr)]; TE = P[P.ID.isin(te)]
        for dt in ([True, False] if inp == 'full' else [False]):
            for msl, it, mln, l2 in GRID:
                p = Predictor(feats, dt, seed=rep)
                p.m = None
                from sklearn.ensemble import HistGradientBoostingRegressor as HGR
                y = TR.y.values - p.base(TR)
                m = HGR(max_iter=it, learning_rate=0.05, max_leaf_nodes=mln, min_samples_leaf=msl, l2_regularization=l2, random_state=rep).fit(TR[feats], y)
                yh = m.predict(TE[feats]) + p.base(TE)
                out.append(dict(ds=ds, rep=rep, input=inp, delta_target=dt, msl=msl, it=it, mln=mln, l2=l2, mae=np.abs(yh - TE.y.values).mean()))
    return pd.DataFrame(out)


if __name__ == '__main__':
    from joblib import Parallel, delayed
    R = pd.concat(Parallel(n_jobs=20)(delayed(one)(ds, r) for ds in ('malaga', 'dublin') for r in range(90, 100)))
    R.to_csv('results/dev/predictor_grid.csv', index=False)
    g = R.groupby(['ds', 'input', 'delta_target', 'msl', 'it', 'mln', 'l2']).mae.mean().reset_index()
    pd.set_option('display.width', 200)
    for k, x in g.groupby(['ds', 'input', 'delta_target']):
        print(k, 'default(30,300,31,1)=', x[(x.msl == 30) & (x.it == 300) & (x.mln == 31) & (x.l2 == 1)].mae.round(3).values)
        print(x.sort_values('mae').head(5).round(3).to_string())
    avg = g.groupby(['msl', 'it', 'mln', 'l2']).mae.mean().sort_values()
    print('rank-avg over all settings'); print(g.assign(r=g.groupby(['ds', 'input', 'delta_target']).mae.rank()).groupby(['msl', 'it', 'mln', 'l2']).r.mean().sort_values().head(8))
