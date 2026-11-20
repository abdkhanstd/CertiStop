"""Anytime VO2max predictors used by the experiments (kept in a module so that saved models can be reloaded)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def hgr(seed, **kw):
    # fixed on development splits (seeds 90-99), see dev_predictor.py
    base = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=10.0, random_state=seed)
    base.update(kw)
    return HGR(**base)


class Predictor:
    """Anytime VO2max regressor. With laboratory inputs it predicts the remaining increase above the running
    peak VO2 (delta target), otherwise VO2max directly."""
    def __init__(self, feats, delta_target, seed=0, kind='gbm', quantile=None):
        self.feats, self.dt, self.seed, self.kind, self.q = feats, delta_target, seed, kind, quantile

    def base(self, D):
        return D.vo2_max_sofar.fillna(0).values if self.dt else 0.0

    def fit(self, D):
        y = D.y.values - self.base(D)
        if self.kind == 'gbm':
            self.m = hgr(self.seed) if self.q is None else hgr(self.seed, loss='quantile', quantile=self.q)
        else:
            self.m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
        self.m.fit(D[self.feats], y)
        return self

    def predict(self, D):
        return self.m.predict(D[self.feats]) + self.base(D)
