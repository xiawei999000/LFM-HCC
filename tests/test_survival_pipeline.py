"""
Smoke tests for the RFS survival pipeline (survival/train_rfs_models.py +
inference/infer_rfs.py) using small synthetic data.

Covers:
  - univariable Cox pre-screening keeps a strong signal feature
  - cross-validated elastic-net hyperparameter selection runs end-to-end
  - a final CoxnetSurvivalAnalysis fitted with fit_baseline_model=True yields
    valid survival probabilities in [0, 1]
  - save_model -> RFSInferencer round trip (risk scores + survival prediction)
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'survival'))
sys.path.insert(0, os.path.join(ROOT, 'inference'))

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import train_rfs_models as trm
from infer_rfs import RFSInferencer


def _synthetic(n=80, n_feat=30, seed=0):
    """Synthetic cohort where feature 0 carries a strong survival signal."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    latent = rng.exponential(scale=40.0, size=n) / np.exp(X[:, 0])
    censor = rng.uniform(10.0, 60.0, size=n)
    rfs = np.minimum(latent, censor)
    status = (latent <= censor).astype(int)
    df_surv = pd.DataFrame({'RFS': rfs, 'RFS_status': status})
    y = Surv.from_arrays(status.astype(bool), rfs,
                         name_event='RFS_status', name_time='RFS')
    feat_names = [f'feat_{i}' for i in range(n_feat)]
    return X, y, df_surv, feat_names


def test_univariate_screen_keeps_signal_feature():
    X, y, df_surv, feat_names = _synthetic()
    sel = trm.univariate_cox_screen(X, df_surv, feature_names=feat_names,
                                    fold_id=0)
    assert isinstance(sel, np.ndarray)
    assert 0 in sel, 'strong signal feature must survive screening'
    assert len(sel) <= 100


def test_cv_select_elastic_net_runs(monkeypatch):
    # Keep the smoke test fast: shrink the hyperparameter grid.
    monkeypatch.setattr(trm, 'alphas', [0.01, 0.1])
    monkeypatch.setattr(trm, 'l1_ratios', [0.5])
    X, y, df_surv, feat_names = _synthetic()
    best_a, best_l1, best_score = trm.cv_select_elastic_net(
        X, y, df_surv, feature_names=feat_names, screen_within_fold=True)
    assert best_a in (0.01, 0.1)
    assert best_l1 == 0.5
    assert 0.0 <= best_score <= 1.0


def test_final_model_survival_function_in_unit_interval():
    """Final model with fit_baseline_model=True must give S(t) in [0, 1]."""
    X, y, df_surv, feat_names = _synthetic()
    imp = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imp.fit_transform(X))
    model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[0.01],
                                   max_iter=10000, fit_baseline_model=True)
    model.fit(Xs, y)
    surv = model.predict_survival_function(Xs, return_array=True)
    assert surv.shape[0] == X.shape[0]
    assert np.all(surv >= 0.0) and np.all(surv <= 1.0)


def test_save_model_and_inferencer_round_trip(tmp_path):
    X, y, df_surv, feat_names = _synthetic()
    imp = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    Xs = scaler.fit_transform(imp.fit_transform(X))
    model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[0.01],
                                   max_iter=10000, fit_baseline_model=True)
    model.fit(Xs, y)
    risk = model.predict(Xs)
    c = concordance_index_censored(y['RFS_status'], y['RFS'], risk)[0]

    model_root = tmp_path / 'RFS_model'
    trm.save_model(str(model_root / 'TestModel'), 'TestModel', 'deep',
                   'synthetic.xlsx', feat_names, imp, scaler, model,
                   0.01, 0.5, c, c)

    infer = RFSInferencer(model_dir=str(model_root))
    assert infer.model_names == ['TestModel']

    data = pd.DataFrame(X, columns=feat_names)
    scores = infer.predict(data)
    assert 'TestModel' in scores
    assert scores['TestModel'].shape == (X.shape[0],)
    assert np.all(np.isfinite(scores['TestModel']))

    surv = infer.predict_survival(data, [12, 24], model_name='TestModel')
    df_s = surv['TestModel']
    assert list(df_s.columns) == ['12m', '24m']
    assert np.all(df_s.values >= 0.0) and np.all(df_s.values <= 1.0)
