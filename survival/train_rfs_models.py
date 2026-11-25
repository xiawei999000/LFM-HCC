"""
RFS prediction model training script.
Usage: python train_rfs_models.py
Input: feature tables (see config.py) + clinical data
Output: model weight files under RFS_model/

Paper (Sec. 4.5):
  - EVC1 (Split == 'ext1') is the RFS model development cohort; EVC2
    (Split == 'ext2') is the independent validation cohort.
  - For LFM-derived features, univariable Cox regression is first performed
    for feature pre-screening, and retained features are entered into an
    elastic-net Cox proportional hazards model.
  - The penalty parameter and mixing parameter are selected by five-fold
    cross-validation according to the concordance index. Feature
    pre-screening is performed WITHIN EACH FOLD using only the training
    partitions; imputation and scaling are likewise fitted within each
    fold's training partition, and the validation partition is only
    transformed.
  - The final model is refitted on the entire EVC1 cohort with the selected
    hyperparameters. EVC2 is not involved in feature selection,
    hyperparameter tuning, or cutoff determination.
  - Two clinical reference models are built with the same elastic-net Cox
    framework WITHOUT univariable pre-screening:
      * Clinical model: 8 preoperative variables (age, sex, tumor size,
        HBsAg, AFP, ALT, AST, GGT).
      * Clinicopathological model: additionally incorporates the
        postoperative pathological variables (MVI and pathological grade).
"""
import pandas as pd, numpy as np, os, sys, json, joblib, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from config import DATA_DIR, FEATURE_TABLES, CLINICAL_INFO_PATH

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RFS_model')
os.makedirs(MODEL_DIR, exist_ok=True)

# Clinical model: eight preoperative variables (paper Sec. 4.5).
CLIN_COLS = ['Sex', 'Age', 'HBsAg', 'AFP', 'ALT', 'AST', 'GGT', 'tumor_size']
# Clinicopathological model: additionally MVI and pathological grade.
CLINICOPATH_COLS = CLIN_COLS + ['MVI', 'PathologicalGrade']

CLINICAL_MODELS = {
    'Clinical': CLIN_COLS,
    'Clinicopathological': CLINICOPATH_COLS,
}

alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


def univariate_cox_screen(X, df_surv, feature_names=None, fold_id=None,
                          n_min=50, n_max=100):
    """
    Univariable Cox regression pre-screening.
    Returns indices of retained features (univariable p < 0.05; falls back to
    the top-n_min by p-value if none pass; capped at n_max).
    Features whose univariable fit fails are assigned p = 1.0 and logged.
    """
    pvals = []
    failures = []
    for j in range(X.shape[1]):
        try:
            cph = CoxPHFitter()
            temp = pd.DataFrame({'RFS': df_surv['RFS'].values,
                                 'RFS_status': df_surv['RFS_status'].values,
                                 'feat': X[:, j]})
            cph.fit(temp, duration_col='RFS', event_col='RFS_status')
            pvals.append(cph.summary.loc['feat', 'p'])
        except Exception as e:
            pvals.append(1.0)
            name = feature_names[j] if feature_names is not None else f'feat_{j}'
            failures.append((name, type(e).__name__, str(e)[:100]))
    if failures:
        where = f'fold {fold_id}: ' if fold_id is not None else ''
        print(f'    [WARN] Univariable screening {where}'
              f'{len(failures)}/{X.shape[1]} features failed (assigned p=1.0):')
        for name, etype, emsg in failures[:10]:
            print(f'           - {name}: {etype}: {emsg}')
    pvals = np.array(pvals)
    sel_idx = np.where(pvals < 0.05)[0]
    if len(sel_idx) == 0:
        sel_idx = np.argsort(pvals)[:n_min]
    elif len(sel_idx) > n_max:
        sel_idx = sel_idx[np.argsort(pvals[sel_idx])[:n_max]]
    return sel_idx


def _prepare_folds(X_raw, y, df_surv, feature_names, screen_within_fold):
    """
    Per-fold preprocessing: imputation and scaling are fitted on the training
    partition only, and (for deep features) univariable Cox pre-screening is
    performed within the fold using only the training partition. Validation
    partitions are transformed only.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_raw)):
        imp = SimpleImputer(strategy='mean')
        X_tr = imp.fit_transform(X_raw[tr_idx])
        X_va = imp.transform(X_raw[va_idx])
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        if screen_within_fold:
            sel = univariate_cox_screen(X_tr, df_surv.iloc[tr_idx],
                                        feature_names=feature_names,
                                        fold_id=fold_id)
        else:
            sel = np.arange(X_raw.shape[1])
        folds.append((tr_idx, va_idx, X_tr, X_va, sel))
    return folds


def cv_select_elastic_net(X_raw, y, df_surv, feature_names=None,
                          screen_within_fold=True):
    """
    Five-fold cross-validation of elastic-net Cox hyperparameters by C-index.
    All preprocessing (imputation, scaling, univariable pre-screening) is
    fitted within each fold using only the training partition (paper
    Sec. 4.5). Returns (best_alpha, best_l1_ratio, best_score).
    """
    folds = _prepare_folds(X_raw, y, df_surv, feature_names,
                           screen_within_fold)

    best_score, best_a, best_l1 = -np.inf, 0.01, 0.1
    for a in alphas:
        for l1 in l1_ratios:
            scores = []
            n_fail = 0
            for fold_id, (tr_idx, va_idx, X_tr, X_va, sel) in enumerate(folds):
                try:
                    m = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=[a],
                                               max_iter=10000)
                    m.fit(X_tr[:, sel], y[tr_idx])
                    p = m.predict(X_va[:, sel])
                    c = concordance_index_censored(
                        y[va_idx]['RFS_status'], y[va_idx]['RFS'], p)[0]
                    scores.append(c)
                except Exception as e:
                    n_fail += 1
                    scores.append(0.5)
                    print(f'    [WARN] alpha={a}, l1_ratio={l1}, fold {fold_id} '
                          f'failed: {type(e).__name__}: {str(e)[:100]}')
            if n_fail:
                print(f'    [WARN] alpha={a}, l1_ratio={l1}: '
                      f'{n_fail}/{len(folds)} folds failed (scored 0.5)')
            mean_c = np.mean(scores)
            if mean_c > best_score:
                best_score, best_a, best_l1 = mean_c, a, l1
    return best_a, best_l1, best_score


def save_model(model_dir, model_name, model_type, feat_file, selected_features,
               imp, scaler, coxnet, best_a, best_l1, c_train, c_test,
               method=None):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(imp, os.path.join(model_dir, 'imputer.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(coxnet, os.path.join(model_dir, 'cox_model.joblib'))
    n_nz = int(np.sum(np.abs(coxnet.coef_) > 1e-6))
    cfg = {
        'model_name': model_name,
        'model_type': model_type,
        'selected_features': list(selected_features),
        'n_selected': len(selected_features),
        'n_nonzero': n_nz,
        'alpha': best_a,
        'l1_ratio': best_l1,
        'train_c_index': round(c_train, 4),
        'test_c_index': round(c_test, 4),
    }
    if feat_file is not None:
        cfg['feature_file'] = feat_file
    if method is not None:
        cfg['method'] = method
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'  Saved to: {model_dir}')
    return n_nz


def main():
    clinical = pd.read_excel(CLINICAL_INFO_PATH)
    DEEP_MODELS = FEATURE_TABLES

    from sksurv.metrics import cumulative_dynamic_auc

    # ============================================================
    # 1. Deep feature models
    # ============================================================
    for model_name, feat_file in DEEP_MODELS.items():
        print(f'{"="*60}')
        print(f'Training: {model_name}')

        feats = pd.read_excel(os.path.join(DATA_DIR, feat_file))
        feature_cols = [c for c in feats.columns if c not in ['Dataset', 'Split', 'ID']]
        data = feats.merge(clinical[['Split', 'ID', 'RFS', 'RFS_status']],
                           on=['Split', 'ID'], how='inner')

        train = data[data['Split'] == 'ext1'].reset_index(drop=True)
        X_train_raw = train[feature_cols].values.astype(float)
        y_train = Surv.from_dataframe('RFS_status', 'RFS',
                                       train[['RFS_status', 'RFS']])

        # 5-fold CV to select alpha/l1_ratio; imputation, scaling and univariable
        # Cox pre-screening are all fitted within each fold's training partition.
        best_a, best_l1, best_score = cv_select_elastic_net(
            X_train_raw, y_train, train, feature_names=feature_cols,
            screen_within_fold=True)
        print(f'  CV best: alpha={best_a}, l1_ratio={best_l1}, '
              f'score={best_score:.4f}')

        # Final preprocessing and univariable screening on the full development
        # cohort (EVC1) for the refit.
        imp = SimpleImputer(strategy='mean')
        X_train_imp = imp.fit_transform(X_train_raw)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_imp)
        sel_idx = univariate_cox_screen(X_train_s, train, feature_names=feature_cols)
        selected_features = [feature_cols[i] for i in sel_idx]
        X_train_sel = X_train_s[:, sel_idx]
        print(f'  Selected features: {len(selected_features)}')

        # Final model refitted on the entire EVC1 cohort (paper Sec. 4.5).
        # fit_baseline_model=True enables predict_survival_function() at
        # inference time (12/24/36-month survival probabilities).
        coxnet = CoxnetSurvivalAnalysis(l1_ratio=best_l1, alphas=[best_a],
                                        max_iter=10000, fit_baseline_model=True)
        coxnet.fit(X_train_sel, y_train)
        n_nz = np.sum(np.abs(coxnet.coef_) > 1e-6)

        # Evaluate on the independent validation cohort (EVC2)
        test = data[data['Split'] == 'ext2']
        X_test = imp.transform(test[feature_cols].values.astype(float))
        X_test_s = scaler.transform(X_test)[:, sel_idx]
        y_test = Surv.from_dataframe('RFS_status', 'RFS', test[['RFS_status', 'RFS']])
        risk_test = coxnet.predict(X_test_s)
        c_test = concordance_index_censored(y_test['RFS_status'], y_test['RFS'], risk_test)[0]

        risk_train = coxnet.predict(X_train_sel)
        c_train = concordance_index_censored(y_train['RFS_status'], y_train['RFS'], risk_train)[0]

        print(f'  Train C-index: {c_train:.4f}, Test C-index: {c_test:.4f}')
        print(f'  Non-zero coefs: {n_nz}')

        save_model(os.path.join(MODEL_DIR, model_name), model_name,
                   'deep_features', feat_file, selected_features,
                   imp, scaler, coxnet, best_a, best_l1, c_train, c_test)


    # ============================================================
    # 2. Clinical reference models (elastic-net Cox, no univariable
    #    pre-screening; paper Sec. 4.5)
    # ============================================================
    for clin_name, clin_cols in CLINICAL_MODELS.items():
        print(f'\n{"="*60}')
        print(f'Training: {clin_name} (Elastic Net, no pre-filtering)')

        train_c = clinical[clinical['Split'] == 'ext1'].reset_index(drop=True)
        X_tr_raw = train_c[clin_cols].values.astype(float)
        y_tr = Surv.from_dataframe('RFS_status', 'RFS', train_c[['RFS_status', 'RFS']])
        print(f'  Features: {len(clin_cols)} (all used directly): {clin_cols}')

        best_a, best_l1, best_score = cv_select_elastic_net(
            X_tr_raw, y_tr, train_c, feature_names=clin_cols,
            screen_within_fold=False)
        print(f'  CV best: alpha={best_a}, l1_ratio={best_l1}, score={best_score:.4f}')

        imp = SimpleImputer(strategy='mean')
        X_tr_imp = imp.fit_transform(X_tr_raw)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)

        coxnet = CoxnetSurvivalAnalysis(l1_ratio=best_l1, alphas=[best_a],
                                        max_iter=10000, fit_baseline_model=True)
        coxnet.fit(X_tr_s, y_tr)
        n_nz = int(np.sum(np.abs(coxnet.coef_) > 1e-6))

        print(f'  Coefficients ({n_nz}/{len(clin_cols)} non-zero):')
        for i, c in enumerate(clin_cols):
            coef = float(coxnet.coef_[i])
            mark = '' if abs(coef) > 1e-6 else ' (zeroed)'
            print(f'    {c:25s}: {coef:+.4f}{mark}')

        # Evaluate on EVC2
        test_c = clinical[clinical['Split'] == 'ext2']
        X_te = scaler.transform(imp.transform(test_c[clin_cols].values.astype(float)))
        y_te = Surv.from_dataframe('RFS_status', 'RFS', test_c[['RFS_status', 'RFS']])
        risk_test = coxnet.predict(X_te)
        c_test = concordance_index_censored(y_te['RFS_status'], y_te['RFS'], risk_test)[0]

        risk_train = coxnet.predict(X_tr_s)
        c_train = concordance_index_censored(y_tr['RFS_status'], y_tr['RFS'], risk_train)[0]

        print(f'  Train C-index: {c_train:.4f}, Test C-index: {c_test:.4f}')

        # Time-dep AUC
        for t in [12, 24, 36]:
            try:
                aucs, _ = cumulative_dynamic_auc(y_tr, y_te, risk_test, [t])
                print(f'  AUC_{t}m: {aucs[0]:.4f}')
            except Exception as e:
                print(f'  AUC_{t}m: N/A ({type(e).__name__})')

        save_model(os.path.join(MODEL_DIR, clin_name), clin_name, 'clinical',
                   None, clin_cols, imp, scaler, coxnet, best_a, best_l1,
                   c_train, c_test,
                   method='Elastic Net Cox, no univariate pre-filtering')

    print(f'\n{"="*60}')
    print(f'All models saved to: {MODEL_DIR}')
    print('Done!')


if __name__ == "__main__":
    main()
