"""
RFS prediction model training script.
Usage: python train_rfs_models.py
Input: feature tables (see config.py) + clinical data
Output: model weight files under RFS_model/
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

clinical = pd.read_excel(CLINICAL_INFO_PATH)

DEEP_MODELS = FEATURE_TABLES
CLIN_COLS = ['Sex', 'Age', 'MVI', 'PathologicalGrade', 'HBsAg', 'AFP',
             'ALT', 'AST', 'GGT', 'tumor_size']

alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

# ============================================================
# 1. Deep feature models
# ============================================================
for model_name, feat_file in DEEP_MODELS.items():
    print(f'{"="*60}')
    print(f'Training: {model_name}')

    feats = pd.read_excel(os.path.join(BASE_DIR, feat_file))
    feature_cols = [c for c in feats.columns if c not in ['Dataset', 'Split', 'ID']]
    data = feats.merge(clinical[['Split', 'ID', 'RFS', 'RFS_status']],
                       on=['Split', 'ID'], how='inner')

    train = data[data['Split'] == 'ext1']
    X_train = train[feature_cols].values.astype(float)

    # Imputation & scaling (computed on training set)
    imp = SimpleImputer(strategy='mean')
    X_train_imp = imp.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)

    y_train = Surv.from_dataframe('RFS_status', 'RFS',
                                   train[['RFS_status', 'RFS']])

    # Univariate Cox screening (training set only)
    pvals = []
    for j in range(X_train_s.shape[1]):
        try:
            cph = CoxPHFitter()
            temp = pd.DataFrame({'RFS': train['RFS'], 'RFS_status': train['RFS_status'],
                                 'feat': X_train_s[:, j]})
            cph.fit(temp, duration_col='RFS', event_col='RFS_status')
            pvals.append(cph.summary.loc['feat', 'p'])
        except:
            pvals.append(1.0)
    pvals = np.array(pvals)
    sel_idx = np.where(pvals < 0.05)[0]
    if len(sel_idx) == 0:
        sel_idx = np.argsort(pvals)[:50]
    elif len(sel_idx) > 100:
        sel_idx = sel_idx[np.argsort(pvals[sel_idx])[:100]]

    selected_features = [feature_cols[i] for i in sel_idx]
    X_train_sel = X_train_s[:, sel_idx]
    print(f'  Selected features: {len(selected_features)}')

    # 5-fold CV to select alpha
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_score, best_a, best_l1 = -np.inf, 0.01, 0.1
    for a in alphas:
        for l1 in l1_ratios:
            scores = []
            for tr_idx, va_idx in kf.split(X_train_sel):
                try:
                    m = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=[a], max_iter=10000)
                    m.fit(X_train_sel[tr_idx], y_train[tr_idx])
                    p = m.predict(X_train_sel[va_idx])
                    c = concordance_index_censored(
                        y_train[va_idx]['RFS_status'], y_train[va_idx]['RFS'], p)[0]
                    scores.append(c)
                except:
                    scores.append(0.5)
            mean_c = np.mean(scores)
            if mean_c > best_score:
                best_score, best_a, best_l1 = mean_c, a, l1

    # Train model
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=best_l1, alphas=[best_a], max_iter=10000)
    coxnet.fit(X_train_sel, y_train)
    n_nz = np.sum(np.abs(coxnet.coef_) > 1e-6)

    # Evaluate
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

    # Save model
    model_dir = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(imp, os.path.join(model_dir, 'imputer.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(coxnet, os.path.join(model_dir, 'cox_model.joblib'))
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump({
            'model_name': model_name,
            'model_type': 'deep_features',
            'feature_file': feat_file,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'n_nonzero': int(n_nz),
            'alpha': best_a,
            'l1_ratio': best_l1,
            'train_c_index': round(c_train, 4),
            'test_c_index': round(c_test, 4),
        }, f, indent=2)
    print(f'  Saved to: {model_dir}')


# ============================================================
    # 2. Clinical model (Elastic Net only, no univariate pre-screening)
# ============================================================
print(f'\n{"="*60}')
print('Training: Clinical (Elastic Net, no pre-filtering)')

train_c = clinical[clinical['Split'] == 'ext1']
X_tr = train_c[CLIN_COLS].values.astype(float)

imp = SimpleImputer(strategy='mean')
X_tr_imp = imp.fit_transform(X_tr)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr_imp)
y_tr = Surv.from_dataframe('RFS_status', 'RFS', train_c[['RFS_status', 'RFS']])
print(f'  Features: {len(CLIN_COLS)} (all used directly)')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_score, best_a, best_l1 = -np.inf, 0.01, 0.1
for a in alphas:
    for l1 in l1_ratios:
        scores = []
        for tr_idx, va_idx in kf.split(X_tr_s):
            try:
                m = CoxnetSurvivalAnalysis(l1_ratio=l1, alphas=[a], max_iter=10000)
                m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
                p = m.predict(X_tr_s[va_idx])
                c = concordance_index_censored(
                    y_tr[va_idx]['RFS_status'], y_tr[va_idx]['RFS'], p)[0]
                scores.append(c)
            except:
                scores.append(0.5)
        mean_c = np.mean(scores)
        if mean_c > best_score:
            best_score, best_a, best_l1 = mean_c, a, l1

print(f'  CV best: alpha={best_a}, l1_ratio={best_l1}, score={best_score:.4f}')

coxnet = CoxnetSurvivalAnalysis(l1_ratio=best_l1, alphas=[best_a], max_iter=10000)
coxnet.fit(X_tr_s, y_tr)
n_nz = int(np.sum(np.abs(coxnet.coef_) > 1e-6))

# Show coefficients
print(f'  Coefficients ({n_nz}/{len(CLIN_COLS)} non-zero):')
for i, c in enumerate(CLIN_COLS):
    coef = float(coxnet.coef_[i])
    mark = '' if abs(coef) > 1e-6 else ' (zeroed)'
    print(f'    {c:25s}: {coef:+.4f}{mark}')

# Evaluate
test_c = clinical[clinical['Split'] == 'ext2']
X_te = scaler.transform(imp.transform(test_c[CLIN_COLS].values.astype(float)))
y_te = Surv.from_dataframe('RFS_status', 'RFS', test_c[['RFS_status', 'RFS']])
risk_test = coxnet.predict(X_te)
c_test = concordance_index_censored(y_te['RFS_status'], y_te['RFS'], risk_test)[0]

risk_train = coxnet.predict(X_tr_s)
c_train = concordance_index_censored(y_tr['RFS_status'], y_tr['RFS'], risk_train)[0]

print(f'  Train C-index: {c_train:.4f}, Test C-index: {c_test:.4f}')

# Time-dep AUC
from sksurv.metrics import cumulative_dynamic_auc
for t in [12, 24, 36]:
    try:
        aucs, _ = cumulative_dynamic_auc(y_tr, y_te, risk_test, [t])
        print(f'  AUC_{t}m: {aucs[0]:.4f}')
    except:
        print(f'  AUC_{t}m: N/A')

model_dir = os.path.join(MODEL_DIR, 'Clinical')
os.makedirs(model_dir, exist_ok=True)
joblib.dump(imp, os.path.join(model_dir, 'imputer.joblib'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
joblib.dump(coxnet, os.path.join(model_dir, 'cox_model.joblib'))
with open(os.path.join(model_dir, 'config.json'), 'w') as f:
    json.dump({
        'model_name': 'Clinical',
        'model_type': 'clinical',
        'method': 'Elastic Net Cox, no univariate pre-filtering',
        'selected_features': CLIN_COLS,
        'n_selected': len(CLIN_COLS),
        'n_nonzero': n_nz,
        'alpha': best_a,
        'l1_ratio': best_l1,
        'train_c_index': round(c_train, 4),
        'test_c_index': round(c_test, 4),
    }, f, indent=2)
print(f'  Saved to: {model_dir}')

print(f'\n{"="*60}')
print(f'All models saved to: {MODEL_DIR}')
print('Done!')
