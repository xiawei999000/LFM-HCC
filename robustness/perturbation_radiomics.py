"""
Radiomics feature-level perturbation robustness test.
Uses full model weights (coef + intercept + scaler).
Adds Gaussian noise (0-25%) directly to extracted feature values.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
from config import PROJECT_ROOT, OUTPUT_DIR as ROOT_OUTPUT_DIR

class Config:
    FEATURES_FILE = os.path.join(ROOT_OUTPUT_DIR, 'radiomics', 'features_all.csv')
    MODEL_DIR = os.path.join(ROOT_OUTPUT_DIR, 'radiomics')
    OUTPUT_DIR = os.path.join(ROOT_OUTPUT_DIR, 'robustness', 'radiomics')
    TARGETS = ['MVI', 'PathologicalGrade']
    PERTURB_LABELS = ['0%', '5%', '10%', '15%', '20%', '25%']
    PERTURB_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    N_RUNS = 50

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

def load_model(target, cfg):
    """Load full model: features, coefficients, intercept, scaler."""
    df = pd.read_csv(os.path.join(cfg.MODEL_DIR, f'lasso_coef_{target}.csv'))
    features = df['feature'].tolist()
    coefs = df['coefficient'].values

    # Load intercept
    intercept_file = os.path.join(cfg.MODEL_DIR, f'intercept_{target}.txt')
    with open(intercept_file, 'r') as f:
        intercept = float(f.read().strip())

    # Load scaler
    scaler_df = pd.read_csv(os.path.join(cfg.MODEL_DIR, f'scaler_{target}.csv'))
    scaler = StandardScaler()
    scaler.mean_ = scaler_df['mean'].values
    scaler.scale_ = scaler_df['scale'].values
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    return features, coefs, intercept, scaler

def main():
    t0 = time.time()
    cfg = Config()

    print('[*] Loading data and models...')
    df_all = pd.read_csv(cfg.FEATURES_FILE)
    print(f'  Total: {len(df_all)} patients, columns={df_all.shape[1]}')

    models = {}
    for target in cfg.TARGETS:
        features, coefs, intercept, scaler = load_model(target, cfg)
        models[target] = (features, coefs, intercept, scaler)
        print(f'  [{target}] {len(features)} features, intercept={intercept:.6f}')

    all_results = {}

    for split_name in ['val', 'ext1', 'ext2']:
        split_mask = df_all['Split'].str.strip().str.lower() == split_name.lower()
        split_df = df_all[split_mask].copy()
        if len(split_df) == 0: continue
        print(f'\n{"="*50}')
        print(f'  Perturbation test: {split_name} ({len(split_df)} pts)')
        print(f'{"="*50}')

        for target in cfg.TARGETS:
            features, coefs, intercept, scaler = models[target]
            y_true = split_df[target].values

            # Build baseline feature matrix
            X_base = np.zeros((len(split_df), len(features)), dtype=np.float64)
            for i, f in enumerate(features):
                if f in split_df.columns:
                    X_base[:, i] = split_df[f].values.astype(np.float64)

            # Feature-level perturbation: add Gaussian noise proportional to feature std
            feat_stds = np.nanstd(X_base, axis=0)
            feat_stds[feat_stds == 0] = 1e-10

            all_aucs = {lbl: [] for lbl in cfg.PERTURB_LABELS}

            for ratio, label in zip(cfg.PERTURB_RATIOS, cfg.PERTURB_LABELS):
                for run in range(cfg.N_RUNS):
                    if ratio == 0.0:
                        X_pert = X_base.copy()
                    else:
                        noise = np.random.randn(*X_base.shape) * feat_stds * ratio
                        X_pert = X_base + noise

                    X_s = scaler.transform(X_pert)
                    logits = X_s @ coefs + intercept
                    probs = 1.0 / (1.0 + np.exp(-logits))

                    if len(np.unique(y_true)) >= 2:
                        auc = roc_auc_score(y_true, probs)
                    else:
                        auc = 0.5
                    all_aucs[label].append(auc)

            key = f'{target}_{split_name}'
            all_results[key] = all_aucs
            means = [f'{np.mean(all_aucs[l]):.4f}' for l in cfg.PERTURB_LABELS]
            print(f'  [{target}] AUC: {" → ".join(means)}')

    print('
[*] Saving results...')
    _save_results(all_results, cfg)
    print(f'
[*] Total time: {(time.time()-t0):.1f}s')

def _save_results(all_results, cfg):
    rows, records = [], []
    for key, aucs in all_results.items():
        target, split = key.rsplit('_', 1)
        for label in cfg.PERTURB_LABELS:
            vals = aucs[label]
            rows.append({'Target': target, 'Split': split, 'PerturbLevel': label,
                         'Mean_AUC': np.mean(vals), 'Std_AUC': np.std(vals),
                         'Median': np.median(vals), 'Min': np.min(vals), 'Max': np.max(vals)})
            for run, val in enumerate(vals):
                records.append({'Target': target, 'Split': split, 'PerturbLevel': label, 'Run': run, 'AUC': val})
    pd.DataFrame(rows).to_excel(os.path.join(cfg.OUTPUT_DIR, 'perturbation_feature_summary.xlsx'), index=False)
    pd.DataFrame(records).to_excel(os.path.join(cfg.OUTPUT_DIR, 'perturbation_feature_50runs.xlsx'), index=False)

    tasks = cfg.TARGETS; splits = ['val', 'ext1', 'ext2']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for t_idx, task in enumerate(tasks):
        for s_idx, split in enumerate(splits):
            k = f'{task}_{split}'
            if k not in all_results: continue
            ax = axes[t_idx, s_idx]
            data = [all_results[k][l] for l in cfg.PERTURB_LABELS]
            bp = ax.boxplot(data, labels=cfg.PERTURB_LABELS, patch_artist=True, widths=0.6,
                           showfliers=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor('#5B9BD5' if i == 0 else '#A5A5A5'); patch.set_alpha(0.7)
            ax.axhline(y=np.median(all_results[k]['0%']), color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_title(f'{task} — {split}', fontsize=13, fontweight='bold')
            ax.set_ylabel('AUC'); ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Radiomics Feature-level Perturbation Robustness (50 runs)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'perturbation_feature_boxplot.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('[*] Files saved.')

if __name__ == '__main__':
    main()
