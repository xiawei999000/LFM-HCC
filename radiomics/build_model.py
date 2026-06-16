"""
Rebuild radiomics models (mRMR + LASSO) on train/val/ext1/ext2 split.
Save model weights to output2.
"""
import os, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RADIOMICS_OUTPUT_DIR, RADIOMICS_FEATURES_PATH, CLINICAL_INFO_PATH

OUTPUT2 = RADIOMICS_OUTPUT_DIR
os.makedirs(OUTPUT2, exist_ok=True)

FEATURES_ALL = RADIOMICS_FEATURES_PATH
TARGETS = ['MVI', 'PathologicalGrade']
N_MRMR = 30
RANDOM_STATE = 42


def remove_zero_variance(df, feature_cols):
    std = df[feature_cols].std()
    keep = std[std > 1e-10].index.tolist()
    dropped = len(feature_cols) - len(keep)
    if dropped:
        print(f"  Dropped {dropped} zero-variance features")
    return keep


def pearson_corr_filter(df, feature_cols, threshold=0.9):
    corr = df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if any(upper[col] > threshold):
            to_drop.add(col)
    kept = [f for f in feature_cols if f not in to_drop]
    if to_drop:
        print(f"  Dropped {len(to_drop)} highly correlated features (r>{threshold})")
    return kept


def mrmr_selection(df, feature_cols, labels, K=30):
    X = df[feature_cols].values
    y = labels.values
    n = X.shape[1]

    relevance = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(n)])
    relevance = np.nan_to_num(relevance, 0)

    selected = [int(np.argmax(relevance))]
    remaining = [i for i in range(n) if i != selected[0]]

    while len(selected) < K and remaining:
        best_score, best_idx = -float('inf'), -1
        for idx in remaining:
            rel = relevance[idx]
            red = np.mean([abs(np.corrcoef(X[:, idx], X[:, s])[0, 1])
                          for s in selected]) if selected else 0
            red = np.nan_to_num(red, 0)
            score = rel - red
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [feature_cols[i] for i in selected]


def lasso_select(df, feature_cols, labels, C_values=None):
    if C_values is None:
        C_values = np.logspace(-3, 1, 50)

    X = df[feature_cols].values
    y = labels.values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_c, best_score = 1.0, 0

    for C in C_values:
        model = LogisticRegression(penalty='l1', C=C, solver='saga',
                                   max_iter=5000, random_state=RANDOM_STATE, n_jobs=-1)
        probs = np.zeros(len(y))
        for tr, va in cv.split(X_s, y):
            model.fit(X_s[tr], y[tr])
            probs[va] = model.predict_proba(X_s[va])[:, 1]
        auc = roc_auc_score(y, probs)
        if auc > best_score:
            best_score, best_c = auc, C

    final = LogisticRegression(penalty='l1', C=best_c, solver='saga',
                               max_iter=5000, random_state=RANDOM_STATE, n_jobs=-1)
    final.fit(X_s, y)
    coef = final.coef_.flatten()
    selected_mask = coef != 0
    final_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]

    return final_features, best_c, coef, scaler, final.intercept_[0]


def main():
    print('[*] Loading features...')
    df_all = pd.read_excel(FEATURES_ALL)
    # Merge MVI and PathologicalGrade from clinical info
    clinical = pd.read_excel(CLINICAL_INFO_PATH)
    target_df = clinical[['Split', 'ID', 'MVI', 'PathologicalGrade']]
    df_all = df_all.merge(target_df, on=['Split', 'ID'], how='left')
    # Identify feature columns
    meta_cols = ['Split', 'ID', 'MVI', 'PathologicalGrade']
    feature_cols = [c for c in df_all.columns if c not in meta_cols]
    print(f'  Total: {df_all.shape[0]} patients, {len(feature_cols)} features')
    print(f'  MVI missing: {df_all["MVI"].isna().sum()}, Grade missing: {df_all["PathologicalGrade"].isna().sum()}')

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"  {target}")
        print(f"{'='*60}")

        # Training data
        train_mask = df_all['Split'] == 'train'
        train_df = df_all[train_mask].copy()
        y_train = train_df[target].values
        print(f'  Train: {len(train_df)} patients, class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}')

        # --- Step 1: Remove zero variance ---
        print('\n  [1/4] Variance filter...')
        kept = remove_zero_variance(train_df, feature_cols)
        print(f'    {len(kept)} features retained')

        # --- Step 2: Correlation filter ---
        print('  [2/4] Correlation filter...')
        kept = pearson_corr_filter(train_df, kept)
        print(f'    {len(kept)} features retained')

        # --- Step 3: mRMR ---
        print(f'  [3/4] mRMR (K={N_MRMR})...')
        mrmr_feats = mrmr_selection(train_df, kept, train_df[target], K=N_MRMR)
        print(f'    {len(mrmr_feats)} features selected')

        # Save mRMR features
        pd.Series(mrmr_feats).to_csv(f'{OUTPUT2}/mrmr_features_{target}.txt', index=False, header=False)

        # --- Step 4: LASSO refinement ---
        print('  [4/4] LASSO refinement...')
        final_features, best_c, coefs_all, scaler_cv, intercept = lasso_select(
            train_df, mrmr_feats, train_df[target])
        n_nonzero = sum(c != 0 for c in coefs_all)
        print(f'    Best C={best_c:.4f}, {n_nonzero} non-zero coefs → {len(final_features)} final features')

        # Select non-zero coefficients for final features
        final_coefs = coefs_all[[mrmr_feats.index(f) for f in final_features]]

        # --- Train final model on all train data ---
        X_train_final = train_df[final_features].values
        y_train_final = train_df[target].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_final)

        final_model = LogisticRegression(penalty='l1', C=best_c, solver='saga',
                                         max_iter=5000, random_state=RANDOM_STATE, n_jobs=-1)
        final_model.fit(X_train_s, y_train_final)
        final_coefs = final_model.coef_.flatten()
        final_intercept = final_model.intercept_[0]

        print(f'\n  Final model: {len(final_features)} features')
        print(f'  Intercept: {final_intercept:.6f}')
        print(f'  Top 5 coefs:')
        for f, c in sorted(zip(final_features, final_coefs), key=lambda x: -abs(x[1]))[:5]:
            print(f'    {f}: {c:.6f}')

        # --- Save model weights ---
        # 1) Features + coefficients
        coef_df = pd.DataFrame({'feature': final_features, 'coefficient': final_coefs})
        coef_df.to_csv(f'{OUTPUT2}/lasso_coef_{target}.csv', index=False)

        # 2) Intercept
        with open(f'{OUTPUT2}/intercept_{target}.txt', 'w') as f:
            f.write(str(final_intercept))

        # 3) Scaler (mean + scale as CSV)
        scaler_df = pd.DataFrame({
            'feature': final_features,
            'mean': scaler.mean_,
            'scale': scaler.scale_,
        })
        scaler_df.to_csv(f'{OUTPUT2}/scaler_{target}.csv', index=False)

        # 4) Best C
        with open(f'{OUTPUT2}/best_c_{target}.txt', 'w') as f:
            f.write(str(best_c))

        # 5) Selected features list
        with open(f'{OUTPUT2}/selected_features_{target}.txt', 'w') as f:
            f.write('\n'.join(final_features))

        # --- Predict on all splits ---
        print(f'\n  Predictions:')
        all_preds = []
        for split in ['train', 'val', 'ext1', 'ext2']:
            split_mask = df_all['Split'] == split
            split_df = df_all[split_mask].copy()
            if len(split_df) == 0:
                continue
            X_s = scaler.transform(split_df[final_features].values)
            logits = X_s @ final_coefs + final_intercept
            probs = 1.0 / (1.0 + np.exp(-logits))
            preds = (probs >= 0.5).astype(int)
            y_true = split_df[target].values

            if len(np.unique(y_true)) >= 2:
                auc = roc_auc_score(y_true, probs)
            else:
                auc = float('nan')
            print(f'    {split:>6}: AUC={auc:.4f}  (n={len(split_df)})')

            for i in range(len(split_df)):
                all_preds.append({
                    'Split': split_df.iloc[i]['Split'],
                    'ID': split_df.iloc[i]['ID'],
                    f'{target}_label': y_true[i],
                    f'{target}_prob': probs[i],
                    f'{target}_pred': preds[i],
                })

        pred_df = pd.DataFrame(all_preds)
        pred_df.to_csv(f'{OUTPUT2}/predictions_{target}.csv', index=False)
        print(f'  Saved predictions to {OUTPUT2}/predictions_{target}.csv')

    print(f'\n[*] All model weights saved to {OUTPUT2}')
    print('    Files: lasso_coef_*.csv, intercept_*.txt, scaler_*.csv, best_c_*.txt, selected_features_*.txt')


if __name__ == '__main__':
    main()
