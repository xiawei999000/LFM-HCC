"""
Univariate + multivariate Cox analysis: LFM-Seq risk score + clinical features.
Evaluates C-index and time-dependent AUC (ext1 + ext2).
Note: 'DINOv2_Continue_risk' is the column name in the prediction Excel file,
corresponding to the LFM-Seq model.
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

LFM_SEQ_RISK_COL = 'DINOv2_Continue_risk'  # column name in prediction Excel file
CONTINUOUS = ['Age', LFM_SEQ_RISK_COL]
BINARY = ['Sex', 'HBsAg', 'AFP', 'ALT', 'AST', 'GGT', 'tumor_size']


def load_and_merge(clinical_path, risk_scores_path):
    cli = pd.read_excel(clinical_path)
    cli = cli.rename(columns={c: c.strip() for c in cli.columns})
    risk = pd.read_excel(risk_scores_path)
    df = cli.merge(risk[['Split', 'ID', LFM_SEQ_RISK_COL]],
                   on=['Split', 'ID'], how='inner')
    df = df[df['RFS'] > 0].copy()
    return df


def calc_td_auc(df, risk_col, times):
    T = df['RFS'].values
    E = df['RFS_status'].values.astype(bool)
    results = []
    for t in times:
        pos = (T <= t) & E
        neg = (T > t)
        valid = pos | neg
        y = pos[valid].astype(int)
        s = df[risk_col].values[valid]
        results.append(roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan)
    return results


def run_analysis(clinical_path, risk_scores_path, output_dir):
    df_all = load_and_merge(clinical_path, risk_scores_path)
    ext1 = df_all[df_all['Split'] == 'ext1'].copy()
    ext2 = df_all[df_all['Split'] == 'ext2'].copy()
    print(f"ext1: N={len(ext1)}, Events={int(ext1['RFS_status'].sum())}")
    print(f"ext2: N={len(ext2)}, Events={int(ext2['RFS_status'].sum())}")

    # Univariate Cox (ext1)
    print("\n" + "=" * 70)
    print("Univariate Cox Regression (ext1)")
    print("=" * 70)
    univ_results = []
    for var in CONTINUOUS + BINARY:
        try:
            cph = CoxPHFitter()
            cph.fit(ext1[['RFS', 'RFS_status', var]], 'RFS', 'RFS_status')
            hr = np.exp(cph.params_[var])
            ci = np.exp(cph.confidence_intervals_.loc[var])
            p = cph.summary.loc[var, 'p']
            univ_results.append({
                'Variable': var, 'HR': hr, 'CI_low': ci.iloc[0], 'CI_high': ci.iloc[1],
                'P': p, 'Significant': p < 0.05})
            sig = '*' if p < 0.05 else ''
            print(f"  {var:25s} HR={hr:.3f} ({ci.iloc[0]:.3f}-{ci.iloc[1]:.3f}) P={p:.4f} {sig}")
        except Exception as e:
            print(f"  {var:25s} ERROR: {e}")

    # Multivariate Cox (ext1)
    sig_vars = [r['Variable'] for r in univ_results
                if r['Significant'] and r['Variable'] != LFM_SEQ_RISK_COL]
    mv_vars = [LFM_SEQ_RISK_COL] + sig_vars
    print(f"\nMultivariate Cox variables: {mv_vars}")

    try:
        cph_mv = CoxPHFitter()
        cph_mv.fit(ext1[['RFS', 'RFS_status'] + mv_vars], 'RFS', 'RFS_status')
        print("\nMultivariate Cox Regression (ext1):")
        print(cph_mv.summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                               'exp(coef) upper 95%', 'p']].to_string())
        ext1['Fusion_risk'] = cph_mv.predict_partial_hazard(ext1[mv_vars])
        ext2['Fusion_risk'] = cph_mv.predict_partial_hazard(ext2[mv_vars])
    except Exception as e:
        print(f"Multivariate Cox failed: {e}, fallback to Continue only")
        ext1['Fusion_risk'] = ext1[LFM_SEQ_RISK_COL]
        ext2['Fusion_risk'] = ext2[LFM_SEQ_RISK_COL]

    # C-index and tAUC
    times = np.arange(6, 37, 6)
    print("\n" + "=" * 70)
    print("C-index and Time-dependent AUC")
    print("=" * 70)
    header = f"{'Model':<22} {'Split':<6} {'C-index':>8} {'tAUC_12m':>8} {'tAUC_24m':>8} {'tAUC_36m':>8}"
    print(header)

    results = []
    for model_name, risk_col in [('Continue', LFM_SEQ_RISK_COL), ('Fusion', 'Fusion_risk')]:
        for split_name, sdf in [('ext1', ext1), ('ext2', ext2)]:
            T = sdf['RFS'].values
            E = sdf['RFS_status'].values.astype(bool)
            c = concordance_index(T, -sdf[risk_col].values, E)
            tauc = calc_td_auc(sdf, risk_col, times)
            auc12, auc24, auc36 = tauc[1], tauc[3], tauc[5]
            print(f"  {model_name:<20} {split_name:<6} {c:8.4f} {auc12:8.4f} {auc24:8.4f} {auc36:8.4f}")
            results.append({
                'Model': model_name, 'Split': split_name,
                'C_index': c, 'tAUC_12m': auc12, 'tAUC_24m': auc24, 'tAUC_36m': auc36,
            })
            for ti, t in enumerate(times):
                results[-1][f'tAUC_{t}m'] = tauc[ti]

    df_res = pd.DataFrame(results)
    out_path = os.path.join(output_dir, 'cox_fusion_model_results.xlsx')
    df_res.to_excel(out_path, index=False)
    print(f"\n[*] Saved: {out_path}")

    fusion_scores = pd.concat([
        ext1[['Split', 'ID', 'RFS', 'RFS_status', LFM_SEQ_RISK_COL, 'Fusion_risk']],
        ext2[['Split', 'ID', 'RFS', 'RFS_status', LFM_SEQ_RISK_COL, 'Fusion_risk']],
    ])
    fusion_out = os.path.join(output_dir, 'fusion_risk_scores.xlsx')
    fusion_scores.to_excel(fusion_out, index=False)
    print(f"[*] Saved: {fusion_out}")

    univ_df = pd.DataFrame(univ_results)
    univ_out = os.path.join(output_dir, 'univariate_cox_results.xlsx')
    univ_df.to_excel(univ_out, index=False)
    print(f"[*] Saved: {univ_out}")

    return df_res, univ_df


def parse_args():
    parser = argparse.ArgumentParser(description="Cox univariate + multivariate analysis")
    parser.add_argument("--clinical", type=str, required=True)
    parser.add_argument("--risk_scores", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_analysis(args.clinical, args.risk_scores, args.output_dir)
    print("\n[*] Done")


if __name__ == '__main__':
    main()
