"""
DeLong test: compare LFM-Seq AUC vs other models for MVI/Grade.
Includes overall and tumor-size subgroup analysis.
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# Old column prefix → new display name
COL_MAP = {
    'Radiomics': 'Radiomics', 'FMCIB_3D': 'FMCIB',
    'DINOv2_Official': 'LFM-Base', 'DINOv2_DeepLesion': 'LFM-DL',
    'DINOv2_Combine': 'LFM-Mix', 'DINOv2_Continue': 'LFM-Seq',
}
MODELS = ['Radiomics', 'FMCIB_3D', 'DINOv2_Official',
          'DINOv2_DeepLesion', 'DINOv2_Combine']
COMPARE = 'DINOv2_Continue'  # → LFM-Seq


def delong_roc_test(y_true, prob1, prob2):
    """DeLong test for two correlated AUCs. Returns z, p_value."""
    n = len(y_true)
    n1 = np.sum(y_true == 1)
    n0 = n - n1
    if n1 < 2 or n0 < 2:
        return np.nan, np.nan

    def compute_V(X, Y):
        V10 = np.zeros(len(X))
        for i in range(len(X)):
            V10[i] = np.mean((X[i] > Y) + 0.5 * (X[i] == Y))
        return V10

    def compute_U(X, Y):
        V01 = np.zeros(len(Y))
        for i in range(len(Y)):
            V01[i] = np.mean((X > Y[i]) + 0.5 * (X == Y[i]))
        return V01

    idx1 = np.where(y_true == 1)[0]
    idx0 = np.where(y_true == 0)[0]
    p1_1, p2_1 = prob1[idx1], prob2[idx1]
    p1_0, p2_0 = prob1[idx0], prob2[idx0]

    V10_1 = compute_V(p1_1, p1_0)
    V10_2 = compute_V(p2_1, p2_0)
    V01_1 = compute_U(p1_1, p1_0)
    V01_2 = compute_U(p2_1, p2_0)

    S10_11 = np.cov(V10_1, V10_1)[0, 0] if n1 > 1 else 0
    S10_22 = np.cov(V10_2, V10_2)[0, 0] if n1 > 1 else 0
    S10_12 = np.cov(V10_1, V10_2)[0, 0] if n1 > 1 else 0
    S01_11 = np.cov(V01_1, V01_1)[0, 0] if n0 > 1 else 0
    S01_22 = np.cov(V01_2, V01_2)[0, 0] if n0 > 1 else 0
    S01_12 = np.cov(V01_1, V01_2)[0, 0] if n0 > 1 else 0

    var1 = S10_11 / n1 + S01_11 / n0
    var2 = S10_22 / n1 + S01_22 / n0
    cov12 = S10_12 / n1 + S01_12 / n0

    se_diff = np.sqrt(max(var1 + var2 - 2 * cov12, 1e-10))
    auc1 = np.mean(V10_1)
    auc2 = np.mean(V10_2)
    z = (auc1 - auc2) / se_diff
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def run(clinical_path, predictions_path, output_dir):
    cli = pd.read_excel(clinical_path)
    cli = cli.rename(columns={c: c.strip() for c in cli.columns})
    pred = pd.read_excel(predictions_path)
    prob_cols = ['Split', 'ID'] + [f'{m}_{t}_prob' for m in MODELS + [COMPARE]
                                   for t in ['MVI', 'Grade']]
    df = cli.merge(pred[prob_cols], on=['Split', 'ID'], how='inner')
    df['sub3'] = pd.cut(df['tumor_max_size_mm'], bins=[0, 30, 50, 999],
                         labels=['<=30mm', '30-50mm', '>50mm'])

    rows = []
    for task, lc, tn in [('MVI', 'MVI', 'MVI'), ('Grade', 'PathologicalGrade', 'E-S Grade')]:
        for split in ['val', 'ext1', 'ext2']:
            for sub in ['Overall', '<=30mm', '30-50mm', '>50mm']:
                s = df[df['Split'] == split]
                if sub != 'Overall':
                    s = s[s['sub3'] == sub]
                y = s[lc].values
                p_continue = s[f'{COMPARE}_{task}_prob'].values
                if len(np.unique(y)) < 2:
                    continue

                for other in MODELS:
                    p_other = s[f'{other}_{task}_prob'].values
                    z, p = delong_roc_test(y, p_continue, p_other)
                    rows.append({
                        'Task': tn, 'Split': split, 'Subgroup': sub,
                        'Model_A': COMPARE, 'Model_B': other,
                        'N': len(s), 'Z': round(z, 4) if not np.isnan(z) else np.nan,
                        'P_value': round(p, 6) if not np.isnan(p) else np.nan,
                    })

    for row in rows:
        s = df[df['Split'] == row['Split']]
        if row['Subgroup'] != 'Overall':
            s = s[s['sub3'] == row['Subgroup']]
        ts = 'MVI' if row['Task'] == 'MVI' else 'Grade'
        lc_auc = 'MVI' if row['Task'] == 'MVI' else 'PathologicalGrade'
        y = s[lc_auc].values
        colA = f"{row['Model_A']}_{ts}_prob"
        colB = f"{row['Model_B']}_{ts}_prob"
        row['AUC_A'] = round(roc_auc_score(y, s[colA].values), 4)
        row['AUC_B'] = round(roc_auc_score(y, s[colB].values), 4)

    out = pd.DataFrame(rows)
    out.to_excel(os.path.join(output_dir, 'delong_test_results.xlsx'), index=False)

    for task in ['MVI', 'E-S Grade']:
        print(f'\n===== {task} DeLong Test (Continue vs others) =====')
        for split in ['val', 'ext1', 'ext2']:
            t = out[(out['Task'] == task) & (out['Split'] == split) & (out['Subgroup'] == 'Overall')]
            print(f'\n  {split} (Overall):')
            for _, r in t.iterrows():
                sig = '***' if r['P_value'] < 0.001 else ('**' if r['P_value'] < 0.01 else
                                                           ('*' if r['P_value'] < 0.05 else ''))
                print(f'    vs {r["Model_B"]:<22s} AUC_A={r["AUC_A"]:.4f} AUC_B={r["AUC_B"]:.4f} '
                      f'Z={r["Z"]:+.3f} P={r["P_value"]:.6f} {sig}')

    print(f'\n[*] Saved to {output_dir}/delong_test_results.xlsx')


def parse_args():
    parser = argparse.ArgumentParser(description="DeLong AUC comparison test")
    parser.add_argument("--clinical", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run(args.clinical, args.predictions, args.output_dir)
    print("[*] Done")


if __name__ == '__main__':
    main()
