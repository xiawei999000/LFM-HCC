"""
Compute MVI/Grade AUC + 95% CI by 3-class tumor size subgroups.
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

# Old column prefix → new display name
COL_MAP = {
    'Radiomics': 'Radiomics', 'FMCIB_3D': 'FMCIB',
    'DINOv2_Official': 'LFM-Base', 'DINOv2_DeepLesion': 'LFM-DL',
    'DINOv2_Combine': 'LFM-Mix', 'DINOv2_Continue': 'LFM-Seq',
}
MODELS = ['Radiomics', 'FMCIB_3D', 'DINOv2_Official',
          'DINOv2_DeepLesion', 'DINOv2_Combine', 'DINOv2_Continue']


def bootstrap_ci(y_true, y_prob, alpha=0.05, n_boot=500):
    n = len(y_true)
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    auc = roc_auc_score(y_true, y_prob)
    ci_low = np.percentile(aucs, alpha / 2 * 100)
    ci_high = np.percentile(aucs, (1 - alpha / 2) * 100)
    return auc, ci_low, ci_high


def run(clinical_path, predictions_path, output_dir):
    cli = pd.read_excel(clinical_path)
    cli = cli.rename(columns={c: c.strip() for c in cli.columns})
    pred = pd.read_excel(predictions_path)
    prob_cols = ['Split', 'ID'] + [f'{m}_{t}_prob' for m in MODELS for t in ['MVI', 'Grade']]
    df = cli.merge(pred[prob_cols], on=['Split', 'ID'], how='inner')
    df['sub3'] = pd.cut(df['tumor_max_size_mm'], bins=[0, 30, 50, 999],
                         labels=['<=30mm', '30-50mm', '>50mm'])

    rows = []
    for task, lc in [('MVI', 'MVI'), ('Grade', 'PathologicalGrade')]:
        for model in MODELS:
            for split in ['val', 'ext1', 'ext2']:
                for sub in ['Overall', '<=30mm', '30-50mm', '>50mm']:
                    s = df[df['Split'] == split]
                    if sub != 'Overall':
                        s = s[s['sub3'] == sub]
                    y = s[lc].values
                    p = s[f'{model}_{task}_prob'].values
                    n, pos = len(s), int(y.sum())

                    if n < 5 or len(np.unique(y)) < 2:
                        auc, lo, hi = np.nan, np.nan, np.nan
                    else:
                        auc, lo, hi = bootstrap_ci(y, p, n_boot=500)

                    rows.append({
                        'Task': task, 'Model': model, 'Split': split,
                        'Subgroup': sub, 'N': n, 'Pos': pos,
                        'AUC': round(auc, 4) if not np.isnan(auc) else np.nan,
                        'CI_low': round(lo, 4) if not np.isnan(lo) else np.nan,
                        'CI_high': round(hi, 4) if not np.isnan(hi) else np.nan,
                    })

    out = pd.DataFrame(rows)
    out.to_excel(os.path.join(output_dir, 'auc_by_3class_subgroups.xlsx'), index=False)

    for task in ['MVI', 'Grade']:
        print(f'\n===== {task} =====')
        for sub in ['Overall', '<=30mm', '30-50mm', '>50mm']:
            s = out[(out['Task'] == task) & (out['Subgroup'] == sub)]
            print(f'\n--- {sub} ---')
            for split in ['val', 'ext1', 'ext2']:
                ss = s[s['Split'] == split]
                print(f'  {split}:')
                for _, r in ss.iterrows():
                    if not np.isnan(r['AUC']):
                        print(f'    {r["Model"]:<22s} n={int(r["N"]):3d} '
                              f'AUC={r["AUC"]:.4f} ({r["CI_low"]:.4f}-{r["CI_high"]:.4f})')
                    else:
                        print(f'    {r["Model"]:<22s} n={int(r["N"]):3d} AUC=N/A')

    print(f'\n[*] Saved to {output_dir}/auc_by_3class_subgroups.xlsx')


def parse_args():
    parser = argparse.ArgumentParser(description="Subgroup AUC + CI computation")
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
