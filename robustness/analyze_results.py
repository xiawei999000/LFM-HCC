"""
Analyze stratified sampling results and generate summary Excel + box plots.

Usage:
    python analyze_results.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==========================================
# Paths
# ==========================================
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'training_test')
os.makedirs(OUTPUT_BASE, exist_ok=True)

MODEL_NAMES = ['dinov2', 'deeplesion', 'combine', 'continue']
MODEL_DISPLAY = {
    'dinov2': 'DINOv2 Official',
    'deeplesion': 'DeepLesion',
    'combine': 'Combine',
    'continue': 'Continue',
}
RATIOS = [0.1, 0.2, 0.5]
RATIO_LABELS = {0.1: '10%', 0.2: '20%', 0.5: '50%'}
TASKS = ['mvi', 'grade']
TASK_DISPLAY = {'mvi': 'MVI', 'grade': 'PathologicalGrade'}
SETS = ['val', 'ext1', 'ext2']
SET_DISPLAY = {'val': 'Val', 'ext1': 'Ext1', 'ext2': 'Ext2'}
COLORS = {'val': '#4C72B0', 'ext1': '#55A868', 'ext2': '#C44E52'}


def load_all_results():
    """Load and combine all result CSV files."""
    dfs = []
    for model in MODEL_NAMES:
        csv_path = os.path.join(OUTPUT_BASE, f'results_{model}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"[*] Loaded {csv_path}: {len(df)} rows")
        else:
            print(f"[!] Missing: {csv_path}")

    if not dfs:
        raise FileNotFoundError("No result files found!")

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def save_summary_excel(all_df):
    """Save raw data and summary statistics to Excel."""
    xlsx_path = os.path.join(OUTPUT_BASE, 'results_summary.xlsx')

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        # Sheet 1: Raw data
        all_df.to_excel(writer, sheet_name='Raw', index=False)

        # Sheet 2: Summary statistics
        metric_cols = [c for c in all_df.columns if c.endswith('_auc')]
        summary = all_df.groupby(['model', 'ratio'])[metric_cols].agg(['mean', 'std'])
        summary = summary.round(4)
        summary.to_excel(writer, sheet_name='Summary')

        # Sheet 3: Pivot - mean AUC per model/ratio/task/set
        pivot_rows = []
        for model in MODEL_NAMES:
            for ratio in RATIOS:
                sub = all_df[(all_df['model'] == model) & (all_df['ratio'] == ratio)]
                if len(sub) == 0:
                    continue
                row = {'model': model, 'ratio': f'{ratio:.0%}'}
                for task in TASKS:
                    for s in SETS:
                        col = f'{s}_{task}_auc'
                        vals = sub[col].values
                        row[f'{s}_{task}'] = f'{np.mean(vals):.4f} +/- {np.std(vals):.4f}'
                pivot_rows.append(row)
        pivot_df = pd.DataFrame(pivot_rows)
        pivot_df.to_excel(writer, sheet_name='Pivot', index=False)

    print(f"[*] Summary Excel saved to: {xlsx_path}")
    return xlsx_path


def draw_boxplots(all_df):
    """Draw box plots comparing models at each ratio for MVI and Grade."""
    for task in TASKS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        fig.suptitle(f'{TASK_DISPLAY[task]} AUC Comparison', fontsize=16, fontweight='bold')

        for ax_idx, ratio in enumerate(RATIOS):
            ax = axes[ax_idx]
            sub = all_df[all_df['ratio'] == ratio]

            # Prepare data: list of arrays for each (model, set)
            positions = []
            data_series = []
            colors_list = []

            for m_idx, model in enumerate(MODEL_NAMES):
                for s_idx, s in enumerate(SETS):
                    col = f'{s}_{task}_auc'
                    vals = sub[sub['model'] == model][col].dropna().values
                    if len(vals) == 0:
                        continue
                    # Position: model group center + offset per set
                    n_sets = len(SETS)
                    width = 0.7 / n_sets
                    offset = (s_idx - (n_sets - 1) / 2) * width
                    pos = m_idx + 1 + offset

                    positions.append(pos)
                    data_series.append(vals)
                    colors_list.append(COLORS[s])

            bp = ax.boxplot(data_series, positions=positions, widths=0.15,
                            patch_artist=True, showfliers=False,
                            medianprops={'color': 'black', 'linewidth': 1.5})

            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(f'Train Ratio: {RATIO_LABELS[ratio]}', fontsize=13)
            ax.set_xticks(range(1, len(MODEL_NAMES) + 1))
            ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_NAMES], rotation=20, ha='right', fontsize=10)
            ax.set_ylabel('AUC', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0.4, 1.0)

            if ax_idx == 2:
                legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[s], alpha=0.7)
                                  for s in SETS]
                ax.legend(legend_patches, [SET_DISPLAY[s] for s in SETS],
                          loc='upper right', fontsize=9)

        plt.tight_layout()
        png_path = os.path.join(OUTPUT_BASE, f'boxplot_{task}.png')
        pdf_path = os.path.join(OUTPUT_BASE, f'boxplot_{task}.pdf')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        print(f"[*] Box plot saved: {png_path}")
        print(f"[*] Box plot saved: {pdf_path}")


def print_summary_table(all_df):
    """Print a formatted summary table to stdout."""
    metric_cols = [c for c in all_df.columns if c.endswith('_auc')]
    print("\n" + "=" * 120)
    print("SUMMARY: Mean AUC +/- Std per model, ratio, and evaluation set")
    print("=" * 120)

    for task in TASKS:
        print(f"\n{'─'*100}")
        print(f"  {TASK_DISPLAY[task]} AUC")
        print(f"{'─'*100}")
        header = f"{'Model':<15} {'Ratio':<8}"
        for s in SETS:
            header += f" {SET_DISPLAY[s]:>22}"
        print(header)
        print("-" * 100)

        for model in MODEL_NAMES:
            for ratio in RATIOS:
                sub = all_df[(all_df['model'] == model) & (all_df['ratio'] == ratio)]
                if len(sub) == 0:
                    continue
                line = f"{MODEL_DISPLAY[model]:<15} {RATIO_LABELS[ratio]:<8}"
                for s in SETS:
                    col = f'{s}_{task}_auc'
                    vals = sub[col].values
                    line += f" {np.mean(vals):.4f}+/-{np.std(vals):.4f}  "
                print(line)

    print("\n" + "=" * 120)


def main():
    all_df = load_all_results()
    print(f"[*] Total results: {len(all_df)} rows")
    print(f"[*] Models: {all_df['model'].unique().tolist()}")
    print(f"[*] Ratios: {sorted(all_df['ratio'].unique())}")

    # Check completeness
    for model in MODEL_NAMES:
        for ratio in RATIOS:
            n = len(all_df[(all_df['model'] == model) & (all_df['ratio'] == ratio)])
            expected = 50
            status = 'OK' if n == expected else f'MISSING {expected - n}'
            print(f"    {model} @ {ratio:.0%}: {n}/{expected} runs {status}")

    save_summary_excel(all_df)
    draw_boxplots(all_df)
    print_summary_table(all_df)


if __name__ == '__main__':
    main()
