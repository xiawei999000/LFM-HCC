"""
Combined figure: all 6 models, MVI (row 1) + Grade (row 2), 10/20/50% (columns).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUTPUT_DIR

BASE = os.path.join(OUTPUT_DIR, 'training_test')
OUTPUT = os.path.join(BASE, "combined_boxplot.png")

MODEL_ORDER = ["dinov2", "deeplesion", "combine", "continue",
               "fmcib_3dresnet", "radiomics_lasso"]
MODEL_DISPLAY = {
    "dinov2": "DINOv2\nOfficial",
    "deeplesion": "DeepLesion",
    "combine": "Combine",
    "continue": "Continue",
    "fmcib_3dresnet": "FMCIB\n3D ResNet",
    "radiomics_lasso": "Radiomics\nLASSO",
}
MODEL_COLORS = {
    "dinov2": "#E74C3C",
    "deeplesion": "#2980B9",
    "combine": "#27AE60",
    "continue": "#8E44AD",
    "fmcib_3dresnet": "#E67E22",
    "radiomics_lasso": "#1ABC9C",
}
RATIOS = [0.1, 0.2, 0.5]
RATIO_LABELS = {0.1: "10% Train", 0.2: "20% Train", 0.5: "50% Train"}
TASKS = ["mvi", "grade"]
TASK_DISPLAY = {"mvi": "MVI", "grade": "PathologicalGrade"}
SETS = ["val", "ext1", "ext2"]
SET_DISPLAY = {"val": "Val", "ext1": "Ext1", "ext2": "Ext2"}
SET_COLORS = {"val": "#4C72B0", "ext1": "#55A868", "ext2": "#C44E52"}


def load_dinov2_results():
    """Load DINOv2 4 models from separate CSV files."""
    dfs = []
    for model in ["dinov2", "deeplesion", "combine", "continue"]:
        path = os.path.join(BASE, "DINOV2", f"results_{model}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["target"] = "both"  # each row has both MVI and Grade
            dfs.append(df)
    if not dfs:
        return None
    all_df = pd.concat(dfs, ignore_index=True)

    # Melt MVI columns
    mvi_cols = {c: c.replace("_mvi_auc", "_auc") for c in all_df.columns
                if "mvi_auc" in c and "grade" not in c}
    mvi_cols["model"] = "model"
    mvi_cols["ratio"] = "ratio"
    mvi_cols["run"] = "run"
    mvi = all_df[["model", "ratio", "run"] + [c for c in all_df.columns if "mvi_auc" in c and "grade" not in c]].copy()
    mvi = mvi.rename(columns={k: v for k, v in mvi_cols.items() if k != "model" and k != "ratio" and k != "run"})
    mvi["target"] = "mvi"

    # Melt Grade columns
    grd_cols = {c: c.replace("_grade_auc", "_auc") for c in all_df.columns
                if "_grade_auc" in c}
    grd_cols["model"] = "model"
    grd_cols["ratio"] = "ratio"
    grd_cols["run"] = "run"
    grd = all_df[["model", "ratio", "run"] + [c for c in all_df.columns if "_grade_auc" in c]].copy()
    grd = grd.rename(columns={k: v for k, v in grd_cols.items() if k != "model" and k != "ratio" and k != "run"})
    grd["target"] = "grade"

    return pd.concat([mvi, grd], ignore_index=True)


def load_naturemi_results():
    path = os.path.join(BASE, "NatureMI", "results_fmcib_3dresnet.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["target"] = "both"
    mvi = df[["model", "ratio", "run", "val_mvi_auc", "ext1_mvi_auc", "ext2_mvi_auc"]].copy()
    mvi = mvi.rename(columns={
        "val_mvi_auc": "val_auc",
        "ext1_mvi_auc": "ext1_auc",
        "ext2_mvi_auc": "ext2_auc",
    })
    mvi["target"] = "mvi"

    grd = df[["model", "ratio", "run", "val_grade_auc", "ext1_grade_auc", "ext2_grade_auc"]].copy()
    grd = grd.rename(columns={
        "val_grade_auc": "val_auc",
        "ext1_grade_auc": "ext1_auc",
        "ext2_grade_auc": "ext2_auc",
    })
    grd["target"] = "grade"

    return pd.concat([mvi, grd], ignore_index=True)


def load_radiomics_results():
    dfs = []
    for target in ["MVI", "PathologicalGrade"]:
        path = os.path.join(BASE, "Radiomics", f"results_radiomics_{target}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["target"] = "mvi" if target == "MVI" else "grade"
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def main():
    print("[*] Loading results...")
    all_parts = []
    for loader, name in [
        (load_dinov2_results, "DINOv2"),
        (load_naturemi_results, "NatureMI"),
        (load_radiomics_results, "Radiomics"),
    ]:
        part = loader()
        if part is not None:
            all_parts.append(part)
            print(f"    {name}: {len(part)} rows")

    all_df = pd.concat(all_parts, ignore_index=True)
    print(f"[*] Total: {len(all_df)} rows")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Few-Shot Training Sample Test — All Models",
                 fontsize=18, fontweight="bold", y=0.98)

    for row_idx, task in enumerate(TASKS):
        for col_idx, ratio in enumerate(RATIOS):
            ax = axes[row_idx][col_idx]
            sub = all_df[(all_df["target"] == task) & (all_df["ratio"] == ratio)]

            positions = []
            data_series = []
            colors_list = []

            for m_idx, model in enumerate(MODEL_ORDER):
                for s_idx, s in enumerate(SETS):
                    col_name = f"{s}_auc"
                    vals = sub[sub["model"] == model][col_name].dropna().values
                    if len(vals) == 0:
                        continue
                    n_sets = len(SETS)
                    width = 0.7 / n_sets
                    offset = (s_idx - (n_sets - 1) / 2) * width
                    pos = m_idx + 1 + offset
                    positions.append(pos)
                    data_series.append(vals)
                    colors_list.append(SET_COLORS[s])

            bp = ax.boxplot(data_series, positions=positions, widths=0.12,
                            patch_artist=True, showfliers=False,
                            medianprops={"color": "black", "linewidth": 1.2})

            for patch, color in zip(bp["boxes"], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)

            # Title
            ax.set_title(f"{TASK_DISPLAY[task]}  |  {RATIO_LABELS[ratio]}",
                         fontsize=13, fontweight="bold")

            # X axis
            ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
            ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER],
                               fontsize=8)
            ax.set_ylabel("AUC", fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0.35, 1.0)

            # Legend only on last column
            if col_idx == 2:
                legend_patches = [
                    plt.Rectangle((0, 0), 1, 1, fc=SET_COLORS[s], alpha=0.75)
                    for s in SETS
                ]
                ax.legend(legend_patches, [SET_DISPLAY[s] for s in SETS],
                          loc="upper right", fontsize=8, title="Eval Set")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(BASE, "combined_boxplot.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(BASE, "combined_boxplot.pdf"), bbox_inches="tight")
    plt.close()
    print(f"[*] Saved: {OUTPUT}")
    print(f"[*] Saved: {OUTPUT.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
