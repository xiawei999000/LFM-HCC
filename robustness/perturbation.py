"""
Input perturbation robustness test: simulates inter-reader variability in tumor localization.
Applies 0-25% random shifts to ROI center across 50 independent runs to assess AUC stability.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np, SimpleITK as sitk
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from timm.models.vision_transformer import vit_small_patch8_224
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from config import PROJECT_ROOT, DATA_DIR, MODEL_DIR, OUTPUT_DIR


class Config:
    # User: set paths to your data and model directories
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'robustness')
    COHORT_A_EXCEL = os.path.join(DATA_DIR, 'clinical_cohort_a.xlsx')
    COHORT_A_AP_DIR = os.path.join(DATA_DIR, 'cohort_a_ap')
    COHORT_A_PV_DIR = os.path.join(DATA_DIR, 'cohort_a_pv')
    COHORT_B_EXCEL = os.path.join(DATA_DIR, 'clinical_cohort_b.xlsx')
    COHORT_B_AP_DIR = os.path.join(DATA_DIR, 'cohort_b_ap')
    COHORT_B_PV_DIR = os.path.join(DATA_DIR, 'cohort_b_pv')

    MODEL_DIR = os.path.join(MODEL_DIR, 'finetuned')
    MVI_WEIGHT = 'best_mvi.pth'
    GRADE_WEIGHT = 'best_grade.pth'

    PERTURB_LEVELS = [0, 6, 13, 19, 26, 32]
    PERTURB_LABELS = ['0%', '5%', '10%', '15%', '20%', '25%']
    N_RUNS = 50
    IMG_SIZE = 128
    EMBED_DIM = 384

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ==========================================
# Model definition
# ==========================================
class CrossPhaseAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm_ap = nn.LayerNorm(dim)
        self.norm_pv = nn.LayerNorm(dim)
        self.ap_query_pv = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.pv_query_ap = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, ap_feat, pv_feat):
        ap_norm = self.norm_ap(ap_feat)
        pv_norm = self.norm_pv(pv_feat)
        ap_cross, _ = self.ap_query_pv(query=ap_norm, key=pv_norm, value=pv_norm)
        pv_cross, _ = self.pv_query_ap(query=pv_norm, key=ap_norm, value=ap_norm)
        return ap_feat + ap_cross, pv_feat + pv_cross


class GatedAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.w(self.V(x) * self.U(x))
        a = torch.softmax(a, dim=0)
        z = torch.sum(x * a, dim=0, keepdim=True)
        return z, a


class MultiTaskMILModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = vit_small_patch8_224(
            img_size=128, in_chans=1, num_classes=0,
            pretrained=False, dynamic_img_size=True, init_values=1e-5,
        )
        D = cfg.EMBED_DIM
        fusion_dim = D * 2
        self.cross_phase_attn = CrossPhaseAttention(dim=D, num_heads=4, dropout=0.1)
        self.attention = GatedAttention(fusion_dim)
        self.head_mvi = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, 1)
        )
        self.head_grade = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, 1)
        )
        self.log_vars = nn.Parameter(torch.zeros(2))

    def load_weights(self, weight_path):
        ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
        self.load_state_dict(ckpt, strict=True)

    def forward_one_patient(self, ap_tensor, pv_tensor):
        feat_ap = self.backbone(ap_tensor)
        feat_pv = self.backbone(pv_tensor)
        feat_ap = feat_ap.unsqueeze(0)
        feat_pv = feat_pv.unsqueeze(0)
        feat_ap, feat_pv = self.cross_phase_attn(feat_ap, feat_pv)
        feat_ap = feat_ap.squeeze(0)
        feat_pv = feat_pv.squeeze(0)
        feat_fused = torch.cat([feat_ap, feat_pv], dim=1)
        z, _ = self.attention(feat_fused)
        out_mvi = self.head_mvi(z).view(-1)
        out_grade = self.head_grade(z).view(-1)
        return torch.cat([out_mvi, out_grade], dim=0)


# ==========================================
# Dataset loading (in-memory)
# ==========================================
def load_and_prepare_data(cfg):
    """Preload all patient volumes into memory."""
    df_cohort_a = pd.read_excel(cfg.COHORT_A_EXCEL)
    df_cohort_b = pd.read_excel(cfg.COHORT_B_EXCEL)

    data = {'val': [], 'ext1': [], 'ext2': []}

    for split_name in ['val', 'ext1', 'ext2']:
        if split_name == 'ext2':
            df = df_cohort_b
        else:
            df = df_cohort_a[df_cohort_a['set_split'].astype(str).str.strip().str.lower() == split_name]

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Loading {split_name}'):
            pid = str(int(row['ID']))
            mvi = int(row['MVI'])
            grade = int(row['PathologicalGrade'])

            if split_name == 'ext2':
                ap_path = os.path.join(cfg.COHORT_B_AP_DIR, f'{pid}.nii.gz')
                pv_path = os.path.join(cfg.COHORT_B_PV_DIR, f'{pid}.nii.gz')
            else:
                ap_path = os.path.join(cfg.COHORT_A_AP_DIR, f'{pid}.nii.gz')
                pv_path = os.path.join(cfg.COHORT_A_PV_DIR, f'{pid}.nii.gz')

            if not (os.path.exists(ap_path) and os.path.exists(pv_path)):
                continue

            ap_img = sitk.ReadImage(ap_path)
            ap_arr = sitk.GetArrayFromImage(ap_img).astype(np.float32) / 255.0
            pv_img = sitk.ReadImage(pv_path)
            pv_arr = sitk.GetArrayFromImage(pv_img).astype(np.float32) / 255.0

            data[split_name].append({
                'pid': pid, 'mvi': mvi, 'grade': grade,
                'ap_vol': ap_arr, 'pv_vol': pv_arr,
                'D': ap_arr.shape[0],
            })

    for s in data:
        mvi_pos = sum(p['mvi'] for p in data[s])
        grade_pos = sum(p['grade'] for p in data[s])
        print(f"[*] {s}: {len(data[s])} pts, MVI+={mvi_pos}, Grade+={grade_pos}")

    return data


# ==========================================
# Perturbation functions
# ==========================================
def shift_volume(vol, sx, sy, sz):
    """Translate volume, fill empty regions with zeros."""
    D, H, W = vol.shape
    shifted = np.zeros_like(vol)

    if sx >= 0:
        sx1, sx2, dx1, dx2 = 0, W - sx, sx, W
    else:
        sx1, sx2, dx1, dx2 = -sx, W, 0, W + sx
    if sy >= 0:
        sy1, sy2, dy1, dy2 = 0, H - sy, sy, H
    else:
        sy1, sy2, dy1, dy2 = -sy, H, 0, H + sy
    if sz >= 0:
        sz1, sz2, dz1, dz2 = 0, D - sz, sz, D
    else:
        sz1, sz2, dz1, dz2 = -sz, D, 0, D + sz

    sx1, sx2 = max(0, sx1), min(W, sx2)
    dx1, dx2 = max(0, dx1), min(W, dx2)
    sy1, sy2 = max(0, sy1), min(H, sy2)
    dy1, dy2 = max(0, dy1), min(H, dy2)
    sz1, sz2 = max(0, sz1), min(D, sz2)
    dz1, dz2 = max(0, dz1), min(D, dz2)

    if sx2 > sx1 and sy2 > sy1 and sz2 > sz1:
        shifted[dz1:dz2, dy1:dy2, dx1:dx2] = vol[sz1:sz2, sy1:sy2, sx1:sx2]
    return shifted


def perturb_and_infer(model, patient_data, max_shift, device):
    """
    Apply random perturbations to patients and run inference. Returns [N, 2] logits.
    """
    all_logits = []
    for p in patient_data:
        if max_shift == 0:
            ap_s = p['ap_vol']
            pv_s = p['pv_vol']
        else:
            sx = np.random.randint(-max_shift, max_shift + 1)
            sy = np.random.randint(-max_shift, max_shift + 1)
            max_z = max(1, int(p['D'] * max_shift / 128))
            sz = np.random.randint(-max_z, max_z + 1)
            ap_s = shift_volume(p['ap_vol'], sx, sy, sz)
            pv_s = shift_volume(p['pv_vol'], sx, sy, sz)

        ap_t = torch.from_numpy(ap_s.copy()).unsqueeze(1).to(device)
        pv_t = torch.from_numpy(pv_s.copy()).unsqueeze(1).to(device)

        with torch.no_grad():
            logits = model.forward_one_patient(ap_t, pv_t)
        all_logits.append(logits.cpu().numpy())

    return np.stack(all_logits, axis=0)


# ==========================================
# Main experiment
# ==========================================
def run_robustness_test(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    # Preprocess data
    print("
[*] Loading data into memory...")
    data = load_and_prepare_data(cfg)

    # Load model
    print("\n[*] Load model...")
    model_mvi = MultiTaskMILModel(cfg).to(device)
    model_mvi.load_weights(os.path.join(cfg.MODEL_DIR, cfg.MVI_WEIGHT))
    model_mvi.eval()

    model_grade = MultiTaskMILModel(cfg).to(device)
    model_grade.load_weights(os.path.join(cfg.MODEL_DIR, cfg.GRADE_WEIGHT))
    model_grade.eval()

    # Results: all_aucs[task][split][level_label] = [50 AUC values]
    all_aucs = {'MVI': {}, 'Grade': {}}
    for task in ['MVI', 'Grade']:
        for s in ['val', 'ext1', 'ext2']:
            all_aucs[task][s] = {lbl: [] for lbl in cfg.PERTURB_LABELS}

    # Baseline (max_shift=0) performance, repeated 50 times
    print("
[*] Baseline test (no perturbation)...")
    for split_name in ['val', 'ext1', 'ext2']:
        y_mvi = np.array([p['mvi'] for p in data[split_name]])
        y_grade = np.array([p['grade'] for p in data[split_name]])

        logits_mvi = perturb_and_infer(model_mvi, data[split_name], 0, device)
        base_auc_mvi = roc_auc_score(y_mvi, 1/(1+np.exp(-logits_mvi[:, 0])))
        logits_grade = perturb_and_infer(model_grade, data[split_name], 0, device)
        base_auc_grade = roc_auc_score(y_grade, 1/(1+np.exp(-logits_grade[:, 1])))

        all_aucs['MVI'][split_name]['0%'] = [base_auc_mvi] * cfg.N_RUNS
        all_aucs['Grade'][split_name]['0%'] = [base_auc_grade] * cfg.N_RUNS
        print(f"  {split_name}: MVI={base_auc_mvi:.4f}, Grade={base_auc_grade:.4f}")

    # Perturbation test
    total_combos = len(cfg.PERTURB_LEVELS[1:]) * 3 * cfg.N_RUNS * 2
    pbar = tqdm(total=total_combos, desc='Perturbation test')

    for max_shift, label in zip(cfg.PERTURB_LEVELS[1:], cfg.PERTURB_LABELS[1:]):
        for split_name in ['val', 'ext1', 'ext2']:
            y_mvi = np.array([p['mvi'] for p in data[split_name]])
            y_grade = np.array([p['grade'] for p in data[split_name]])

            for run in range(cfg.N_RUNS):
                # MVI model -> MVI AUC
                logits_mvi = perturb_and_infer(model_mvi, data[split_name], max_shift, device)
                probs_mvi = 1.0 / (1.0 + np.exp(-logits_mvi[:, 0]))
                auc_mvi = roc_auc_score(y_mvi, probs_mvi) if len(np.unique(y_mvi)) > 1 else 0.5

                # Grade model -> Grade AUC
                logits_grade = perturb_and_infer(model_grade, data[split_name], max_shift, device)
                probs_grade = 1.0 / (1.0 + np.exp(-logits_grade[:, 1]))
                auc_grade = roc_auc_score(y_grade, probs_grade) if len(np.unique(y_grade)) > 1 else 0.5

                all_aucs['MVI'][split_name][label].append(auc_mvi)
                all_aucs['Grade'][split_name][label].append(auc_grade)

                pbar.update(2)

    pbar.close()

    # Summarize
    save_results(all_aucs, cfg)
    return all_aucs


def save_results(all_aucs, cfg):
    # Summary table
    rows = []
    for task in ['MVI', 'Grade']:
        for split_name in ['val', 'ext1', 'ext2']:
            for label in cfg.PERTURB_LABELS:
                aucs = all_aucs[task][split_name][label]
                rows.append({
                    'Task': task, 'Split': split_name, 'PerturbLevel': label,
                    'Mean': np.mean(aucs), 'Std': np.std(aucs),
                    'Median': np.median(aucs), 'Min': np.min(aucs), 'Max': np.max(aucs),
                    'Q1': np.percentile(aucs, 25), 'Q3': np.percentile(aucs, 75),
                })

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(cfg.OUTPUT_DIR, 'perturbation_auc_summary.xlsx'), index=False)
    print(f"
[*] AUC summary saved")

    # 50-run raw AUC data
    auc_records = []
    for task in ['MVI', 'Grade']:
        for split_name in ['val', 'ext1', 'ext2']:
            for label in cfg.PERTURB_LABELS:
                for run, auc_val in enumerate(all_aucs[task][split_name][label]):
                    auc_records.append({
                        'Task': task, 'Split': split_name, 'PerturbLevel': label, 'Run': run, 'AUC': auc_val
                    })
    df_aucs = pd.DataFrame(auc_records)
    df_aucs.to_excel(os.path.join(cfg.OUTPUT_DIR, 'perturbation_50runs_aucs.xlsx'), index=False)
    print("[*] 50-run raw AUC data saved")

    # Boxplot
    plot_boxplots(all_aucs, cfg)


def plot_boxplots(all_aucs, cfg):
    tasks = ['MVI', 'Grade']
    splits = ['val', 'ext1', 'ext2']
    colors = ['#4472C4', '#ED7D31', '#70AD47']

    # One task per column, one split per row
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, task in enumerate(tasks):
        for col, split_name in enumerate(splits):
            ax = axes[row, col]
            data = [all_aucs[task][split_name][l] for l in cfg.PERTURB_LABELS]

            bp = ax.boxplot(data, labels=cfg.PERTURB_LABELS, patch_artist=True,
                           widths=0.6, showfliers=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

            for patch, color in zip(bp['boxes'], ['#5B9BD5']*1 + ['#A5A5A5']*5):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # 0% reference line
            baseline = np.median(all_aucs[task][split_name]['0%'])
            ax.axhline(y=baseline, color='red', linestyle='--', linewidth=1, alpha=0.5)

            ax.set_title(f'{task} — {split_name}', fontsize=13, fontweight='bold')
            ax.set_ylabel('AUC', fontsize=11)
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Input Perturbation Robustness — LFM-Seq (50 runs)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(cfg.OUTPUT_DIR, 'perturbation_boxplot.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("[*] Boxplot saved")


if __name__ == '__main__':
    t0 = time.time()
    cfg = Config()
    run_robustness_test(cfg)
    print(f'\n[*] Total time: {(time.time()-t0)/60:.1f} min')
