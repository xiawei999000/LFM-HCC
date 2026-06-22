"""
Occlusion sensitivity visualization: 8x8 occlusion -> probability change -> heatmap -> 128x128 upsampling.
One folder per patient with original slice, 16x16 heatmap, 128x128 heatmap, and composite figure.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import SimpleITK as sitk
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from timm.models.vision_transformer import vit_small_patch8_224
from scipy.ndimage import zoom
from config import (PROJECT_ROOT, MODEL_DIR, DATA_DIR, OUTPUT_DIR,
                     CLINICAL_INFO_PATH, PRED_RFS_PATH, PRED_MVI_GRADE_PATH)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT = os.path.join(OUTPUT_DIR, 'visualization')

# User: fill in representative case IDs and descriptions
CASES = [
    # {'id': '...', 'label': 'Case 1', 'desc': '...'},
]

MODEL_PATH = os.path.join(MODEL_DIR, 'checkpoint.pth')
AP_DIR = os.path.join(DATA_DIR, 'ext_ap')
PV_DIR = os.path.join(DATA_DIR, 'ext_pv')
RISK_DF = pd.read_excel(PRED_RFS_PATH)
PRED_DF = pd.read_excel(PRED_MVI_GRADE_PATH)
CLI_DF = pd.read_excel(CLINICAL_INFO_PATH)
CLI_DF.columns = CLI_DF.columns.str.strip()

OCCLUSION_SIZE = 8  # occlusion block size in pixels
STRIDE = 8          # stride (8 gives 16×16 grid)
GRID_SIZE = 128 // STRIDE  # 8


class CrossPhaseAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm_ap = nn.LayerNorm(dim); self.norm_pv = nn.LayerNorm(dim)
        self.ap_query_pv = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.pv_query_ap = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
    def forward(self, ap_feat, pv_feat):
        ap_n, pv_n = self.norm_ap(ap_feat), self.norm_pv(pv_feat)
        ap_c, _ = self.ap_query_pv(ap_n, pv_n, pv_n)
        pv_c, _ = self.pv_query_ap(pv_n, ap_n, ap_n)
        return ap_feat + ap_c, pv_feat + pv_c

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

class MILModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vit_small_patch8_224(img_size=128, in_chans=1, num_classes=0,
                                              pretrained=False, dynamic_img_size=True, init_values=1e-5)
        D = 384; fusion_dim = D * 2
        self.cross_phase_attn = CrossPhaseAttention(dim=D)
        self.attention = GatedAttention(fusion_dim)
        self.head_mvi = nn.Sequential(nn.Linear(fusion_dim, fusion_dim//2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(fusion_dim//2, 1))
        self.head_grade = nn.Sequential(nn.Linear(fusion_dim, fusion_dim//2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(fusion_dim//2, 1))
        self.log_vars = nn.Parameter(torch.zeros(2))

    def load_weights(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(ckpt, strict=True)

    def forward_one_patient(self, ap_tensor, pv_tensor):
        feat_ap = self.backbone(ap_tensor); feat_pv = self.backbone(pv_tensor)
        feat_ap = feat_ap.unsqueeze(0); feat_pv = feat_pv.unsqueeze(0)
        feat_ap, feat_pv = self.cross_phase_attn(feat_ap, feat_pv)
        feat_ap = feat_ap.squeeze(0); feat_pv = feat_pv.squeeze(0)
        feat_fused = torch.cat([feat_ap, feat_pv], dim=1)
        z, a = self.attention(feat_fused)
        self.slice_weights = a.squeeze(-1).detach().cpu().numpy()
        out_mvi = self.head_mvi(z).view(-1)
        out_grade = self.head_grade(z).view(-1)
        return out_mvi, out_grade


def load_volume(pid, phase):
    d = AP_DIR if phase == 'AP' else PV_DIR
    img = sitk.ReadImage(os.path.join(d, pid + '.nii.gz'))
    return sitk.GetArrayFromImage(img).astype(np.float32) / 255.0

def get_metadata(pid):
    r = RISK_DF[(RISK_DF['Split']=='ext2')&(RISK_DF['ID']==int(pid))]
    p = PRED_DF[(PRED_DF['Split']=='ext2')&(PRED_DF['ID']==int(pid))]
    c = CLI_DF[(CLI_DF['Split']=='ext2')&(CLI_DF['ID']==int(pid))]
    return {
        'mvi_true': int(p['MVI'].values[0]), 'grade_true': int(p['PathologicalGrade'].values[0]),
        'size': c['tumor_max_size_mm'].values[0],
        'risk': r['DINOv2_Continue_risk'].values[0] if len(r)>0 else np.nan,
        'rfs': r['RFS'].values[0] if len(r)>0 else np.nan,
        'rfs_status': int(r['RFS_status'].values[0]) if len(r)>0 else np.nan,
    }


def compute_occlusion(model, ap_vol, pv_vol, slice_idx, phase, task):
    """Apply 8x8 occlusion on specified slice; return (8,8) heatmap and baseline prob."""
    D = ap_vol.shape[0]
    ap_t = torch.from_numpy(ap_vol.copy()).unsqueeze(1).to(DEVICE)
    pv_t = torch.from_numpy(pv_vol.copy()).unsqueeze(1).to(DEVICE)

    # Baseline
    with torch.no_grad():
        out_mvi, out_grade = model.forward_one_patient(ap_t, pv_t)
    base_mvi = torch.sigmoid(out_mvi[0]).item()
    base_grade = torch.sigmoid(out_grade[0]).item()
    base = base_mvi if task == 'mvi' else base_grade

    occl_map = np.zeros((GRID_SIZE, GRID_SIZE))
    vol_src = ap_vol if phase == 'ap' else pv_vol

    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            y1, x1 = gy * STRIDE, gx * STRIDE
            y2, x2 = y1 + OCCLUSION_SIZE, x1 + OCCLUSION_SIZE

            vol_occluded = vol_src.copy()
            vol_occluded[slice_idx, y1:y2, x1:x2] = 0.0

            ap_occ = torch.from_numpy(ap_vol.copy()).unsqueeze(1).to(DEVICE)
            pv_occ = torch.from_numpy(pv_vol.copy()).unsqueeze(1).to(DEVICE)
            if phase == 'ap':
                ap_occ[slice_idx, 0, y1:y2, x1:x2] = 0.0
            else:
                pv_occ[slice_idx, 0, y1:y2, x1:x2] = 0.0

            with torch.no_grad():
                om, og = model.forward_one_patient(ap_occ, pv_occ)
            prob_occ = torch.sigmoid(om[0]).item() if task == 'mvi' else torch.sigmoid(og[0]).item()

            occl_map[gy, gx] = base - prob_occ  # positive = occlusion reduces prob (model relied on this region)

    return occl_map, base_mvi, base_grade, base


def visualize_case(case):
    pid = case['id']
    print('[*] ID=%s' % pid)
    case_dir = os.path.join(OUT, 'ID_' + pid)
    os.makedirs(case_dir, exist_ok=True)

    ap_vol = load_volume(pid, 'AP')
    pv_vol = load_volume(pid, 'PV')
    meta = get_metadata(pid)
    D = ap_vol.shape[0]
    n_show = min(5, D)
    slice_indices = np.linspace(0, D-1, n_show).astype(int)

    model = MILModel().to(DEVICE)
    model.load_weights(MODEL_PATH)
    model.eval()

    # Baseline
    with torch.no_grad():
        ap_t = torch.from_numpy(ap_vol.copy()).unsqueeze(1).to(DEVICE)
        pv_t = torch.from_numpy(pv_vol.copy()).unsqueeze(1).to(DEVICE)
        out_mvi, out_grade = model.forward_one_patient(ap_t, pv_t)
    prob_mvi = torch.sigmoid(out_mvi[0]).item()
    prob_grade = torch.sigmoid(out_grade[0]).item()
    slice_w = model.slice_weights.copy()

    # Compute occlusion maps for shown slices
    maps_ap_mvi = []; maps_ap_grade = []; maps_pv_mvi = []; maps_pv_grade = []
    for s_idx in slice_indices:
        hm_ap_mvi, _, _, _ = compute_occlusion(model, ap_vol, pv_vol, s_idx, 'ap', 'mvi')
        maps_ap_mvi.append(hm_ap_mvi)
        hm_ap_grade, _, _, _ = compute_occlusion(model, ap_vol, pv_vol, s_idx, 'ap', 'grade')
        maps_ap_grade.append(hm_ap_grade)
        hm_pv_mvi, _, _, _ = compute_occlusion(model, ap_vol, pv_vol, s_idx, 'pv', 'mvi')
        maps_pv_mvi.append(hm_pv_mvi)
        hm_pv_grade, _, _, _ = compute_occlusion(model, ap_vol, pv_vol, s_idx, 'pv', 'grade')
        maps_pv_grade.append(hm_pv_grade)

    del model; torch.cuda.empty_cache()

    # Save individual maps and upsampled versions
    all_cams = {'ap_mvi': maps_ap_mvi, 'ap_grade': maps_ap_grade,
                'pv_mvi': maps_pv_mvi, 'pv_grade': maps_pv_grade}

    for name, maps in all_cams.items():
        for i, (s_idx, hm) in enumerate(zip(slice_indices, maps)):
            # Save 8x8 raw
            np.save(os.path.join(case_dir, 'occlusion_%s_slice%d_8x8.npy' % (name, s_idx+1)), hm)
            # Save 8x8 as image
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(hm, cmap='hot', vmin=0, vmax=max(hm.max(), 1e-6))
            ax.set_title('%s slice %d (8x8)' % (name, s_idx+1), fontsize=8)
            ax.axis('off')
            fig.savefig(os.path.join(case_dir, 'occlusion_%s_slice%d_8x8.png' % (name, s_idx+1)), dpi=100, bbox_inches='tight')
            plt.close()
            # Save 128x128 upsampled
            hm_up = zoom(hm, 128/GRID_SIZE, order=1)
            np.save(os.path.join(case_dir, 'occlusion_%s_slice%d_128.npy' % (name, s_idx+1)), hm_up)
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(hm_up, cmap='RdBu_r', vmin=-np.abs(hm_up).max(), vmax=np.abs(hm_up).max())
            ax.set_title('%s slice %d (128x128)' % (name, s_idx+1), fontsize=8)
            ax.axis('off')
            fig.savefig(os.path.join(case_dir, 'occlusion_%s_slice%d_128.png' % (name, s_idx+1)), dpi=100, bbox_inches='tight')
            plt.close()

    # Save original slices
    for phase, vol in [('ap', ap_vol), ('pv', pv_vol)]:
        for s_idx in slice_indices:
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(vol[s_idx], cmap='gray', vmin=0, vmax=1)
            ax.set_title('%s slice %d' % (phase.upper(), s_idx+1), fontsize=8)
            ax.axis('off')
            fig.savefig(os.path.join(case_dir, 'CT_%s_slice%d.png' % (phase, s_idx+1)), dpi=100, bbox_inches='tight')
            plt.close()

    # Global vmin/vmax for heatmap overlay (positive only)
    all_vals = []
    for maps in all_cams.values():
        for hm in maps:
            hm_up = zoom(hm, 128/GRID_SIZE, order=1)
            all_vals.append(hm_up[hm_up > 0].flatten())
    vmax = 0.01
    if all_vals:
        all_vals = np.concatenate(all_vals)
        if len(all_vals) > 0:
            vmax = np.percentile(all_vals, 98) if len(all_vals) > 10 else all_vals.max()
            if vmax <= 1e-8:
                vmax = 0.01

    # ===== Combined figure =====
    fig = plt.figure(figsize=(3.5 + n_show * 2.8, 10.5))
    n_img_cols = n_show
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[n_img_cols, 0.45], wspace=0.02)
    gs_left = GridSpecFromSubplotSpec(4, n_img_cols, subplot_spec=gs_main[0], hspace=0.3, wspace=0.04)
    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1], hspace=0.3, height_ratios=[1, 1])

    row_names = ['(a) AP — MVI Occlusion', '(b) AP — Grade Occlusion',
                 '(c) PV — MVI Occlusion', '(d) PV — Grade Occlusion']
    row_data = [('ap_mvi', 'ap', ap_vol), ('ap_grade', 'ap', ap_vol),
                ('pv_mvi', 'pv', pv_vol), ('pv_grade', 'pv', pv_vol)]

    for row, (name, phase, vol) in enumerate(row_data):
        maps = all_cams[name]
        for i, (s_idx, hm) in enumerate(zip(slice_indices, maps)):
            ax = fig.add_subplot(gs_left[row, i])
            hm_up = zoom(hm, 128/GRID_SIZE, order=1)
            hm_up_pos = np.maximum(hm_up, 0)  # only positive (regions model relies on)
            ax.imshow(vol[s_idx], cmap='gray', vmin=0, vmax=1)
            ax.imshow(hm_up_pos, cmap='hot', alpha=0.6, vmin=0, vmax=vmax)
            ax.set_title('Slice %d/%d' % (s_idx+1, D), fontsize=7)
            ax.axis('off')

    # Slice weights (right top)
    ax_w = fig.add_subplot(gs_right[0])
    colors_w = ['red' if i in slice_indices else 'gray' for i in range(D)]
    ax_w.bar(range(D), slice_w, color=colors_w, width=0.6)
    ax_w.set_xlabel('Slice index', fontsize=8)
    ax_w.set_ylabel('Attention weight', fontsize=8)
    ax_w.set_title('Slice Attention (Gated MIL)', fontsize=9, fontweight='bold')
    for s_idx in slice_indices:
        ax_w.annotate(str(s_idx+1), (s_idx, slice_w[s_idx]),
                     textcoords='offset points', xytext=(0, 5), fontsize=7, color='red', ha='center')

    # Score card (right bottom) + colorbar
    ax_s = fig.add_subplot(gs_right[1])
    ax_s.axis('off')
    mvi_tag = 'TP' if meta['mvi_true']==1 and prob_mvi>=0.35 else ('FN' if meta['mvi_true']==1 else ('TN' if prob_mvi<0.35 else 'FP'))
    grade_tag = 'TP' if meta['grade_true']==1 and prob_grade>=0.5 else ('FN' if meta['grade_true']==1 else ('TN' if prob_grade<0.5 else 'FP'))
    risk_tag = 'High' if meta['risk']>=0.0215 else 'Low'
    rfs_str = 'event' if meta['rfs_status']==1 else 'no event'
    sz = meta['size']

    info = ('Patient: %s  |  %.0f mm\n' % (pid, sz) +
            '=' * 28 + '\n' +
            'MVI:  true=%d  pred=%.3f' % (meta['mvi_true'], prob_mvi) +
            ' (%s)\n' % mvi_tag +
            'Grade: true=%d  pred=%.3f' % (meta['grade_true'], prob_grade) +
            ' (%s)\n' % grade_tag +
            '-' * 28 + '\n' +
            'RFS: %.0f months (%s)\n' % (meta['rfs'], rfs_str) +
            'Risk score: %.3f' % meta['risk'] +
            ' [%s]' % risk_tag)
    ax_s.text(0.05, 0.72, info, transform=ax_s.transAxes, fontsize=7.5,
              va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85, edgecolor='gray'))

    # Colorbar below score
    cbar_ax = fig.add_axes([0.82, 0.08, 0.04, 0.12])
    norm = Normalize(vmin=0, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='hot')
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Delta prob', fontsize=7)
    cbar_ax.set_title('Occlusion\n(red=relied\nregion)', fontsize=6)

    fig.suptitle('%s\n%s' % (case['label'], case['desc']), fontsize=10, fontweight='bold', y=1.005)
    save_path = os.path.join(case_dir, 'combined_figure.png')
    fig.savefig(save_path, dpi=250, facecolor='white', bbox_inches='tight')
    fig.savefig(os.path.join(case_dir, 'combined_figure.pdf'), facecolor='white', bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % save_path)


if __name__ == '__main__':
    for case in CASES:
        visualize_case(case)
    print('\n[*] Done')
