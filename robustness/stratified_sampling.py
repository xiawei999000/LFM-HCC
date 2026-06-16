"""
DINOv2 Stratified Sampling Test Pipeline
=========================================
Tests downstream fine-tuning performance with stratified random sampling
at 10%, 20%, 50% of training data (50 repeated runs each), evaluating
MVI and PathologicalGrade AUC on val/ext1/ext2 sets.

Usage:
    python test_stratified_sampling.py --model_name dinov2
    python test_stratified_sampling.py --model_name deeplesion
    python test_stratified_sampling.py --model_name combine
    python test_stratified_sampling.py --model_name continue

Models:
    dinov2      - Official DINOv2 ViT-S/14 (224x224, 3ch)
    deeplesion  - DeepLesion pretrained ViT-S/8 (128x128, 1ch)
    combine     - Combined pretrained ViT-S/8 (128x128, 1ch)
    continue    - Continue pretrained ViT-S/8 (128x128, 1ch)
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
import SimpleITK as sitk
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import timm
import warnings
warnings.filterwarnings('ignore')


# ==========================================
# Model Configurations
# ==========================================
MODEL_CONFIGS = {
    'dinov2': {
        'backbone_name': 'vit_small_patch14_dinov2.lvd142m',
        'weights_file': 'dinov2_vits14_pretrain.pth',
        'img_size': 224,
        'in_chans': 3,
        'lr': 1e-3,
        'pretrain_type': 'official',
        'use_timm_create': True,
    },
    'deeplesion': {
        'backbone_name': 'vit_small_patch8_224',
        'weights_file': 'deeplesion_dinov2_epoch047_loss7.8206_std0.0428.pth',
        'img_size': 128,
        'in_chans': 1,
        'lr': 3e-5,
        'pretrain_type': 'deeplesion',
        'use_timm_create': False,
    },
    'combine': {
        'backbone_name': 'vit_small_patch8_224',
        'weights_file': 'Combine_dinov2_epoch055_loss8.1210_std0.0430.pth',
        'img_size': 128,
        'in_chans': 1,
        'lr': 3e-5,
        'pretrain_type': 'combine',
        'use_timm_create': False,
    },
    'continue': {
        'backbone_name': 'vit_small_patch8_224',
        'weights_file': 'Continue_dinov2_epoch000_loss7.6499_std0.0391_best.pth',
        'img_size': 128,
        'in_chans': 1,
        'lr': 3e-5,
        'pretrain_type': 'continue',
        'use_timm_create': False,
    },
}

# ==========================================
# Paths (relative to project root)
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_BASE = os.path.join(PROJECT_ROOT, 'training_test')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLINICAL_INFO_PATH
CLINICAL_EXCEL = CLINICAL_INFO_PATH

# ==========================================
# Training hyperparameters
# ==========================================
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 5
WEIGHT_DECAY = 1e-4
EMBED_DIM = 384
RATIOS = [0.1, 0.2, 0.5]
N_RUNS = 50
SEED_BASE = 42


# ==========================================
# Dataset
# ==========================================
class HCCDataset(data.Dataset):
    def __init__(self, df, ap_dir, pv_dir, img_size=128, in_chans=1, desc="Loading"):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.in_chans = in_chans
        self.data_cache = []

        print(f"[*] Loading {desc} into RAM (size={img_size}, ch={in_chans})...")
        for idx in tqdm(range(len(self.df)), desc=desc):
            row = self.df.iloc[idx]
            pid = str(row['ID'])
            label = torch.tensor(
                [float(row['MVI']), float(row['PathologicalGrade'])],
                dtype=torch.float32
            )

            ap_path = os.path.join(ap_dir, f"{pid}.nii.gz")
            pv_path = os.path.join(pv_dir, f"{pid}.nii.gz")

            ap_tensor = self._load_nii(ap_path)
            pv_tensor = self._load_nii(pv_path)

            self.data_cache.append({
                'ap': ap_tensor,
                'pv': pv_tensor,
                'label': label,
                'pid': pid
            })

    def __len__(self):
        return len(self.data_cache)

    def _load_nii(self, file_path):
        img = sitk.ReadImage(file_path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if arr.max() > 1:
            arr = arr / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(1)  # (N,1,H,W)
        tensor = F.interpolate(
            tensor,
            size=(self.img_size, self.img_size),
            mode='bilinear',
            align_corners=False
        )
        if self.in_chans == 3:
            tensor = tensor.repeat(1, 3, 1, 1)  # (N,3,H,W)
        return tensor

    def __getitem__(self, idx):
        item = self.data_cache[idx]
        return item['ap'], item['pv'], item['label'], item['pid']


def my_collate(batch):
    ap_list, pv_list, labels, pids = [], [], [], []
    for ap, pv, label, pid in batch:
        ap_list.append(ap)
        pv_list.append(pv)
        labels.append(label)
        pids.append(pid)
    labels = torch.stack(labels, dim=0)
    return ap_list, pv_list, labels, pids


# ==========================================
# Model Components
# ==========================================
class CrossPhaseAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm_ap = nn.LayerNorm(dim)
        self.norm_pv = nn.LayerNorm(dim)
        self.ap_query_pv = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.pv_query_ap = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

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
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        if model_config['use_timm_create']:
            self.backbone = timm.create_model(
                model_config['backbone_name'],
                pretrained=False,
                img_size=model_config['img_size'],
                in_chans=model_config['in_chans'],
                num_classes=0,
                dynamic_img_size=True,
            )
            self._load_official_dinov2_weights(model_config['weights_file'])
        else:
            from timm.models.vision_transformer import vit_small_patch8_224
            self.backbone = vit_small_patch8_224(
                img_size=model_config['img_size'],
                in_chans=model_config['in_chans'],
                num_classes=0,
                pretrained=False,
                dynamic_img_size=True,
                init_values=1e-5,
            )
            self._load_custom_pretrained_weights(model_config['weights_file'])

        D = EMBED_DIM
        fusion_dim = D * 2

        self.cross_phase_attn = CrossPhaseAttention(dim=D, num_heads=4, dropout=0.1)
        self.attention = GatedAttention(fusion_dim)

        self.head_mvi = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, 1)
        )
        self.head_grade = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, 1)
        )

        self.log_vars = nn.Parameter(torch.zeros(2))

    # ---- Official DINOv2 weight loading (from 0423 reference) ----
    @staticmethod
    def _strip_prefix(state_dict, prefixes):
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_k.startswith(prefix):
                        new_k = new_k[len(prefix):]
                        changed = True
            new_sd[new_k] = v
        return new_sd

    def _resize_pos_embed(self, ckpt_pos_embed, model_pos_embed):
        if ckpt_pos_embed.shape == model_pos_embed.shape:
            return ckpt_pos_embed
        if ckpt_pos_embed.ndim != 3 or model_pos_embed.ndim != 3:
            return None

        cls_tokens = ckpt_pos_embed[:, :1, :]
        patch_pos = ckpt_pos_embed[:, 1:, :]

        old_num = patch_pos.shape[1]
        new_num = model_pos_embed.shape[1] - 1
        embed_dim = patch_pos.shape[2]

        old_grid = int(math.sqrt(old_num))
        new_grid = int(math.sqrt(new_num))

        if old_grid * old_grid != old_num or new_grid * new_grid != new_num:
            return None

        patch_pos = patch_pos.reshape(1, old_grid, old_grid, embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(new_grid, new_grid), mode='bicubic', align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, embed_dim)
        return torch.cat([cls_tokens, patch_pos], dim=1)

    def _load_official_dinov2_weights(self, weight_file):
        weight_path = os.path.join(MODELS_DIR, weight_file)
        if not os.path.exists(weight_path):
            print(f"[!] Weight file not found: {weight_path}, using random init.")
            return

        ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)

        if isinstance(ckpt, dict):
            for candidate in ['model_state_dict', 'state_dict', 'model', 'teacher']:
                if candidate in ckpt:
                    state_dict = ckpt[candidate]
                    break
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        prefixes = ['module.', 'model.', 'backbone.', 'student.', 'teacher.']
        state_dict = self._strip_prefix(state_dict, prefixes)

        model_dict = self.backbone.state_dict()

        if 'pos_embed' in state_dict and 'pos_embed' in model_dict:
            if state_dict['pos_embed'].shape != model_dict['pos_embed'].shape:
                resized = self._resize_pos_embed(state_dict['pos_embed'], model_dict['pos_embed'])
                if resized is not None:
                    state_dict['pos_embed'] = resized
                    print(f"[*] pos_embed interpolated: {tuple(resized.shape)}")
                else:
                    state_dict.pop('pos_embed')

        matched = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched[k] = v
            else:
                skipped.append(k)

        msg = self.backbone.load_state_dict(matched, strict=False)
        print(f"[*] Loaded official DINOv2 weights: {weight_file}")
        print(f"    matched={len(matched)}, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        if skipped:
            print(f"    skipped keys (first 10): {skipped[:10]}")

    # ---- Custom pretrained weight loading (from 0508 reference) ----
    def _load_custom_pretrained_weights(self, weight_file):
        weight_path = os.path.join(MODELS_DIR, weight_file)
        if not os.path.exists(weight_path):
            print(f"[!] Weight file not found: {weight_path}, using random init.")
            return

        ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
        full = ckpt.get('model_state_dict', ckpt)

        sub = {}
        for k, v in full.items():
            if k.startswith('teacher_backbone.vit.'):
                sub[k.replace('teacher_backbone.vit.', '')] = v

        msg = self.backbone.load_state_dict(sub, strict=False)
        print(f"[*] Loaded custom pretrained weights: {weight_file}")
        print(f"    missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

    def forward_one_patient(self, ap_tensor, pv_tensor):
        feat_ap = self.backbone(ap_tensor)
        feat_pv = self.backbone(pv_tensor)

        feat_ap = feat_ap.unsqueeze(0)
        feat_pv = feat_pv.unsqueeze(0)

        feat_ap, feat_pv = self.cross_phase_attn(feat_ap, feat_pv)

        feat_ap = feat_ap.squeeze(0)
        feat_pv = feat_pv.squeeze(0)

        n = min(feat_ap.size(0), feat_pv.size(0))
        feat_ap = feat_ap[:n]
        feat_pv = feat_pv[:n]

        feat_fused = torch.cat([feat_ap, feat_pv], dim=1)
        z, _ = self.attention(feat_fused)

        out_mvi = self.head_mvi(z).view(-1)
        out_grade = self.head_grade(z).view(-1)

        return torch.cat([out_mvi, out_grade], dim=0)

    def forward(self, ap_list, pv_list):
        outputs = []
        for ap, pv in zip(ap_list, pv_list):
            outputs.append(self.forward_one_patient(ap, pv))
        return torch.stack(outputs, dim=0)


# ==========================================
# Loss
# ==========================================
def compute_loss(logits, labels, log_vars):
    pred_mvi = logits[:, 0].view(-1)
    pred_grade = logits[:, 1].view(-1)
    target_mvi = labels[:, 0].view(-1)
    target_grade = labels[:, 1].view(-1)

    loss_mvi = F.binary_cross_entropy_with_logits(pred_mvi, target_mvi)
    loss_grade = F.binary_cross_entropy_with_logits(pred_grade, target_grade)

    prec_mvi = torch.exp(-log_vars[0])
    prec_grade = torch.exp(-log_vars[1])

    total = prec_mvi * loss_mvi + log_vars[0] + prec_grade * loss_grade + log_vars[1]
    return total, loss_mvi.detach(), loss_grade.detach()


# ==========================================
# Evaluation
# ==========================================
def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for ap_list, pv_list, labels, _ in dataloader:
            ap_list = [t.to(device) for t in ap_list]
            pv_list = [t.to(device) for t in pv_list]
            labels = labels.to(device)

            logits = model(ap_list, pv_list)
            loss, _, _ = compute_loss(logits, labels, model.log_vars)

            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    aucs = []
    for i in range(all_labels.shape[1]):
        try:
            aucs.append(roc_auc_score(all_labels[:, i], all_probs[:, i]))
        except ValueError:
            aucs.append(0.5)

    return aucs, avg_loss


# ==========================================
# Training for one (ratio, run)
# ==========================================
def train_one_run(model_config, train_ds, val_loader, ext1_loader, ext2_loader, device, run_seed):
    """Train model on the given training subset, return best AUCs on val/ext1/ext2."""
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    lr = model_config['lr']
    model = MultiTaskMILModel(model_config).to(device)

    train_loader = data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=my_collate, num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr},
        {'params': model.cross_phase_attn.parameters(), 'lr': lr},
        {'params': model.attention.parameters(), 'lr': lr},
        {'params': model.head_mvi.parameters(), 'lr': lr},
        {'params': model.head_grade.parameters(), 'lr': lr},
        {'params': [model.log_vars], 'lr': lr},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0
    best_aucs = {'val_mvi': 0.5, 'val_grade': 0.5, 'ext1_mvi': 0.5, 'ext1_grade': 0.5, 'ext2_mvi': 0.5, 'ext2_grade': 0.5}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for ap_list, pv_list, labels, _ in train_loader:
            ap_list = [t.to(device) for t in ap_list]
            pv_list = [t.to(device) for t in pv_list]
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(ap_list, pv_list)
            loss, _, _ = compute_loss(logits, labels, model.log_vars)
            loss.backward()
            optimizer.step()

        scheduler.step()

        auc_val, val_loss = evaluate(model, val_loader, device)
        auc_ext1, _ = evaluate(model, ext1_loader, device)
        auc_ext2, _ = evaluate(model, ext2_loader, device)

        if auc_val[0] > best_aucs['val_mvi']:
            best_aucs['val_mvi'] = auc_val[0]
            best_aucs['ext1_mvi'] = auc_ext1[0]
            best_aucs['ext2_mvi'] = auc_ext2[0]
        if auc_val[1] > best_aucs['val_grade']:
            best_aucs['val_grade'] = auc_val[1]
            best_aucs['ext1_grade'] = auc_ext1[1]
            best_aucs['ext2_grade'] = auc_ext2[1]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_aucs


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['dinov2', 'deeplesion', 'combine', 'continue'])
    parser.add_argument('--n_runs', type=int, default=N_RUNS)
    parser.add_argument('--ratios', type=float, nargs='+', default=RATIOS)
    parser.add_argument('--seed_base', type=int, default=SEED_BASE)
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model_name]
    model_name = args.model_name

    print("=" * 80)
    print(f"Stratified Sampling Test: {model_name}")
    print(f"  Backbone: {model_config['backbone_name']}")
    print(f"  Input: {model_config['img_size']}x{model_config['img_size']}x{model_config['in_chans']}")
    print(f"  LR: {model_config['lr']}")
    print(f"  Weights: {model_config['weights_file']}")
    print(f"  Ratios: {args.ratios}, N_Runs: {args.n_runs}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    # ---- Load full data ----
    df = pd.read_excel(CLINICAL_EXCEL)
    df['Split'] = df['Split'].astype(str).str.strip().str.lower()

    def load_split(split_name):
        sub = df[df['Split'] == split_name]
        ap_dir = os.path.join(DATA_DIR, f'{split_name}_ap')
        pv_dir = os.path.join(DATA_DIR, f'{split_name}_pv')
        valid_mask = [
            os.path.exists(os.path.join(ap_dir, f"{pid}.nii.gz")) and
            os.path.exists(os.path.join(pv_dir, f"{pid}.nii.gz"))
            for pid in sub['ID'].astype(str)
        ]
        sub = sub[valid_mask].reset_index(drop=True)
        ds = HCCDataset(sub, ap_dir, pv_dir,
                        img_size=model_config['img_size'],
                        in_chans=model_config['in_chans'],
                        desc=f'{split_name} set')
        return ds

    full_train_ds = load_split('train')
    val_ds = load_split('val')
    ext1_ds = load_split('ext1')
    ext2_ds = load_split('ext2')

    print(f"[*] Loaded: train={len(full_train_ds)}, val={len(val_ds)}, ext1={len(ext1_ds)}, ext2={len(ext2_ds)}")

    loader_kwargs = dict(batch_size=BATCH_SIZE, collate_fn=my_collate, num_workers=0, pin_memory=True)
    val_loader = data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
    ext1_loader = data.DataLoader(ext1_ds, shuffle=False, **loader_kwargs)
    ext2_loader = data.DataLoader(ext2_ds, shuffle=False, **loader_kwargs)

    # ---- Output file ----
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    output_csv = os.path.join(OUTPUT_BASE, f'results_{model_name}.csv')
    columns = ['model', 'ratio', 'run', 'val_mvi_auc', 'val_grade_auc',
               'ext1_mvi_auc', 'ext1_grade_auc', 'ext2_mvi_auc', 'ext2_grade_auc']

    # Load existing progress for checkpointing
    done_set = set()
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        done_set = set(zip(existing['ratio'], existing['run']))
        print(f"[*] Found {len(done_set)} completed runs in {output_csv}")

    # ---- Prepare stratification labels ----
    full_indices = np.arange(len(full_train_ds))
    full_labels_mvi = full_train_ds.df['MVI'].values
    full_labels_grade = full_train_ds.df['PathologicalGrade'].values
    # Combined stratification: 4 classes
    stratify_labels = [f'{int(mvi)}_{int(grd)}' for mvi, grd in zip(full_labels_mvi, full_labels_grade)]

    # ---- Run experiments ----
    total_runs = len(args.ratios) * args.n_runs
    completed = 0

    for ratio in args.ratios:
        n_samples = max(int(len(full_train_ds) * ratio), 2)
        print(f"\n{'='*60}")
        print(f"Ratio {ratio:.0%}: {n_samples} training samples, {args.n_runs} runs")
        print(f"{'='*60}")

        for run_id in range(1, args.n_runs + 1):
            if (ratio, run_id) in done_set:
                print(f"  [{ratio:.0%}] Run {run_id}/{args.n_runs} - already done, skipping")
                completed += 1
                continue

            seed = args.seed_base + int(ratio * 100) + run_id
            run_start = time.time()

            # Stratified subsample
            sub_indices, _ = train_test_split(
                full_indices,
                train_size=n_samples,
                stratify=stratify_labels,
                random_state=seed
            )

            # Create subset dataset
            sub_df = full_train_ds.df.iloc[sub_indices].reset_index(drop=True)
            sub_ap_dir = os.path.join(DATA_DIR, 'train_ap')
            sub_pv_dir = os.path.join(DATA_DIR, 'train_pv')
            sub_ds = HCCDataset(sub_df, sub_ap_dir, sub_pv_dir,
                                img_size=model_config['img_size'],
                                in_chans=model_config['in_chans'],
                                desc=f"train {ratio:.0%} run {run_id}")

            # Train
            aucs = train_one_run(model_config, sub_ds, val_loader, ext1_loader, ext2_loader, device, seed)

            # Save
            row = {
                'model': model_name,
                'ratio': ratio,
                'run': run_id,
                'val_mvi_auc': round(aucs['val_mvi'], 4),
                'val_grade_auc': round(aucs['val_grade'], 4),
                'ext1_mvi_auc': round(aucs['ext1_mvi'], 4),
                'ext1_grade_auc': round(aucs['ext1_grade'], 4),
                'ext2_mvi_auc': round(aucs['ext2_mvi'], 4),
                'ext2_grade_auc': round(aucs['ext2_grade'], 4),
            }

            df_row = pd.DataFrame([row])
            if not os.path.exists(output_csv):
                df_row.to_csv(output_csv, index=False)
            else:
                df_row.to_csv(output_csv, mode='a', header=False, index=False)

            completed += 1
            elapsed = time.time() - run_start
            print(f"  [{ratio:.0%}] Run {run_id}/{args.n_runs} done ({elapsed:.0f}s) | "
                  f"Val MVI={aucs['val_mvi']:.4f} Grade={aucs['val_grade']:.4f} | "
                  f"Ext1 MVI={aucs['ext1_mvi']:.4f} Grade={aucs['ext1_grade']:.4f} | "
                  f"Ext2 MVI={aucs['ext2_mvi']:.4f} Grade={aucs['ext2_grade']:.4f} | "
                  f"Progress: {completed}/{total_runs}")

    print(f"\n{'='*80}")
    print(f"All {completed}/{total_runs} runs completed for {model_name}!")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
