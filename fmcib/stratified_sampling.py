"""
NatureMI 3D FMCIB Stratified Sampling Test Pipeline
=====================================================
Tests downstream fine-tuning performance of the 3D FMCIB pretrained model
with stratified random sampling at 10%, 20%, 50% of training data
(50 repeated runs each), evaluating MVI and PathologicalGrade AUC
on val/ext1/ext2 sets.

Usage:
    conda activate fmcib
    python test_stratified_sampling.py
"""

import os
import sys
import time
import math
import argparse
import random
import warnings

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

warnings.filterwarnings('ignore')

from fmcib.models import fmcib_model


# ==========================================
# Config
# ==========================================
class Config:
    PATCH_SIZE = (50, 50, 50)
    INPUT_CHANNELS = 1
    CLIP_MIN = 0.0
    CLIP_MAX = 400.0
    NORMALIZE_MODE = "0_400"

    BATCH_SIZE = 32
    LR = 3e-5
    WEIGHT_DECAY = 1e-4
    EPOCHS = 10
    PATIENCE = 3

    PROJ_DIM = 512
    CROSS_ATTN_HEADS = 4
    DROPOUT = 0.3
    LABEL_NAMES = ["MVI", "PathologicalGrade"]

    MODEL_NAME = "fmcib_3dresnet"


# ==========================================
# Paths
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "patches_3d")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLINICAL_INFO_PATH
CLINICAL_EXCEL = CLINICAL_INFO_PATH
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "training_test")

RATIOS = [0.1, 0.2, 0.5]
N_RUNS = 50
SEED_BASE = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================
# Dataset
# ==========================================
class HCC3DDataset(data.Dataset):
    def __init__(self, df, ap_dir, pv_dir, cfg, desc="Loading"):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.data_cache = []

        print(f"[*] Loading {desc} into RAM...")
        for idx in tqdm(range(len(self.df)), desc=desc):
            row = self.df.iloc[idx]
            pid = str(row["ID"])
            label = torch.tensor(
                [float(row["MVI"]), float(row["PathologicalGrade"])],
                dtype=torch.float32
            )

            ap_path = os.path.join(ap_dir, f"{pid}_ap_patch.nii.gz")
            pv_path = os.path.join(pv_dir, f"{pid}_pv_patch.nii.gz")

            ap_tensor = self._load_nii(ap_path)
            pv_tensor = self._load_nii(pv_path)

            self.data_cache.append({
                "ap": ap_tensor,
                "pv": pv_tensor,
                "label": label,
                "pid": pid
            })

    def __len__(self):
        return len(self.data_cache)

    def _normalize(self, arr):
        arr = arr.astype(np.float32)
        arr = np.clip(arr, self.cfg.CLIP_MIN, self.cfg.CLIP_MAX)
        arr = (arr - self.cfg.CLIP_MIN) / max(self.cfg.CLIP_MAX - self.cfg.CLIP_MIN, 1e-6)
        return arr

    def _load_nii(self, file_path):
        img = sitk.ReadImage(file_path)
        arr = sitk.GetArrayFromImage(img)  # [D, H, W]
        arr = self._normalize(arr)
        tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]

        if tuple(tensor.shape[2:]) != self.cfg.PATCH_SIZE:
            tensor = F.interpolate(
                tensor, size=self.cfg.PATCH_SIZE, mode="trilinear", align_corners=False
            )
        tensor = tensor.squeeze(0)  # [1, D, H, W]
        return tensor.contiguous()

    def __getitem__(self, idx):
        item = self.data_cache[idx]
        return item["ap"], item["pv"], item["label"], item["pid"]


def collate_3d(batch):
    aps, pvs, labels, pids = zip(*batch)
    aps = torch.stack(aps, dim=0)
    pvs = torch.stack(pvs, dim=0)
    labels = torch.stack(labels, dim=0)
    return aps, pvs, labels, list(pids)


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
        ap_token = ap_feat.unsqueeze(1)
        pv_token = pv_feat.unsqueeze(1)
        ap_norm = self.norm_ap(ap_token)
        pv_norm = self.norm_pv(pv_token)
        ap_cross, _ = self.ap_query_pv(query=ap_norm, key=pv_norm, value=pv_norm)
        pv_cross, _ = self.pv_query_ap(query=pv_norm, key=ap_norm, value=ap_norm)
        ap_out = ap_token + ap_cross
        pv_out = pv_token + pv_cross
        return ap_out.squeeze(1), pv_out.squeeze(1)


class MultiTask3DModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.fm_model_ap = fmcib_model()
        self.fm_model_pv = fmcib_model()

        feat_dim_ap = self._infer_feature_dim(self.fm_model_ap)
        feat_dim_pv = self._infer_feature_dim(self.fm_model_pv)

        print(f"[*] AP feat dim: {feat_dim_ap}, PV feat dim: {feat_dim_pv}")

        self.proj_ap = nn.Sequential(
            nn.Linear(feat_dim_ap, cfg.PROJ_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT)
        )
        self.proj_pv = nn.Sequential(
            nn.Linear(feat_dim_pv, cfg.PROJ_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT)
        )

        self.cross_phase_attn = CrossPhaseAttention(
            dim=cfg.PROJ_DIM, num_heads=cfg.CROSS_ATTN_HEADS, dropout=0.1
        )

        fusion_dim = cfg.PROJ_DIM * 2
        self.head_mvi = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(fusion_dim // 2, 1)
        )
        self.head_grade = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(fusion_dim // 2, 1)
        )

        self.log_vars = nn.Parameter(torch.zeros(2))

    def _find_first_conv3d_in_channels(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv3d):
                return m.in_channels
        return None

    def _adapt_input_channels(self, x, encoder):
        expected_c = self._find_first_conv3d_in_channels(encoder)
        if expected_c is None:
            return x
        current_c = x.shape[1]
        if current_c == expected_c:
            return x
        if current_c == 1 and expected_c == 3:
            return x.repeat(1, 3, 1, 1, 1)
        if current_c > expected_c:
            return x[:, :expected_c]
        repeat_times = (expected_c + current_c - 1) // current_c
        x = x.repeat(1, repeat_times, 1, 1, 1)
        return x[:, :expected_c]

    def _extract_feature(self, encoder, x):
        x = self._adapt_input_channels(x, encoder)
        if hasattr(encoder, "forward_features") and callable(getattr(encoder, "forward_features")):
            feat = encoder.forward_features(x)
        else:
            feat = encoder(x)

        if isinstance(feat, dict):
            for k in ["feat", "feature", "features", "embedding", "embeddings", "x", "out", "logits"]:
                if k in feat:
                    feat = feat[k]
                    break
            else:
                feat = list(feat.values())[0]

        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        if not torch.is_tensor(feat):
            raise ValueError("fmcib_model output cannot be parsed as Tensor")

        if feat.ndim == 5:
            feat = F.adaptive_avg_pool3d(feat, output_size=1).flatten(1)
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)
        elif feat.ndim == 2:
            pass
        else:
            feat = torch.flatten(feat, start_dim=1)
        return feat

    def _infer_feature_dim(self, encoder):
        was_training = encoder.training
        encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1,
                              self.cfg.PATCH_SIZE[0],
                              self.cfg.PATCH_SIZE[1],
                              self.cfg.PATCH_SIZE[2])
            feat = self._extract_feature(encoder, dummy)
        if was_training:
            encoder.train()
        return int(feat.shape[1])

    def forward(self, ap, pv):
        feat_ap = self._extract_feature(self.fm_model_ap, ap)
        feat_pv = self._extract_feature(self.fm_model_pv, pv)

        feat_ap = self.proj_ap(feat_ap)
        feat_pv = self.proj_pv(feat_pv)

        feat_ap, feat_pv = self.cross_phase_attn(feat_ap, feat_pv)

        feat_fused = torch.cat([feat_ap, feat_pv], dim=1)

        out_mvi = self.head_mvi(feat_fused).squeeze(1)
        out_grade = self.head_grade(feat_fused).squeeze(1)

        return torch.stack([out_mvi, out_grade], dim=1)


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
        for ap, pv, labels, _ in dataloader:
            ap = ap.to(device, non_blocking=True)
            pv = pv.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(ap, pv)
            loss, _, _ = compute_loss(logits, labels, model.log_vars)

            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    if len(all_probs) == 0:
        return [0.5, 0.5], avg_loss

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
# Training for one run
# ==========================================
def train_one_run(cfg, train_ds, val_loader, ext1_loader, ext2_loader, device, run_seed):
    set_seed(run_seed)

    model = MultiTask3DModel(cfg).to(device)

    train_loader = data.DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=collate_3d, num_workers=0, pin_memory=True
    )

    optimizer = torch.optim.AdamW([
        {"params": model.fm_model_ap.parameters(), "lr": cfg.LR},
        {"params": model.fm_model_pv.parameters(), "lr": cfg.LR},
        {"params": model.proj_ap.parameters(), "lr": cfg.LR},
        {"params": model.proj_pv.parameters(), "lr": cfg.LR},
        {"params": model.cross_phase_attn.parameters(), "lr": cfg.LR},
        {"params": model.head_mvi.parameters(), "lr": cfg.LR},
        {"params": model.head_grade.parameters(), "lr": cfg.LR},
        {"params": [model.log_vars], "lr": cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0
    best_aucs = {"val_mvi": 0.5, "val_grade": 0.5, "ext1_mvi": 0.5, "ext1_grade": 0.5,
                 "ext2_mvi": 0.5, "ext2_grade": 0.5}

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        for ap, pv, labels, _ in train_loader:
            ap = ap.to(device, non_blocking=True)
            pv = pv.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(ap, pv)
            loss, _, _ = compute_loss(logits, labels, model.log_vars)
            loss.backward()
            optimizer.step()

        scheduler.step()

        auc_val, val_loss = evaluate(model, val_loader, device)
        auc_ext1, _ = evaluate(model, ext1_loader, device)
        auc_ext2, _ = evaluate(model, ext2_loader, device)

        if auc_val[0] > best_aucs["val_mvi"]:
            best_aucs["val_mvi"] = auc_val[0]
            best_aucs["ext1_mvi"] = auc_ext1[0]
            best_aucs["ext2_mvi"] = auc_ext2[0]
        if auc_val[1] > best_aucs["val_grade"]:
            best_aucs["val_grade"] = auc_val[1]
            best_aucs["ext1_grade"] = auc_ext1[1]
            best_aucs["ext2_grade"] = auc_ext2[1]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= cfg.PATIENCE:
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
    parser.add_argument("--n_runs", type=int, default=N_RUNS)
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIOS)
    parser.add_argument("--seed_base", type=int, default=SEED_BASE)
    args = parser.parse_args()

    cfg = Config()
    model_name = cfg.MODEL_NAME

    print("=" * 80)
    print(f"Stratified Sampling Test: {model_name}")
    print(f"  Patch: {cfg.PATCH_SIZE}, Ch: {cfg.INPUT_CHANNELS}")
    print(f"  LR: {cfg.LR}, Epochs: {cfg.EPOCHS}, Patience: {cfg.PATIENCE}")
    print(f"  Ratios: {args.ratios}, N_Runs: {args.n_runs}")
    print("=" * 80)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    # ---- Load data ----
    df = pd.read_excel(CLINICAL_EXCEL)
    df["Split"] = df["Split"].astype(str).str.strip().str.lower()

    def load_split(split_name):
        sub = df[df["Split"] == split_name]
        ap_dir = os.path.join(DATA_ROOT, split_name, "ap")
        pv_dir = os.path.join(DATA_ROOT, split_name, "pv")
        valid_mask = [
            os.path.exists(os.path.join(ap_dir, f"{pid}_ap_patch.nii.gz")) and
            os.path.exists(os.path.join(pv_dir, f"{pid}_pv_patch.nii.gz"))
            for pid in sub["ID"].astype(str)
        ]
        sub = sub[valid_mask].reset_index(drop=True)
        ds = HCC3DDataset(sub, ap_dir, pv_dir, cfg, desc=f"{split_name} set")
        return ds

    full_train_ds = load_split("train")
    val_ds = load_split("val")
    ext1_ds = load_split("ext1")
    ext2_ds = load_split("ext2")

    print(f"[*] Loaded: train={len(full_train_ds)}, val={len(val_ds)}, ext1={len(ext1_ds)}, ext2={len(ext2_ds)}")

    loader_kwargs = dict(batch_size=cfg.BATCH_SIZE, collate_fn=collate_3d, num_workers=0, pin_memory=True)
    val_loader = data.DataLoader(val_ds, shuffle=False, **loader_kwargs)
    ext1_loader = data.DataLoader(ext1_ds, shuffle=False, **loader_kwargs)
    ext2_loader = data.DataLoader(ext2_ds, shuffle=False, **loader_kwargs)

    # ---- Output ----
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    output_csv = os.path.join(OUTPUT_BASE, f"results_{model_name}.csv")
    columns = ["model", "ratio", "run", "val_mvi_auc", "val_grade_auc",
               "ext1_mvi_auc", "ext1_grade_auc", "ext2_mvi_auc", "ext2_grade_auc"]

    done_set = set()
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        done_set = set(zip(existing["ratio"], existing["run"]))
        print(f"[*] Found {len(done_set)} completed runs in {output_csv}")

    # ---- Stratification labels ----
    full_indices = np.arange(len(full_train_ds))
    full_labels_mvi = full_train_ds.df["MVI"].values
    full_labels_grade = full_train_ds.df["PathologicalGrade"].values
    stratify_labels = [f"{int(mvi)}_{int(grd)}" for mvi, grd in zip(full_labels_mvi, full_labels_grade)]

    # ---- Run ----
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

            sub_indices, _ = train_test_split(
                full_indices, train_size=n_samples,
                stratify=stratify_labels, random_state=seed
            )

            sub_df = full_train_ds.df.iloc[sub_indices].reset_index(drop=True)
            sub_ap_dir = os.path.join(DATA_ROOT, "train", "ap")
            sub_pv_dir = os.path.join(DATA_ROOT, "train", "pv")
            sub_ds = HCC3DDataset(sub_df, sub_ap_dir, sub_pv_dir, cfg,
                                   desc=f"train {ratio:.0%} run {run_id}")

            aucs = train_one_run(cfg, sub_ds, val_loader, ext1_loader, ext2_loader, device, seed)

            row = {
                "model": model_name,
                "ratio": ratio,
                "run": run_id,
                "val_mvi_auc": round(aucs["val_mvi"], 4),
                "val_grade_auc": round(aucs["val_grade"], 4),
                "ext1_mvi_auc": round(aucs["ext1_mvi"], 4),
                "ext1_grade_auc": round(aucs["ext1_grade"], 4),
                "ext2_mvi_auc": round(aucs["ext2_mvi"], 4),
                "ext2_grade_auc": round(aucs["ext2_grade"], 4),
            }

            df_row = pd.DataFrame([row])
            if not os.path.exists(output_csv):
                df_row.to_csv(output_csv, index=False)
            else:
                df_row.to_csv(output_csv, mode="a", header=False, index=False)

            completed += 1
            elapsed = time.time() - run_start
            print(f"  [{ratio:.0%}] Run {run_id}/{args.n_runs} done ({elapsed:.0f}s) | "
                  f"Val MVI={aucs['val_mvi']:.4f} Grade={aucs['val_grade']:.4f} | "
                  f"Ext1 MVI={aucs['ext1_mvi']:.4f} Grade={aucs['ext1_grade']:.4f} | "
                  f"Ext2 MVI={aucs['ext2_mvi']:.4f} Grade={aucs['ext2_grade']:.4f} | "
                  f"Progress: {completed}/{total_runs}")

    print(f"\n{'='*80}")
    print(f"All {completed}/{total_runs} runs completed!")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
