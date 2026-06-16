"""
Inference for FMCIB fine-tuned models on HCC CT patches.
Supports two modes:
  - predict: output per-patient MVI and Grade probabilities
  - extract: output fused features (pre-classification head)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils import data as torch_data
from tqdm import tqdm

from fmcib.models import fmcib_model


# ============================================================
# Model components (must match training code exactly)
# ============================================================
class CrossPhaseAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm_ap = nn.LayerNorm(dim)
        self.norm_pv = nn.LayerNorm(dim)
        self.ap_query_pv = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.pv_query_ap = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

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
    def __init__(self, proj_dim=512, cross_attn_heads=4, dropout=0.3,
                 patch_size=(50, 50, 50), input_channels=1):
        super().__init__()
        self.fm_model_ap = fmcib_model()
        self.fm_model_pv = fmcib_model()
        self.patch_size = tuple(patch_size)
        self.input_channels = input_channels
        self.proj_dim = proj_dim

        feat_dim_ap = self._infer_feature_dim(self.fm_model_ap)
        feat_dim_pv = self._infer_feature_dim(self.fm_model_pv)
        print(f"[*] AP backbone output dim: {feat_dim_ap}")
        print(f"[*] PV backbone output dim: {feat_dim_pv}")

        self.proj_ap = nn.Sequential(
            nn.Linear(feat_dim_ap, proj_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout))
        self.proj_pv = nn.Sequential(
            nn.Linear(feat_dim_pv, proj_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout))

        self.cross_phase_attn = CrossPhaseAttention(
            dim=proj_dim, num_heads=cross_attn_heads, dropout=0.1)

        fusion_dim = proj_dim * 2
        self.head_mvi = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fusion_dim // 2, 1))
        self.head_grade = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fusion_dim // 2, 1))

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
        if hasattr(encoder, "forward_features") and callable(
            getattr(encoder, "forward_features")):
            feat = encoder.forward_features(x)
        else:
            feat = encoder(x)

        if isinstance(feat, dict):
            for k in ["feat", "feature", "features", "embedding", "embeddings",
                      "x", "out", "logits"]:
                if k in feat:
                    feat = feat[k]
                    break
            else:
                feat = list(feat.values())[0]

        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        if not torch.is_tensor(feat):
            raise ValueError("fmcib_model output could not be resolved to a Tensor.")
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
        encoder_was_training = encoder.training
        encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels,
                                self.patch_size[0], self.patch_size[1],
                                self.patch_size[2])
            feat = self._extract_feature(encoder, dummy)
        if encoder_was_training:
            encoder.train()
        return int(feat.shape[1])

    def forward_one(self, ap, pv):
        # ap, pv: [1, D, H, W] single-patient tensors
        feat_ap = self._extract_feature(self.fm_model_ap, ap)
        feat_pv = self._extract_feature(self.fm_model_pv, pv)
        feat_ap = self.proj_ap(feat_ap)
        feat_pv = self.proj_pv(feat_pv)
        feat_ap, feat_pv = self.cross_phase_attn(feat_ap, feat_pv)
        feat_fused = torch.cat([feat_ap, feat_pv], dim=1)
        return feat_fused

    def predict(self, ap, pv):
        feat_fused = self.forward_one(ap, pv)
        out_mvi = self.head_mvi(feat_fused).view(-1)
        out_grade = self.head_grade(feat_fused).view(-1)
        return torch.cat([out_mvi, out_grade], dim=0)

    def extract_feature(self, ap, pv):
        feat_fused = self.forward_one(ap, pv)
        return feat_fused.squeeze(0)


def load_model(checkpoint_path, device="cpu"):
    model = MultiTask3DModel(
        proj_dim=512, cross_attn_heads=4, dropout=0.3,
        patch_size=(50, 50, 50), input_channels=1)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Strip "module." prefix if present (DDP wrapper)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    print(f"[*] Loaded FMCIB model from: {checkpoint_path}")
    return model


# ============================================================
# Data loading
# ============================================================
class Patch3DDataset(torch_data.Dataset):
    def __init__(self, patient_ids, ap_dir, pv_dir,
                 patch_size=(50, 50, 50), clip_min=0.0, clip_max=400.0):
        self.patient_ids = list(patient_ids)
        self.patch_size = tuple(patch_size)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.data_cache = []
        for pid in tqdm(self.patient_ids, desc="Loading patches"):
            self.data_cache.append(self._load_patient(pid, ap_dir, pv_dir))

    def _normalize(self, arr):
        arr = arr.astype(np.float32)
        arr = np.clip(arr, self.clip_min, self.clip_max)
        arr = (arr - self.clip_min) / max(self.clip_max - self.clip_min, 1e-6)
        return arr

    def _load_patient(self, pid, ap_dir, pv_dir):
        ap_path = os.path.join(ap_dir, f"{pid}_ap_patch.nii.gz")
        pv_path = os.path.join(pv_dir, f"{pid}_pv_patch.nii.gz")

        ap_arr = sitk.GetArrayFromImage(sitk.ReadImage(ap_path))
        pv_arr = sitk.GetArrayFromImage(sitk.ReadImage(pv_path))

        ap_arr = self._normalize(ap_arr)
        pv_arr = self._normalize(pv_arr)

        ap_tensor = torch.from_numpy(ap_arr).float().unsqueeze(0).unsqueeze(0)
        pv_tensor = torch.from_numpy(pv_arr).float().unsqueeze(0).unsqueeze(0)

        if tuple(ap_tensor.shape[2:]) != self.patch_size:
            ap_tensor = F.interpolate(
                ap_tensor, size=self.patch_size, mode="trilinear",
                align_corners=False)
            pv_tensor = F.interpolate(
                pv_tensor, size=self.patch_size, mode="trilinear",
                align_corners=False)

        return ap_tensor.squeeze(0).contiguous(), pv_tensor.squeeze(0).contiguous(), pid

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        ap, pv, pid = self.data_cache[idx]
        return ap, pv, pid


def my_collate(batch):
    aps, pvs, pids = [], [], []
    for ap, pv, pid in batch:
        aps.append(ap)
        pvs.append(pv)
        pids.append(pid)
    return aps, pvs, pids


def get_patient_list(ap_dir, pv_dir, excel_path=None):
    ap_ids = set()
    pv_ids = set()
    for f in os.listdir(ap_dir):
        if f.endswith("_ap_patch.nii.gz"):
            ap_ids.add(f.replace("_ap_patch.nii.gz", ""))
    for f in os.listdir(pv_dir):
        if f.endswith("_pv_patch.nii.gz"):
            pv_ids.add(f.replace("_pv_patch.nii.gz", ""))
    common = sorted(ap_ids & pv_ids)

    if excel_path and os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        excel_ids = set(df["ID"].astype(str))
        common = sorted(excel_ids & set(common))

    return common


# ============================================================
# Inference modes
# ============================================================
@torch.no_grad()
def run_predict(model, dataloader, device):
    all_probs, all_pids = [], []
    for ap_list, pv_list, pids in dataloader:
        results = []
        for ap, pv in zip(ap_list, pv_list):
            ap = ap.to(device)
            pv = pv.to(device)
            logits = model.predict(ap, pv)
            probs = torch.sigmoid(logits).cpu().numpy()
            results.append(probs)
        all_probs.append(np.stack(results, axis=0))
        all_pids.extend(pids)

    probs = np.concatenate(all_probs, axis=0)
    df = pd.DataFrame({
        "ID": [int(p) for p in all_pids],
        "MVI_prediction": (probs[:, 0] >= 0.5).astype(int),
        "MVI_prob": probs[:, 0].round(4),
        "Grade_prediction": (probs[:, 1] >= 0.5).astype(int),
        "Grade_prob": probs[:, 1].round(4),
    })
    return df


@torch.no_grad()
def run_extract(model, dataloader, device):
    all_feats, all_pids = [], []
    for ap_list, pv_list, pids in dataloader:
        for ap, pv in zip(ap_list, pv_list):
            ap = ap.to(device)
            pv = pv.to(device)
            feat = model.extract_feature(ap, pv).cpu().numpy()
            all_feats.append(feat)
        all_pids.extend(pids)

    feats = np.stack(all_feats, axis=0)
    cols = [f"feat_{i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, columns=cols)
    df.insert(0, "ID", [int(p) for p in all_pids])
    return df


# ============================================================
# Entry point
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="FMCIB fine-tuned model inference on HCC CT patches")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned model checkpoint (.pth)")
    parser.add_argument("--ap_dir", type=str, required=True,
                        help="Directory of AP phase 3D patches (*_ap_patch.nii.gz)")
    parser.add_argument("--pv_dir", type=str, required=True,
                        help="Directory of PV phase 3D patches (*_pv_patch.nii.gz)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file (.xlsx for predict, .xlsx/.csv for extract)")

    parser.add_argument("--mode", type=str, default="predict",
                        choices=["predict", "extract"],
                        help="predict: MVI/Grade probabilities; extract: fused features")

    parser.add_argument("--excel", type=str, default=None,
                        help="Path to clinical Excel (filters by ID, merges labels for AUC)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, auto-detect if not specified)")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Mode: {args.mode}")

    patient_ids = get_patient_list(args.ap_dir, args.pv_dir, args.excel)
    print(f"[*] Patients with both AP and PV patches: {len(patient_ids)}")

    if len(patient_ids) == 0:
        print("[!] No patients found. Check --ap_dir, --pv_dir, and --excel.")
        sys.exit(1)

    model = load_model(args.checkpoint, device)

    dataset = Patch3DDataset(patient_ids, args.ap_dir, args.pv_dir)
    loader = torch_data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=my_collate, num_workers=args.num_workers, pin_memory=True)

    if args.mode == "predict":
        df = run_predict(model, loader, device)
        if args.excel and os.path.exists(args.excel):
            clinical = pd.read_excel(args.excel)
            label_cols = [c for c in ["MVI", "PathologicalGrade", "Recurrence", "RFS"]
                          if c in clinical.columns]
            if label_cols:
                df = df.merge(clinical[["ID"] + label_cols], on="ID", how="left")
                for col in label_cols:
                    if col == "MVI":
                        df.rename(columns={"MVI": "MVI_label"}, inplace=True)
                    elif col == "PathologicalGrade":
                        df.rename(columns={"PathologicalGrade": "Grade_label"}, inplace=True)

        if "MVI_label" in df.columns and len(df) >= 2:
            print(f"  MVI AUC: {roc_auc_score(df['MVI_label'], df['MVI_prob']):.4f}")
        if "Grade_label" in df.columns and len(df) >= 2:
            print(f"  Grade AUC: {roc_auc_score(df['Grade_label'], df['Grade_prob']):.4f}")
    else:
        df = run_extract(model, loader, device)

    df.to_excel(args.output, index=False, engine="openpyxl") \
        if args.output.endswith(".xlsx") \
        else df.to_csv(args.output, index=False)

    print(f"[*] Output saved: {args.output} ({df.shape[0]} patients, {df.shape[1]} columns)")
    print("Done.")


if __name__ == "__main__":
    main()
