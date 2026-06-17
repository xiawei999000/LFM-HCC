"""
Inference for DINOv2-based fine-tuned models on HCC CT volumes.
Supports two modes:
  - predict: output per-patient MVI and Grade probabilities
  - extract: output 768-dim patient-level features (pre-classification head)
Supports two backbones:
  - custom:  custom pretrained ViT-S/8, 128x128, 1 channel
  - official: official DINOv2 ViT-S/14, 224x224, 3 channels
"""

import argparse
import math
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


class MILModelCustom(nn.Module):
    """Custom pretrained DINOv2 backbone: ViT-S/8, 128x128, 1 channel."""

    def __init__(self, embed_dim=384):
        super().__init__()
        from timm.models.vision_transformer import vit_small_patch8_224
        self.backbone = vit_small_patch8_224(
            img_size=128, in_chans=1, num_classes=0,
            pretrained=False, dynamic_img_size=True, init_values=1e-5)

        D = embed_dim
        fusion_dim = D * 2
        self.cross_phase_attn = CrossPhaseAttention(dim=D, num_heads=4, dropout=0.1)
        self.attention = GatedAttention(fusion_dim)
        self.head_mvi = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(fusion_dim // 2, 1))
        self.head_grade = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(fusion_dim // 2, 1))

    def _encode_phase(self, x):
        return self.backbone(x)

    def forward_one(self, ap_tensor, pv_tensor):
        feat_ap = self._encode_phase(ap_tensor)
        feat_pv = self._encode_phase(pv_tensor)
        feat_ap = feat_ap.unsqueeze(0)
        feat_pv = feat_pv.unsqueeze(0)
        feat_ap, feat_pv = self.cross_phase_attn(feat_ap, feat_pv)
        feat_ap = feat_ap.squeeze(0)
        feat_pv = feat_pv.squeeze(0)
        feat_fused = torch.cat([feat_ap, feat_pv], dim=1)
        z, _ = self.attention(feat_fused)
        return z, feat_fused

    def predict(self, ap_tensor, pv_tensor):
        z, _ = self.forward_one(ap_tensor, pv_tensor)
        out_mvi = self.head_mvi(z).view(-1)
        out_grade = self.head_grade(z).view(-1)
        return torch.cat([out_mvi, out_grade], dim=0)

    def extract_feature(self, ap_tensor, pv_tensor):
        z, _ = self.forward_one(ap_tensor, pv_tensor)
        return z.squeeze(0)


class MILModelOfficial(nn.Module):
    """Official DINOv2 backbone: ViT-S/14, 224x224, 3 channels."""

    def __init__(self, embed_dim=384):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=False,
            img_size=224, in_chans=3, num_classes=0, dynamic_img_size=True)

        D = embed_dim
        fusion_dim = D * 2
        self.cross_phase_attn = CrossPhaseAttention(dim=D, num_heads=4, dropout=0.1)
        self.attention = GatedAttention(fusion_dim)
        self.head_mvi = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(fusion_dim // 2, 1))
        self.head_grade = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(fusion_dim // 2, 1))

    def _encode_phase(self, x):
        return self.backbone(x)

    def forward_one(self, ap_tensor, pv_tensor):
        feat_ap = self._encode_phase(ap_tensor)
        feat_pv = self._encode_phase(pv_tensor)
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
        return z, feat_fused

    def predict(self, ap_tensor, pv_tensor):
        z, _ = self.forward_one(ap_tensor, pv_tensor)
        out_mvi = self.head_mvi(z).view(-1)
        out_grade = self.head_grade(z).view(-1)
        return torch.cat([out_mvi, out_grade], dim=0)

    def extract_feature(self, ap_tensor, pv_tensor):
        z, _ = self.forward_one(ap_tensor, pv_tensor)
        return z.squeeze(0)


def load_model(checkpoint_path, backbone="custom", device="cpu"):
    if backbone == "custom":
        model = MILModelCustom(embed_dim=384)
    else:
        model = MILModelOfficial(embed_dim=384)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"[*] Loaded {backbone} model from: {checkpoint_path}")
    return model


# ============================================================
# Data loading
# ============================================================
class VolumeDataset(torch_data.Dataset):
    def __init__(self, patient_ids, ap_dir, pv_dir, backbone="custom"):
        self.patient_ids = list(patient_ids)
        self.ap_dir = ap_dir
        self.pv_dir = pv_dir
        self.backbone = backbone
        self.data_cache = []
        for pid in tqdm(self.patient_ids, desc="Loading volumes"):
            self.data_cache.append(self._load_patient(pid))

    def _load_patient(self, pid):
        ap_path = os.path.join(self.ap_dir, f"{pid}.nii.gz")
        pv_path = os.path.join(self.pv_dir, f"{pid}.nii.gz")
        ap = self._load_nii(ap_path)
        pv = self._load_nii(pv_path)
        return ap, pv, pid

    def _load_nii(self, path):
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(1)  # (Z, 1, H, W)

        if self.backbone == "official":
            tensor = F.interpolate(
                tensor, size=(224, 224), mode="bilinear", align_corners=False)
            tensor = tensor.repeat(1, 3, 1, 1)  # 1 -> 3 channels

        return tensor

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        ap, pv, pid = self.data_cache[idx]
        return ap, pv, pid


def my_collate(batch):
    ap_list, pv_list, pids = [], [], []
    for ap, pv, pid in batch:
        ap_list.append(ap)
        pv_list.append(pv)
        pids.append(pid)
    return ap_list, pv_list, pids


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


def get_patient_list(ap_dir, pv_dir, excel_path=None):
    ap_ids = set(f.replace(".nii.gz", "") for f in os.listdir(ap_dir)
                 if f.endswith(".nii.gz"))
    pv_ids = set(f.replace(".nii.gz", "") for f in os.listdir(pv_dir)
                 if f.endswith(".nii.gz"))
    common = sorted(ap_ids & pv_ids)

    if excel_path and os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        excel_ids = set(df["ID"].astype(str))
        common = sorted(excel_ids & set(common))

    return common


# ============================================================
# Entry point
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv2 fine-tuned model inference on HCC CT volumes")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned model checkpoint (.pth)")
    parser.add_argument("--ap_dir", type=str, required=True,
                        help="Directory of AP phase .nii.gz volumes")
    parser.add_argument("--pv_dir", type=str, required=True,
                        help="Directory of PV phase .nii.gz volumes")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path (.xlsx for predict, .xlsx/.csv for extract)")

    # Mode
    parser.add_argument("--mode", type=str, default="predict",
                        choices=["predict", "extract"],
                        help="predict: MVI/Grade probabilities; extract: 768-dim features")

    # Backbone
    parser.add_argument("--backbone", type=str, default="custom",
                        choices=["custom", "official"],
                        help="custom: ViT-S/8 128x128 1ch; official: ViT-S/14 224x224 3ch")

    # Optional
    parser.add_argument("--excel", type=str, default=None,
                        help="Path to clinical Excel (filters patients by ID column if provided)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (usually 1 for variable-length volumes)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, auto-detect if not specified)")

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Mode: {args.mode}, Backbone: {args.backbone}")

    # Find patients
    patient_ids = get_patient_list(args.ap_dir, args.pv_dir, args.excel)
    print(f"[*] Patients with both AP and PV: {len(patient_ids)}")

    if len(patient_ids) == 0:
        print("[!] No patients found. Check --ap_dir, --pv_dir, and --excel.")
        sys.exit(1)

    # Load model
    model = load_model(args.checkpoint, args.backbone, device)

    # Build dataset and loader
    dataset = VolumeDataset(patient_ids, args.ap_dir, args.pv_dir, args.backbone)
    loader = torch_data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=my_collate, num_workers=args.num_workers, pin_memory=True)

    # Run inference
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
            mvi_auc = roc_auc_score(df["MVI_label"], df["MVI_prob"])
            print(f"  MVI AUC: {mvi_auc:.4f}")
        if "Grade_label" in df.columns and len(df) >= 2:
            grade_auc = roc_auc_score(df["Grade_label"], df["Grade_prob"])
            print(f"  Grade AUC: {grade_auc:.4f}")
    else:
        df = run_extract(model, loader, device)

    df.to_excel(args.output, index=False, engine="openpyxl") \
        if args.output.endswith(".xlsx") \
        else df.to_csv(args.output, index=False)

    print(f"[*] Output saved: {args.output} ({df.shape[0]} patients, {df.shape[1]} columns)")
    print("Done.")


if __name__ == "__main__":
    main()
