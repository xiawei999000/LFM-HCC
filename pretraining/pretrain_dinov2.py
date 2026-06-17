#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINOv2-style self-supervised pretraining for CT lesion patches.

This script supports three LFM pretraining strategies used in the study:
  1) LFM-DL:  pretrain on generic lesion CT patches, such as DeepLesion.
  2) LFM-Mix: pretrain on pooled generic lesion and HCC-specific CT patches.
  3) LFM-Seq: sequential pretraining by loading a previous checkpoint with
              --resume and continuing/adapting on HCC-specific CT patches.

Important behavior of --resume:
  - By default, --resume loads model weights only and resets the optimizer and
    epoch counter. This is the recommended mode for sequential pretraining
    (for example, DeepLesion checkpoint -> HCC-specific adaptation).
  - Use --resume_mode full to truly continue an interrupted run, including the
    optimizer state, best loss, and epoch counter.

Example sequential pretraining:
  python pretraining/pretrain_dinov2.py \
      --data_dir /path/to/HCC_patches \
      --output_dir ./checkpoints/LFM_Seq \
      --resume ./checkpoints/LFM_DL/best.pth \
      --epochs 1
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import lightly.data as lightly_data
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOv2ProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import random_block_mask, update_drop_path_rate, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.debug import std_of_l2_normalized
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule
from PIL import Image
from timm.models.vision_transformer import vit_small_patch8_224
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CTLesionDataset(Dataset):
    """
    CT lesion patch dataset for DINOv2-style multi-view augmentation.

    The underlying LightlyDataset reads image files from a directory. Each image
    is converted to single-channel grayscale before CT-compatible DINO transforms
    are applied. The returned sample is a list of augmented views.
    """

    def __init__(self, input_dir: str, dino_transform: DINOTransform) -> None:
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"data_dir does not exist: {input_dir}")
        self.base_dataset = lightly_data.LightlyDataset(input_dir, transform=None)
        self.dino_transform = dino_transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, target, fname = self.base_dataset[index]
        image = image.convert("L")
        views = self.dino_transform(image)

        processed_views: List[Tensor] = []
        for view in views:
            if isinstance(view, Image.Image):
                view = self.to_tensor(view)
            if view.ndim != 3:
                raise ValueError(f"Unexpected view shape for {fname}: {tuple(view.shape)}")
            if view.shape[0] == 3:
                view = view[0:1, :, :]
            processed_views.append(view.contiguous())

        return processed_views, target, fname


def freeze_eval_module(module: Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class DINOv2Head(Module):
    def __init__(self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2CT(Module):
    """ViT-S/8 DINOv2-style student-teacher model for single-channel CT patches."""

    def __init__(self, ibot_separate_head: bool = False) -> None:
        super().__init__()

        vit_teacher = vit_small_patch8_224(
            img_size=128,
            in_chans=1,
            pretrained=False,
            pos_embed="learn",
            dynamic_img_size=True,
            init_values=1e-5,
        )

        self.teacher_backbone = MaskedVisionTransformerTIMM(
            vit=vit_teacher,
            antialias=False,
            pos_embed_initialization="skip",
        )
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate(self.student_backbone.vit, drop_path_rate=0.1, mode="uniform")
        freeze_eval_module(self.teacher_backbone)

        dino_head_factory = partial(DINOv2ProjectionHead, input_dim=384)
        teacher_dino_head = dino_head_factory()
        student_dino_head = dino_head_factory()
        ibot_head_factory = partial(DINOv2ProjectionHead, input_dim=384)

        if ibot_separate_head:
            teacher_ibot_head = ibot_head_factory()
            student_ibot_head = ibot_head_factory()
        else:
            teacher_ibot_head = teacher_dino_head
            student_ibot_head = student_dino_head

        self.teacher_head = DINOv2Head(dino_head=teacher_dino_head, ibot_head=teacher_ibot_head)
        self.student_head = DINOv2Head(dino_head=student_dino_head, ibot_head=student_ibot_head)
        freeze_eval_module(self.teacher_head)

    def forward(self, x: Tensor) -> Tensor:
        return self.teacher_backbone(x)

    def forward_teacher(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(self, x: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features


# -----------------------------------------------------------------------------
# Checkpoint utilities
# -----------------------------------------------------------------------------
def resolve_state_dict(checkpoint: object) -> Dict[str, Tensor]:
    """Return model state_dict from a full checkpoint or a plain state_dict."""
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        # Plain state dict: all values are tensors or tensor-like.
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint  # type: ignore[return-value]
    raise ValueError(
        "Could not find a model state dict in the checkpoint. Expected a dict with "
        "'model_state_dict', 'state_dict', or a plain PyTorch state_dict."
    )


def load_checkpoint(
    model: Module,
    optimizer: AdamW,
    resume_path: Optional[str],
    resume_mode: str,
    device: torch.device,
) -> Tuple[int, float, int]:
    """
    Load checkpoint if provided.

    Returns:
        start_epoch: epoch index to start from.
        best_loss: best training loss restored if available.
        collapse_counter: restored collapse counter, otherwise 0.
    """
    if not resume_path:
        return 0, float("inf"), 0

    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"--resume checkpoint does not exist: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    state_dict = resolve_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[*] Loaded model weights from: {resume_path}")
    if missing:
        print(f"    Missing keys: {len(missing)}")
    if unexpected:
        print(f"    Unexpected keys: {len(unexpected)}")

    if resume_mode == "model_only":
        print("[*] Resume mode: model_only. Optimizer and epoch counter are reset.")
        return 0, float("inf"), 0

    if not isinstance(checkpoint, dict):
        print("[!] Full resume requested, but checkpoint is not a full checkpoint. Starting at epoch 0.")
        return 0, float("inf"), 0

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("[*] Optimizer state restored.")
    else:
        print("[!] Optimizer state not found. Optimizer is reset.")

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", float("inf"))))
    collapse_counter = int(checkpoint.get("consecutive_collapse_epochs", 0))

    print(f"[*] Resume mode: full. Continuing from epoch {start_epoch}.")
    return start_epoch, best_loss, collapse_counter


def save_checkpoint(
    path: str,
    model: Module,
    optimizer: AdamW,
    epoch: int,
    loss: float,
    std: float,
    best_loss: float,
    expected_std: float,
    feature_dim: int,
    consecutive_collapse_epochs: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": float(loss),
            "std": float(std),
            "best_loss": float(best_loss),
            "expected_std": float(expected_std),
            "feature_dim": int(feature_dim),
            "consecutive_collapse_epochs": int(consecutive_collapse_epochs),
            "args": vars(args),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
        path,
    )


def save_teacher_backbone(path: str, model: DINOv2CT, epoch: int, loss: float, std: float, args: argparse.Namespace) -> None:
    """Save a lightweight checkpoint containing only the EMA teacher backbone."""
    torch.save(
        {
            "epoch": epoch,
            "teacher_backbone_state_dict": model.teacher_backbone.state_dict(),
            "loss": float(loss),
            "std": float(std),
            "args": vars(args),
        },
        path,
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def build_transform(args: argparse.Namespace) -> DINOTransform:
    return DINOTransform(
        global_crop_size=args.global_crop_size,
        local_crop_size=args.local_crop_size,
        global_crop_scale=(args.global_crop_scale_min, args.global_crop_scale_max),
        local_crop_scale=(args.local_crop_scale_min, args.local_crop_scale_max),
        n_local_views=args.n_local_views,
        hf_prob=0.5,
        vf_prob=0.5,
        gaussian_blur=(0.5, 0.5, 0.3),
        sigmas=(0.1, 2.0),
        normalize=None,
        cj_prob=0.0,
        cj_strength=0.0,
        random_gray_scale=0.0,
        solarization_prob=0.0,
    )


def make_mask(model: DINOv2CT, global_views: Tensor) -> Tuple[Tensor, Tensor]:
    batch_size = len(global_views)
    sequence_length = model.teacher_backbone.sequence_length
    mask = global_views.new_zeros((batch_size, sequence_length), dtype=torch.bool)
    height, width = model.teacher_backbone.vit.patch_embed.grid_size
    if height * width != sequence_length - 1:
        raise RuntimeError(
            f"Unexpected ViT grid: {height}x{width}, sequence_length={sequence_length}"
        )
    block_mask = random_block_mask(size=(batch_size, height, width), device=mask.device)
    mask[:, 1:] = block_mask.flatten(start_dim=1)
    return mask, block_mask


def train_one_epoch(
    model: DINOv2CT,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
    start_epoch: int,
    total_epochs: int,
    total_steps: int,
    training_start_time: float,
    dino_criterion: DINOLoss,
    ibot_criterion: IBOTPatchLoss,
    koleo_criterion: KoLeoLoss,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    model.train()
    # Teacher networks must stay frozen/eval.
    model.teacher_backbone.eval()
    model.teacher_head.eval()

    total_loss = 0.0
    epoch_stds: List[float] = []
    num_batches = len(dataloader)
    epoch_start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        step_start = time.time()
        views = [view.to(device, non_blocking=True) for view in batch[0]]

        global_views = torch.cat(views[:2], dim=0)
        local_views = torch.cat(views[2:], dim=0)

        mask, block_mask = make_mask(model, global_views)

        with torch.no_grad():
            teacher_cls_token, teacher_features = model.forward_teacher(global_views)
            teacher_cls_out = model.teacher_head.dino_head.forward(teacher_cls_token)
            teacher_masked_out = model.teacher_head.ibot_head.forward(teacher_features[mask])

        student_global_cls_token, student_global_masked_features = model.forward_student(global_views, mask=mask)
        student_global_cls_out = model.student_head.dino_head.forward(student_global_cls_token)
        student_global_masked_out = model.student_head.ibot_head.forward(student_global_masked_features)

        student_local_cls_token, _ = model.forward_student(local_views, mask=None)
        student_local_cls_out = model.student_head.dino_head.forward(student_local_cls_token)
        student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out], dim=0)

        # For full resume, the global step continues from start_epoch. For model-only
        # resume, start_epoch is 0, which resets the schedule for the new dataset.
        global_step = epoch * num_batches + batch_idx
        warmup_steps = max(1, int(args.teacher_temp_warmup_epochs / max(total_epochs, 1) * total_steps))
        teacher_temp = linear_warmup_schedule(
            step=global_step,
            warmup_steps=warmup_steps,
            start_value=args.teacher_temp_start,
            end_value=args.teacher_temp_end,
        )

        dino_loss = dino_criterion(
            teacher_out=teacher_cls_out.chunk(2),
            student_out=student_cls_out.chunk(len(views)),
            teacher_temp=teacher_temp,
        )
        ibot_loss = ibot_criterion(
            teacher_out=teacher_masked_out,
            student_out=student_global_masked_out,
            mask=block_mask,
            teacher_temp=teacher_temp,
        )
        koleo_loss = args.koleo_weight * sum(koleo_criterion(t) for t in student_global_cls_token.chunk(2))
        loss = dino_loss + ibot_loss + koleo_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())

        weight_decay = cosine_schedule(
            step=global_step,
            max_steps=total_steps,
            start_value=args.weight_decay_start,
            end_value=args.weight_decay_end,
        )
        for group in optimizer.param_groups:
            if group.get("weight_decay", 0.0) != 0.0:
                group["weight_decay"] = weight_decay

        momentum = cosine_schedule(
            step=global_step,
            max_steps=total_steps,
            start_value=args.momentum_teacher_start,
            end_value=1.0,
        )
        update_momentum(model.student_backbone, model.teacher_backbone, m=momentum)
        update_momentum(model.student_head, model.teacher_head, m=momentum)

        if batch_idx % args.std_interval == 0:
            with torch.no_grad():
                sample_size = min(args.std_sample_size, global_views.size(0))
                cls_features, _ = model.forward_teacher(global_views[:sample_size])
                if cls_features.dim() == 1:
                    cls_features = cls_features.unsqueeze(0)
                current_std = std_of_l2_normalized(cls_features).item()
                epoch_stds.append(float(current_std))

        current_step = (epoch - start_epoch) * num_batches + batch_idx + 1
        planned_steps = max(1, (total_epochs - start_epoch) * num_batches)
        elapsed = time.time() - training_start_time
        steps_per_sec = current_step / elapsed if elapsed > 0 else 0.0
        remaining_steps = planned_steps - current_step
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0
        step_time = max(time.time() - step_start, 1e-6)

        print(
            f"\rEpoch [{epoch + 1:03d}/{total_epochs:03d}] "
            f"Step [{batch_idx + 1:04d}/{num_batches:04d}] | "
            f"Loss {loss.item():.4f} "
            f"(DINO {dino_loss.item():.4f}, iBOT {ibot_loss.item():.4f}, KoLeo {koleo_loss.item():.4f}) | "
            f"{1.0 / step_time:.2f} batch/s | ETA {timedelta(seconds=int(eta_seconds))}",
            end="",
            flush=True,
        )

    print()
    avg_loss = total_loss / max(1, num_batches)
    avg_std = float(np.mean(epoch_stds)) if epoch_stds else 0.0
    epoch_time = time.time() - epoch_start_time
    return avg_loss, avg_std, epoch_time


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINOv2-style self-supervised pretraining on CT lesion patches"
    )

    # I/O
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CT lesion patch images")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory for checkpoints and logs")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run folder name under output_dir")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path for sequential pretraining or interrupted-run resume")
    parser.add_argument(
        "--resume_mode",
        type=str,
        default="model_only",
        choices=["model_only", "full"],
        help=(
            "model_only: load weights and reset optimizer/epoch, recommended for LFM-Seq; "
            "full: restore optimizer and epoch, for interrupted runs"
        ),
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay_start", type=float, default=0.04)
    parser.add_argument("--weight_decay_end", type=float, default=0.4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_collapse_epochs", type=int, default=5)

    # DINO/iBOT settings
    parser.add_argument("--global_crop_size", type=int, default=128)
    parser.add_argument("--local_crop_size", type=int, default=96)
    parser.add_argument("--global_crop_scale_min", type=float, default=0.5)
    parser.add_argument("--global_crop_scale_max", type=float, default=1.0)
    parser.add_argument("--local_crop_scale_min", type=float, default=0.3)
    parser.add_argument("--local_crop_scale_max", type=float, default=0.7)
    parser.add_argument("--n_local_views", type=int, default=8)
    parser.add_argument("--teacher_temp_start", type=float, default=0.04)
    parser.add_argument("--teacher_temp_end", type=float, default=0.07)
    parser.add_argument("--teacher_temp_warmup_epochs", type=int, default=30)
    parser.add_argument("--momentum_teacher_start", type=float, default=0.992)
    parser.add_argument("--koleo_weight", type=float, default=0.1)

    # Collapse monitoring
    parser.add_argument("--std_interval", type=int, default=10)
    parser.add_argument("--std_sample_size", type=int, default=16)
    parser.add_argument("--collapse_threshold_ratio", type=float, default=0.3)

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    transform = build_transform(args)
    dataset = CTLesionDataset(input_dir=args.data_dir, dino_transform=transform)
    if len(dataset) == 0:
        raise RuntimeError(f"No images were found in data_dir: {args.data_dir}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(dataset, **loader_kwargs)
    if len(dataloader) == 0:
        raise RuntimeError(
            "The dataloader has zero batches. Reduce --batch_size or add more training images."
        )

    print(f"[*] Dataset size: {len(dataset)} images")
    print(f"[*] Batches per epoch: {len(dataloader)}")

    model = DINOv2CT().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay_start)

    start_epoch, best_loss, consecutive_collapse_epochs = load_checkpoint(
        model=model,
        optimizer=optimizer,
        resume_path=args.resume,
        resume_mode=args.resume_mode,
        device=device,
    )

    if start_epoch >= args.epochs:
        raise RuntimeError(
            f"start_epoch ({start_epoch}) >= --epochs ({args.epochs}). "
            "For sequential pretraining, use the default --resume_mode model_only, or set --epochs "
            "larger than the checkpoint epoch when using --resume_mode full."
        )

    feature_dim = 384
    expected_std = 1.0 / (feature_dim ** 0.5)
    collapse_threshold = expected_std * args.collapse_threshold_ratio
    print(
        f"[*] Feature dim: {feature_dim}, expected std: {expected_std:.4f}, "
        f"collapse threshold: {collapse_threshold:.4f}"
    )

    dino_criterion = DINOLoss().to(device)
    ibot_criterion = IBOTPatchLoss().to(device)
    koleo_criterion = KoLeoLoss().to(device)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[*] Checkpoints will be saved to: {save_dir}")

    log_path = os.path.join(save_dir, "training_log.csv")
    is_new_log = not os.path.exists(log_path) or args.resume_mode == "model_only"
    log_file = open(log_path, "w" if is_new_log else "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if is_new_log:
        log_writer.writerow([
            "epoch", "loss", "std", "expected_std", "collapse_threshold",
            "best_loss", "epoch_time_sec", "total_time_sec", "resume", "resume_mode"
        ])
        log_file.flush()

    total_steps = args.epochs * len(dataloader)
    training_start_time = time.time()
    print(f"[*] Training from epoch {start_epoch + 1} to {args.epochs}")

    try:
        for epoch in range(start_epoch, args.epochs):
            avg_loss, avg_std, epoch_time = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                start_epoch=start_epoch,
                total_epochs=args.epochs,
                total_steps=total_steps,
                training_start_time=training_start_time,
                dino_criterion=dino_criterion,
                ibot_criterion=ibot_criterion,
                koleo_criterion=koleo_criterion,
                args=args,
            )

            total_time = time.time() - training_start_time
            collapsed = avg_std < collapse_threshold
            if collapsed:
                consecutive_collapse_epochs += 1
                collapse_msg = " [WARNING: representation collapse detected]"
            else:
                consecutive_collapse_epochs = 0
                collapse_msg = ""

            print(
                f"Epoch {epoch:03d} | loss={avg_loss:.4f} | std={avg_std:.4f} "
                f"(expected={expected_std:.4f}){collapse_msg} | "
                f"epoch_time={timedelta(seconds=int(epoch_time))} | "
                f"total={timedelta(seconds=int(total_time))}"
            )

            if consecutive_collapse_epochs >= args.max_collapse_epochs:
                print(
                    f"[!] Early stopping: representation collapse for "
                    f"{args.max_collapse_epochs} consecutive epochs."
                )
                break

            improved = avg_loss < best_loss
            no_collapse = avg_std > collapse_threshold

            # Always maintain a latest checkpoint.
            latest_path = os.path.join(save_dir, "latest.pth")
            save_checkpoint(
                latest_path, model, optimizer, epoch, avg_loss, avg_std, best_loss,
                expected_std, feature_dim, consecutive_collapse_epochs, args
            )

            if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
                periodic_path = os.path.join(
                    save_dir,
                    f"dinov2_epoch{epoch:03d}_loss{avg_loss:.4f}_std{avg_std:.4f}.pth",
                )
                save_checkpoint(
                    periodic_path, model, optimizer, epoch, avg_loss, avg_std, best_loss,
                    expected_std, feature_dim, consecutive_collapse_epochs, args
                )
                print(f"    Saved periodic checkpoint: {os.path.basename(periodic_path)}")

            if improved and no_collapse:
                best_loss = avg_loss
                best_path = os.path.join(save_dir, "best.pth")
                save_checkpoint(
                    best_path, model, optimizer, epoch, avg_loss, avg_std, best_loss,
                    expected_std, feature_dim, consecutive_collapse_epochs, args
                )
                backbone_path = os.path.join(save_dir, "best_teacher_backbone.pth")
                save_teacher_backbone(backbone_path, model, epoch, avg_loss, avg_std, args)
                print("    Saved best checkpoint: best.pth")
            elif not no_collapse:
                print(
                    f"    Skip best saving: std {avg_std:.4f} < collapse threshold {collapse_threshold:.4f}"
                )
            else:
                print(f"    Skip best saving: loss not improved ({avg_loss:.4f} >= {best_loss:.4f})")

            log_writer.writerow([
                epoch, f"{avg_loss:.6f}", f"{avg_std:.6f}", f"{expected_std:.6f}",
                f"{collapse_threshold:.6f}", f"{best_loss:.6f}",
                f"{epoch_time:.1f}", f"{total_time:.1f}", args.resume or "", args.resume_mode,
            ])
            log_file.flush()

    finally:
        log_file.close()

    print(f"[*] Training complete. Outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
