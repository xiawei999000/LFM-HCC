import argparse
import copy
import os
import time
from datetime import timedelta, datetime
from functools import partial


import torch
from PIL import Image
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import Dataset
from torchvision import transforms
from timm.models.vision_transformer import vit_small_patch8_224

import lightly.data as data
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOv2ProjectionHead, MaskedVisionTransformerTIMM
from lightly.models.utils import random_block_mask, update_drop_path_rate, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule
from lightly.utils.debug import std_of_l2_normalized


class HCCLightlyDataset(Dataset):
    """
    HCC CT Dataset: 3-channel PNG -> single channel (first channel) -> per-image Z-score normalization.
    Each image is independently normalized to mitigate scanner/protocol variation.
    """

    def __init__(self, input_dir, dino_transform, eps=1e-6):
        self.base_dataset = data.LightlyDataset(input_dir, transform=None)
        self.dino_transform = dino_transform
        self.to_tensor = transforms.ToTensor()
        self.eps = eps

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, target, fname = self.base_dataset[index]

        # Convert to grayscale (single-channel PIL, [0, 255])
        image = image.convert('L')

        # DINO multi-view transform
        views = self.dino_transform(image)

        processed_views = []
        for view in views:
            if isinstance(view, Image.Image):
                view = self.to_tensor(view)

            # Keep only the first channel for grayscale
            if view.shape[0] == 3:
                view = view[0:1, :, :]

            processed_views.append(view)

        return processed_views, target, fname


def freeze_eval_module(module: Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2Head(Module):
    def __init__(self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2(Module):
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

        dino_head = partial(DINOv2ProjectionHead, input_dim=384)
        teacher_dino_head = dino_head()
        student_dino_head = dino_head()
        ibot_head = partial(DINOv2ProjectionHead, input_dim=384)

        if ibot_separate_head:
            teacher_ibot_head = ibot_head()
            student_ibot_head = ibot_head()
        else:
            teacher_ibot_head = teacher_dino_head
            student_ibot_head = student_dino_head

        self.teacher_head = DINOv2Head(dino_head=teacher_dino_head, ibot_head=teacher_ibot_head)
        self.student_head = DINOv2Head(dino_head=student_dino_head, ibot_head=student_ibot_head)
        freeze_eval_module(self.teacher_head)

    def forward(self, x):
        return self.teacher_backbone(x)

    def forward_teacher(self, x):
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(self, x, mask):
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features


def target_transform(t):
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv2 pretraining on combined DeepLesion + HCC CT patches"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to directory containing training image patches"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints",
        help="Directory to save model checkpoints and training log"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_collapse_epochs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    transform = DINOTransform(
        global_crop_size=128,
        local_crop_size=96,
        global_crop_scale=(0.5, 1.0),
        local_crop_scale=(0.3, 0.7),
        n_local_views=6,
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing dataset...")
    dataset = HCCLightlyDataset(
        input_dir=args.data_dir,
        dino_transform=transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
    )

    print("Initializing model...")
    model = DINOv2()
    model.to(device)

    feature_dim = 384
    expected_std = 1.0 / (feature_dim ** 0.5)
    collapse_threshold = expected_std * 0.3

    print(f"Feature dim: {feature_dim}, Expected std: {expected_std:.4f}, "
          f"Collapse threshold: {collapse_threshold:.4f}")

    dino_criterion = DINOLoss().to(device)
    ibot_criterion = IBOTPatchLoss().to(device)
    koleo_criterion = KoLeoLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.04)

    epochs = args.epochs
    num_batches = len(dataloader)
    total_steps = epochs * num_batches

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    log_file = open(os.path.join(save_dir, "training_log.txt"), "w")
    log_file.write("epoch,loss,std,expected_std,epoch_time,total_time\n")

    best_loss = float('inf')
    training_start_time = time.time()

    print(f"Total batches: {num_batches}, Total steps: {total_steps}")
    print(f"Training for {epochs} epochs, saving to: {save_dir}")

    consecutive_collapse_epochs = 0
    max_collapse_epochs = args.max_collapse_epochs

    print("Starting Training")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        epoch_stds = []

        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()

            views = batch[0]
            views = [view.to(device) for view in views]

            global_views = torch.cat(views[:2])
            local_views = torch.cat(views[2:])

            # iBOT block-wise masking
            B = len(global_views)
            sequence_length = model.teacher_backbone.sequence_length
            mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)
            H, W = model.teacher_backbone.vit.patch_embed.grid_size
            assert H * W == sequence_length - 1, \
                f"Unexpected grid size: {H}x{W}, sequence_length {sequence_length}"
            block_mask = random_block_mask(size=(B, H, W), device=mask.device)
            mask[:, 1:] = block_mask.flatten(start_dim=1)

            # Teacher forward
            with torch.no_grad():
                teacher_cls_token, teacher_features = model.forward_teacher(global_views)
                teacher_cls_out = model.teacher_head.dino_head.forward(teacher_cls_token)
                teacher_masked_out = model.teacher_head.ibot_head.forward(teacher_features[mask])

            # Student forward
            student_global_cls_token, student_global_masked_features = \
                model.forward_student(global_views, mask=mask)
            student_global_cls_out = model.student_head.dino_head.forward(student_global_cls_token)
            student_global_masked_out = model.student_head.ibot_head.forward(student_global_masked_features)
            student_local_cls_token, _ = model.forward_student(local_views, mask=None)
            student_local_cls_out = model.student_head.dino_head.forward(student_local_cls_token)
            student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

            # Loss computation
            global_step = epoch * num_batches + batch_idx
            teacher_temp = linear_warmup_schedule(
                step=global_step,
                warmup_steps=int(30 / epochs * total_steps),
                start_value=0.04,
                end_value=0.07,
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
            koleo_loss = 0.1 * sum(koleo_criterion(t) for t in student_global_cls_token.chunk(2))
            loss = dino_loss + ibot_loss + koleo_loss

            total_loss += loss.detach()
            loss.backward()

            # Learning rate warmup for the last layer
            if epoch < 1:
                for param_group in optimizer.param_groups:
                    if "last_layer" in param_group:
                        param_group["lr"] = 0.0

            # Weight decay schedule
            weight_decay = cosine_schedule(
                step=global_step, max_steps=total_steps, start_value=0.04, end_value=0.4
            )
            for group in optimizer.param_groups:
                if group["weight_decay"] != 0.0:
                    group["weight_decay"] = weight_decay

            optimizer.step()
            optimizer.zero_grad()

            # EMA update for teacher model
            momentum = cosine_schedule(
                step=global_step, max_steps=total_steps, start_value=0.992, end_value=1.0
            )
            update_momentum(model.student_backbone, model.teacher_backbone, m=momentum)
            update_momentum(model.student_head, model.teacher_head, m=momentum)

            # Progress display
            current_step_global = epoch * num_batches + batch_idx + 1
            step_time = time.time() - step_start

            elapsed = time.time() - training_start_time
            steps_per_sec = current_step_global / elapsed if elapsed > 0 else 0
            remaining_steps = total_steps - current_step_global
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

            print(
                f"\rEpoch: [{epoch + 1:03d}/{epochs:03d}] "
                f"Step: [{batch_idx + 1:04d}/{num_batches:04d}] "
                f"({current_step_global}/{total_steps}) | "
                f"Loss: {loss.item():.4f} "
                f"(DINO: {dino_loss.item():.4f}, iBOT: {ibot_loss.item():.4f}, "
                f"KoLeo: {koleo_loss.item():.4f}) | "
                f"Speed: {1 / step_time:.1f} batch/s | ETA: {timedelta(seconds=int(eta_seconds))}",
                end='', flush=True
            )

            # Monitor representation collapse via std of L2-normalized features
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    sample_size = min(16, global_views.size(0))
                    sample_views = global_views[:sample_size]
                    cls_features, _ = model.forward_teacher(sample_views)
                    if cls_features.dim() == 1:
                        cls_features = cls_features.unsqueeze(0)
                    current_std = std_of_l2_normalized(cls_features).item()
                    epoch_stds.append(current_std)

        print()

        avg_loss = total_loss / len(dataloader)
        avg_std = sum(epoch_stds) / len(epoch_stds) if epoch_stds else 0.0

        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time

        log_file.write(
            f"{epoch},{avg_loss:.4f},{avg_std:.4f},{expected_std:.4f},"
            f"{epoch_time:.1f},{total_time:.1f}\n"
        )
        log_file.flush()

        collapse_warning = ""
        if avg_std < collapse_threshold:
            collapse_warning = " [WARNING: Collapse detected!]"
            consecutive_collapse_epochs += 1
            if consecutive_collapse_epochs >= max_collapse_epochs:
                print(f"Early stopping: Model collapsed for {max_collapse_epochs} "
                      f"consecutive epochs!")
                break
        else:
            consecutive_collapse_epochs = 0

        print(
            f"epoch: {epoch:>02}, loss: {avg_loss:.4f}, std: {avg_std:.4f} "
            f"(exp: {expected_std:.4f}){collapse_warning}, "
            f"time: {timedelta(seconds=int(epoch_time))}, "
            f"total: {timedelta(seconds=int(total_time))}"
        )

        # Periodic checkpoint (every 10 epochs)
        if epoch % 10 == 0:
            latest_path = os.path.join(
                save_dir,
                f"dinov2_epoch{epoch:03d}_loss{avg_loss:.4f}_std{avg_std:.4f}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'std': avg_std,
            }, latest_path)

        # Save best model (improved loss and no collapse)
        is_better_loss = avg_loss < best_loss
        no_collapse = avg_std > collapse_threshold

        if is_better_loss and no_collapse:
            best_loss = avg_loss

            checkpoint_name = f"dinov2_epoch{epoch:03d}_loss{avg_loss:.4f}_std{avg_std:.4f}.pth"
            checkpoint_path = os.path.join(save_dir, checkpoint_name)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'std': avg_std,
                'expected_std': expected_std,
                'feature_dim': feature_dim,
                'epoch_time': epoch_time,
                'total_time': total_time,
            }, checkpoint_path)

            print(f"  -> Saved best model: {checkpoint_name}")
        elif not no_collapse:
            print(f"  -> Skip saving: Model collapsed "
                  f"(std {avg_std:.4f} < {collapse_threshold:.4f})")
        elif not is_better_loss:
            print(f"  -> Skip saving: Loss not improved "
                  f"({avg_loss:.4f} >= {best_loss:.4f})")

    log_file.close()
    print(f"Training complete. Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
