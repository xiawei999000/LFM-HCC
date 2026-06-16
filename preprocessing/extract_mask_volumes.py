"""
Extract lesion mask annotation volumes: crop + pad + resize mask slices,
saved as one .nii.gz per patient. Uses the same bounding-box logic as
extract_lesion_volumes.py to ensure spatial alignment with image volumes.
"""

import argparse
import json
import os
import traceback
import numpy as np
import SimpleITK as sitk
from PIL import Image


class MaskVolumeConfig:
    """Configuration for mask volume extraction."""

    def __init__(self, input_dir, output_dir, data_set_name="HCC", center="Unknown"):
        self.data_set_name = data_set_name
        self.center = center

        # Auto-detect mask filename: try mask-labels.nrrd first, then mask.nii.gz
        self.mask_candidates = ["mask-labels.nrrd", "mask.nii.gz"]

        self.INPUT_BASE = input_dir
        self.OUTPUT_BASE = output_dir

        self.SPACING = [1.0, 1.0]

        self.PERITUMORAL_MARGIN = 10
        # No tumor size filter for masks: extract all annotations
        self.MIN_TUMOR_MM = 0
        self.MAX_TUMOR_MM = 9999

        self.PATCH_SIZE = (128, 128)

        # Whether to binarize multi-label masks (keep individual label values otherwise)
        self.BINARIZE = False
        self.MASK_SUFFIX = "_mask"


class MaskVolumeExtractor:
    def __init__(self, config):
        self.cfg = config

    def resample_xy(self, image):
        spacing = image.GetSpacing()
        size = image.GetSize()

        if (abs(spacing[0] - self.cfg.SPACING[0]) < 0.01 and
                abs(spacing[1] - self.cfg.SPACING[1]) < 0.01):
            return image

        new_size = [
            int(np.round(size[0] * spacing[0] / self.cfg.SPACING[0])),
            int(np.round(size[1] * spacing[1] / self.cfg.SPACING[1])),
            size[2]
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing([self.cfg.SPACING[0], self.cfg.SPACING[1], spacing[2]])
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

        return resample.Execute(image)

    def get_union_square_bbox(self, mask_slice):
        """
        Compute union bounding box of all valid lesions on a slice, expanded to a square.
        Side length = max(union_height, union_width) + 2 * margin.
        """
        if np.sum(mask_slice) == 0:
            return None

        mask_sitk = sitk.GetImageFromArray(mask_slice.astype(np.uint8))
        connected = sitk.ConnectedComponent(mask_sitk)
        labels = sitk.GetArrayFromImage(connected)

        valid_components = []

        margin_px_x = int(round(self.cfg.PERITUMORAL_MARGIN / self.cfg.SPACING[0]))
        margin_px_y = int(round(self.cfg.PERITUMORAL_MARGIN / self.cfg.SPACING[1]))

        for label_id in np.unique(labels):
            if label_id == 0:
                continue

            tumor_mask = (labels == label_id).astype(np.uint8)
            y, x = np.where(tumor_mask > 0)
            if len(y) == 0:
                continue

            tumor_h_px = int(y.max() - y.min() + 1)
            tumor_w_px = int(x.max() - x.min() + 1)

            tumor_h_mm = tumor_h_px * self.cfg.SPACING[1]
            tumor_w_mm = tumor_w_px * self.cfg.SPACING[0]
            max_diameter_mm = max(tumor_h_mm, tumor_w_mm)

            if max_diameter_mm < self.cfg.MIN_TUMOR_MM:
                continue
            if max_diameter_mm > self.cfg.MAX_TUMOR_MM:
                continue

            valid_components.append({
                "label_id": int(label_id),
                "ymin": int(y.min()),
                "ymax": int(y.max()),
                "xmin": int(x.min()),
                "xmax": int(x.max()),
                "diameter_mm": float(max_diameter_mm),
                "area_px": int(np.sum(tumor_mask > 0))
            })

        if not valid_components:
            return None

        union_y1 = min(c["ymin"] for c in valid_components)
        union_y2 = max(c["ymax"] for c in valid_components)
        union_x1 = min(c["xmin"] for c in valid_components)
        union_x2 = max(c["xmax"] for c in valid_components)

        union_h_px = int(union_y2 - union_y1 + 1)
        union_w_px = int(union_x2 - union_x1 + 1)

        square_side_px = int(max(
            union_h_px + 2 * margin_px_y,
            union_w_px + 2 * margin_px_x
        ))

        cy = int((union_y1 + union_y2) // 2)
        cx = int((union_x1 + union_x2) // 2)
        half_side = square_side_px // 2

        y1 = cy - half_side
        y2 = cy + half_side
        x1 = cx - half_side
        x2 = cx + half_side

        img_h, img_w = mask_slice.shape
        pad_y1 = max(0, -y1)
        pad_y2 = max(0, y2 - (img_h - 1))
        pad_x1 = max(0, -x1)
        pad_x2 = max(0, x2 - (img_w - 1))

        y1 = max(0, y1)
        y2 = min(img_h - 1, y2)
        x1 = max(0, x1)
        x2 = min(img_w - 1, x2)

        actual_h = y2 - y1 + 1
        actual_w = x2 - x1 + 1

        return {
            "coords": (int(y1), int(y2), int(x1), int(x2)),
            "center": (int(cy), int(cx)),
            "target_side_px": int(square_side_px),
            "tumor_size_mm": float(max(c["diameter_mm"] for c in valid_components)),
            "tumor_area_px": int(np.sum(mask_slice > 0)),
            "num_lesions": int(len(valid_components)),
            "padding": (int(pad_y1), int(pad_y2), int(pad_x1), int(pad_x2)),
            "actual_size": (int(actual_h), int(actual_w)),
            "label_ids": sorted(set(c["label_id"] for c in valid_components)),
        }

    def crop_and_resize(self, mask_slice, bbox_info, preserve_labels=False):
        y1, y2, x1, x2 = bbox_info["coords"]
        target_side_px = bbox_info["target_side_px"]
        pad_y1, pad_y2, pad_x1, pad_x2 = bbox_info["padding"]

        if target_side_px < 2:
            return None, None

        cropped = mask_slice[y1:y2 + 1, x1:x2 + 1].copy()
        if cropped.size == 0:
            return None, None

        actual_h, actual_w = cropped.shape

        if (actual_h == target_side_px and actual_w == target_side_px and
                sum([pad_y1, pad_y2, pad_x1, pad_x2]) == 0):
            resized = np.array(
                Image.fromarray(cropped).resize(self.cfg.PATCH_SIZE, Image.NEAREST))
            if not preserve_labels:
                resized = (resized > 0).astype(np.uint8)
            return resized, {"scale": self.cfg.PATCH_SIZE[0] / target_side_px}

        pad_width = ((pad_y1, pad_y2), (pad_x1, pad_x2))
        padded = np.pad(cropped, pad_width, mode='constant', constant_values=0)

        if padded.shape[0] != target_side_px or padded.shape[1] != target_side_px:
            fixed = np.zeros((target_side_px, target_side_px), dtype=padded.dtype)
            h = min(target_side_px, padded.shape[0])
            w = min(target_side_px, padded.shape[1])
            fixed[:h, :w] = padded[:h, :w]
            padded = fixed

        resized = np.array(
            Image.fromarray(padded).resize(self.cfg.PATCH_SIZE, Image.NEAREST))

        if not preserve_labels:
            resized = (resized > 0).astype(np.uint8)

        return resized, {
            "original_side_px": target_side_px,
            "scale": self.cfg.PATCH_SIZE[0] / target_side_px,
            "padding": (pad_y1, pad_y2, pad_x1, pad_x2)
        }

    def find_mask_file(self, patient_path):
        for candidate in self.cfg.mask_candidates:
            full_path = os.path.join(patient_path, candidate)
            if os.path.exists(full_path):
                return full_path, os.path.basename(full_path)
        return None, None

    def process_patient(self, patient_name):
        patient_path = os.path.join(self.cfg.INPUT_BASE, patient_name)
        mask_file, mask_fname = self.find_mask_file(patient_path)

        if mask_file is None:
            return False, {
                "patient": patient_name,
                "saved": False,
                "reason": "mask_not_found"
            }

        try:
            mask = sitk.ReadImage(mask_file)
            mask = self.resample_xy(mask)
            mask_array = sitk.GetArrayFromImage(mask)

            # Detect if multi-label or binary
            unique_vals = np.unique(mask_array)
            preserve_labels = (not self.cfg.BINARIZE) and (len(unique_vals) > 2)

            if self.cfg.BINARIZE:
                mask_array = (mask_array > 0).astype(np.uint8)

            volume_slices = []
            slice_meta = []

            for z in range(mask_array.shape[0]):
                mask_slice = mask_array[z]
                if np.sum(mask_slice > 0) == 0:
                    continue

                bbox_info = self.get_union_square_bbox(mask_slice)
                if bbox_info is None:
                    continue

                patch_mask, resize_info = self.crop_and_resize(
                    mask_slice, bbox_info, preserve_labels=preserve_labels)

                if patch_mask is None:
                    continue

                tumor_ratio = float(np.sum(patch_mask > 0) / patch_mask.size)
                volume_slices.append(patch_mask.astype(np.uint8 if not preserve_labels else patch_mask.dtype))

                slice_meta.append({
                    "slice": int(z),
                    "tumor_size_mm": float(round(bbox_info["tumor_size_mm"], 2)),
                    "square_side_px": int(bbox_info["target_side_px"]),
                    "square_side_mm_x": float(round(bbox_info["target_side_px"] * self.cfg.SPACING[0], 2)),
                    "tumor_ratio": float(tumor_ratio),
                    "resize_scale": float(resize_info["scale"]),
                    "num_lesions": int(bbox_info["num_lesions"]),
                    "label_ids": bbox_info.get("label_ids", []),
                })

            if not volume_slices:
                return False, {
                    "patient": patient_name,
                    "saved": False,
                    "reason": "no_valid_tumor_slice"
                }

            volume_np = np.stack(volume_slices, axis=0)
            volume_itk = sitk.GetImageFromArray(volume_np)
            volume_itk.SetSpacing((self.cfg.SPACING[0], self.cfg.SPACING[1], mask.GetSpacing()[2]))

            out_file = os.path.join(self.cfg.OUTPUT_BASE, f"{patient_name}{self.cfg.MASK_SUFFIX}.nii.gz")
            sitk.WriteImage(volume_itk, out_file)

            return True, {
                "patient": patient_name,
                "saved": True,
                "mask_source": mask_fname,
                "output_file": os.path.basename(out_file),
                "num_slices": int(len(volume_slices)),
                "selected_slices": [int(m["slice"]) for m in slice_meta],
                "patch_size": list(self.cfg.PATCH_SIZE),
                "preserve_labels": preserve_labels,
                "unique_labels": [int(v) for v in unique_vals if v != 0],
                "slice_meta": slice_meta
            }

        except Exception as e:
            print(f"  Error {patient_name}: {str(e)}")
            traceback.print_exc()
            return False, {
                "patient": patient_name,
                "saved": False,
                "reason": f"exception: {str(e)}"
            }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract lesion mask annotation volumes as .nii.gz per patient")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to patient directories containing mask files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .nii.gz mask volumes")
    parser.add_argument("--mask_names", type=str, default="mask-labels.nrrd,mask.nii.gz",
                        help="Comma-separated mask filename candidates (tried in order)")
    parser.add_argument("--binarize", action="store_true",
                        help="Binarize multi-label masks (all labels >0 -> 1)")
    parser.add_argument("--dataset_name", type=str, default="HCC",
                        help="Dataset name for metadata output")
    parser.add_argument("--center", type=str, default="Unknown",
                        help="Center name for metadata output")
    parser.add_argument("--margin_mm", type=float, default=10,
                        help="Peritumoral margin in mm (default: 10)")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Output patch size (default: 128)")
    parser.add_argument("--spacing", type=float, nargs=2, default=[1.0, 1.0],
                        help="Target XY spacing in mm (default: 1.0 1.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = MaskVolumeConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        data_set_name=args.dataset_name,
        center=args.center,
    )
    cfg.mask_candidates = [s.strip() for s in args.mask_names.split(",")]
    cfg.BINARIZE = args.binarize
    cfg.PERITUMORAL_MARGIN = args.margin_mm
    cfg.PATCH_SIZE = (args.patch_size, args.patch_size)
    cfg.SPACING = list(args.spacing)

    os.makedirs(cfg.OUTPUT_BASE, exist_ok=True)

    extractor = MaskVolumeExtractor(cfg)

    patients = [
        d for d in os.listdir(cfg.INPUT_BASE)
        if os.path.isdir(os.path.join(cfg.INPUT_BASE, d))
    ]

    print("=" * 70)
    print("Mask Volume Extraction — one .nii.gz per patient")
    print("=" * 70)
    print(f"Dataset: {cfg.data_set_name}")
    print(f"Input: {cfg.INPUT_BASE}")
    print(f"Output: {cfg.OUTPUT_BASE}")
    print(f"Mask candidates: {cfg.mask_candidates}")
    print(f"Binarize: {cfg.BINARIZE}")
    print(f"XY spacing: {cfg.SPACING}")
    print(f"Patch size: {cfg.PATCH_SIZE}")
    print(f"Margin: {cfg.PERITUMORAL_MARGIN} mm")
    print(f"Patients: {len(patients)}")
    print("=" * 70)

    all_meta = []
    success_count = 0
    fail_count = 0

    for i, patient in enumerate(patients, 1):
        print(f"[{i:3d}/{len(patients)}] {patient}...", end=" ", flush=True)
        success, meta = extractor.process_patient(patient)
        all_meta.append(meta)

        if success:
            success_count += 1
            print(f"OK, saved {meta['num_slices']} slices -> {meta['output_file']}")
        else:
            fail_count += 1
            print(f"Skip, reason: {meta['reason']}")

    json_fname = f"{cfg.data_set_name}_mask_volume_metadata.json"
    meta_file = os.path.join(cfg.OUTPUT_BASE, json_fname)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"Done: {success_count} succeeded, {fail_count} failed/skipped")
    print(f"Metadata saved to: {meta_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
