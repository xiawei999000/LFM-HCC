import argparse
import os
import glob
import numpy as np
import pandas as pd
import cv2


def convert_to_hu(img_uint16):
    """
    Convert 16-bit PNG to Hounsfield Units.
    Formula: HU = pixel_value - 32768
    """
    return img_uint16.astype(np.int32) - 32768


def apply_dicom_window(hu_array, dicom_window_str):
    """
    Apply DICOM window/level to HU array.
    Formula: I = clip((HU - A) / (B - A) * 255, 0, 255)

    Args:
        hu_array: HU values (int32)
        dicom_window_str: CSV DICOM_windows field, e.g. "-1350,200" (lung) or "40,400" (abdomen)
    Returns:
        uint8 array in [0, 255]
    """
    try:
        window_vals = [float(x) for x in str(dicom_window_str).split(',')]
        if len(window_vals) != 2:
            raise ValueError(f"Invalid window format: {dicom_window_str}")
        A, B = window_vals[0], window_vals[1]

        if abs(B - A) < 1e-6:
            return np.zeros_like(hu_array, dtype=np.uint8)

        windowed = (hu_array - A) / (B - A) * 255.0
        windowed = np.clip(windowed, 0, 255)
        return windowed.astype(np.uint8)

    except Exception as e:
        print(f"Warning: Failed to apply window '{dicom_window_str}', using fallback. Error: {e}")
        hu_min, hu_max = np.min(hu_array), np.max(hu_array)
        if hu_max > hu_min:
            normalized = (hu_array - hu_min) / (hu_max - hu_min) * 255
            return normalized.astype(np.uint8)
        return np.zeros_like(hu_array, dtype=np.uint8)


def crop_and_expand(img, bbox_px, expand_ratio=0.2):
    """
    Crop a square region centered on the lesion bounding box with margin expansion.
    Returns the cropped region (keeps original uint16 depth).
    """
    width = bbox_px[2] - bbox_px[0]
    height = bbox_px[3] - bbox_px[1]
    side_length = max(width, height) * (1 + expand_ratio)

    center_x = int(bbox_px[0] + width / 2)
    center_y = int(bbox_px[1] + height / 2)
    half_side = int(side_length / 2)

    left = max(0, int(center_x - half_side))
    top = max(0, int(center_y - half_side))
    right = min(img.shape[1], int(center_x + half_side))
    bottom = min(img.shape[0], int(center_y + half_side))

    return img[top:bottom, left:right]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DICOM-windowed lesion patches from DeepLesion dataset"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to DL_info.csv")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Root directory containing DeepLesion PNG images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for lesion patches")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Output patch size (square)")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    os.makedirs(args.output_dir, exist_ok=True)

    png_files = list(glob.iglob(os.path.join(args.image_dir, '**', '*.png'), recursive=True))

    print(f"Found {len(png_files)} PNG files")
    print(f"CSV entries: {len(df)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")

    for png_file in png_files:
        image_rel_path = os.path.relpath(png_file, args.image_dir)
        parts = image_rel_path.split(os.sep)
        image_name = '_'.join(parts[2:])

        matching_row = df[df['File_name'] == image_name]

        if matching_row.empty:
            continue

        try:
            bounding_box_str = matching_row['Bounding_boxes'].iloc[0]
            bbox_px = np.array(bounding_box_str.split(','), dtype=float)
            int_bbox = [int(coord) for coord in bbox_px]

            dicom_window_str = matching_row['DICOM_windows'].iloc[0]

            img = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not load {png_file}")
                continue

            if img.dtype != np.uint16:
                print(f"Warning: {image_name} is not uint16 (dtype: {img.dtype})")

            cropped_img = crop_and_expand(img, int_bbox)

            hu_img = convert_to_hu(cropped_img)
            windowed_img = apply_dicom_window(hu_img, dicom_window_str)
            resized_img = cv2.resize(windowed_img, (args.patch_size, args.patch_size),
                                     interpolation=cv2.INTER_LINEAR)

            output_name = f'{os.path.splitext(image_name)[0]}_patch.png'
            output_path = os.path.join(args.output_dir, output_name)
            cv2.imwrite(output_path, resized_img)

            print(f'Saved: {output_path} | Window: {dicom_window_str}')

        except Exception as e:
            print(f'Error processing {image_name}: {e}')
            continue

    print("Done: All images processed with DICOM windowing.")


if __name__ == "__main__":
    main()
