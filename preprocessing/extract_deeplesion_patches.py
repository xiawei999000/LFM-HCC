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


def apply_window(hu_array, level, width):
    """
    Apply a fixed CT window/level to an HU array.
    Formula: I = clip((HU - (level - width/2)) / width * 255, 0, 255)
    """
    lower = level - width / 2.0
    windowed = (hu_array - lower) / float(width) * 255.0
    windowed = np.clip(windowed, 0, 255)
    return windowed.astype(np.uint8)


def parse_dicom_window(dicom_window_str):
    """Parse a DICOM_windows CSV field (e.g. "-1350,200") into (level, width)."""
    window_vals = [float(x) for x in str(dicom_window_str).split(',')]
    if len(window_vals) != 2:
        raise ValueError(f"Invalid window format: {dicom_window_str}")
    lower, upper = window_vals
    return (lower + upper) / 2.0, upper - lower


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
        description="Extract soft-tissue-windowed lesion patches from DeepLesion dataset"
    )
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to DL_info.csv")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Root directory containing DeepLesion PNG images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for lesion patches")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Output patch size (square)")
    # Paper (Sec. 4.2): "All pretraining patches were windowed using a
    # soft-tissue setting (width of 400 HU and level of 40 HU)".
    parser.add_argument("--window_level", type=float, default=40.0,
                        help="Window level in HU (default: 40, soft-tissue)")
    parser.add_argument("--window_width", type=float, default=400.0,
                        help="Window width in HU (default: 400, soft-tissue)")
    parser.add_argument("--use_dicom_window", action="store_true",
                        help="Use the per-image DICOM_windows field from DL_info.csv "
                             "instead of the fixed soft-tissue window")
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
    if args.use_dicom_window:
        print("Windowing: per-image DICOM_windows from CSV")
    else:
        print(f"Windowing: fixed soft-tissue window (level={args.window_level}, "
              f"width={args.window_width})")

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

            if args.use_dicom_window:
                level, width = parse_dicom_window(matching_row['DICOM_windows'].iloc[0])
            else:
                level, width = args.window_level, args.window_width

            img = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not load {png_file}")
                continue

            if img.dtype != np.uint16:
                print(f"Warning: {image_name} is not uint16 (dtype: {img.dtype})")

            cropped_img = crop_and_expand(img, int_bbox)

            hu_img = convert_to_hu(cropped_img)
            windowed_img = apply_window(hu_img, level, width)
            resized_img = cv2.resize(windowed_img, (args.patch_size, args.patch_size),
                                     interpolation=cv2.INTER_LINEAR)

            output_name = f'{os.path.splitext(image_name)[0]}_patch.png'
            output_path = os.path.join(args.output_dir, output_name)
            cv2.imwrite(output_path, resized_img)

            print(f'Saved: {output_path} | Window: level={level}, width={width}')

        except Exception as e:
            print(f'Error processing {image_name}: {e}')
            continue

    print("Done: All images processed.")


if __name__ == "__main__":
    main()
