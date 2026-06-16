"""
Overlay occlusion heatmap onto CT image.
Usage:
    python overlay_heatmap.py <ap_image> <occ_image> <output>
"""
import sys
import numpy as np
from PIL import Image

TARGET_SIZE = 128
ALPHA = 0.5   # heatmap opacity (0~1)


def overlay(ap_path, occ_path, out_path, target_size=TARGET_SIZE, alpha=ALPHA):
    ap = Image.open(ap_path).convert('RGB').resize((target_size, target_size), Image.LANCZOS)
    occ = Image.open(occ_path).convert('RGB').resize((target_size, target_size), Image.LANCZOS)

    ap_arr = np.array(ap).astype(np.float64)
    occ_arr = np.array(occ).astype(np.float64)

    # Direct alpha blend: preserve original heatmap colors
    blended = (ap_arr * (1 - alpha) + occ_arr * alpha).astype(np.uint8)
    Image.fromarray(blended).save(out_path)
    print(f'Saved: {out_path} ({target_size}x{target_size}, alpha={alpha})')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python overlay_heatmap.py <ap_image> <occ_image> <output>')
        sys.exit(1)
    overlay(sys.argv[1], sys.argv[2], sys.argv[3])
