#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import skimage as ski

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def to_gray(image):
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim == 2:
        return ski.util.img_as_float(image)
    if image.shape[-1] == 4:
        image = ski.color.rgba2rgb(image)
    return ski.color.rgb2gray(image)


def make_mooney(path, blur, use_smart, manual_threshold):
    gray = to_gray(iio.imread(path))
    if blur > 0:
        gray = ski.filters.gaussian(gray, sigma=blur, truncate=3.5)
    t = ski.filters.threshold_otsu(gray) if use_smart else manual_threshold
    mooney = ((gray > t) * 255).astype(np.uint8)
    return mooney, float(t)


def parse_args():
    p = argparse.ArgumentParser(description="Source images to Mooney images.")
    p.add_argument("-i", "--input", default="./out/original",
                   help="source folder (default: %(default)s)")
    p.add_argument("-o", "--output", default="./out/mooney",
                   help="output folder (default: %(default)s)")
    p.add_argument("-b", "--blur", type=float, default=2.0,
                   help="Gaussian sigma value in pixels (default: %(default)s)")
    p.add_argument("--smart", action=argparse.BooleanOptionalAction, default=True,
                   help="smart threshold via Otsu's method; use --no-smart to turn off "
                        "(default: on)")
    p.add_argument("-t", "--threshold", type=float, default=None,
                   help="manual threshold between 0 and 1, providing it turns smart threshold off. ")
    return p.parse_args()


def main():
    args = parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    if not in_dir.is_dir():
        sys.exit(f"Input folder not found: {in_dir}")
    if args.blur < 0:
        sys.exit("--blur must be >= 0")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        sys.exit("--threshold must be between 0 and 1")

    use_smart = args.smart and args.threshold is None

    if args.threshold is None:
        manual = 0.5
    else:
        manual = args.threshold

    files = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"No images found in {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "smart (Otsu)" if use_smart else f"manual ({manual:g})"
    print(f"{len(files)} images / blur sigma={args.blur:g} / threshold: {mode}")

    thresholds = []
    for f in files:
        mooney, t = make_mooney(f, args.blur, use_smart, manual)
        thresholds.append(t)
        iio.imwrite(out_dir / f"{f.stem}.png", mooney)
        print(f"  {f.name}: threshold={t:.3f}")

    print(f"Succesfuly processed all images. Saved to {out_dir} "
          f"(threshold min/mean/max: {min(thresholds):.3f} / "
          f"{np.mean(thresholds):.3f} / {max(thresholds):.3f})")


if __name__ == "__main__":
    main()