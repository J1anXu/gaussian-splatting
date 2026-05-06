"""Convert an image to several sketch / binary line-art variants.

Usage:
    python to_sketch.py <image_path> [--out_dir DIR]

Outputs (next to source by default):
    {stem}_canny.png         pure Canny edges (clean technical look)
    {stem}_xdog.png          XDoG (extended Difference of Gaussians, sketch-like)
    {stem}_threshold.png     adaptive threshold (engraving-ish)
    {stem}_pencil.png        OpenCV stylized pencil sketch (grayscale, soft)
"""
import sys
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np


def canny_edges(img_bgr, low=80, high=180, blur=3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (blur * 2 + 1, blur * 2 + 1), 0)
    edges = cv2.Canny(gray, low, high)
    return cv2.bitwise_not(edges)  # white bg, black lines


def xdog(img_bgr, sigma=1.0, k=1.6, p=20.0, eps=-0.1, phi=10.0):
    """Extended DoG: classic sketch effect, gives clean dark lines on white."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma * k)
    dog = (1.0 + p) * g1 - p * g2
    out = np.where(dog >= eps, 1.0, 1.0 + np.tanh(phi * (dog - eps)))
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def adaptive_threshold(img_bgr, block=15, C=10):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block * 2 + 1, C,
    )


def pencil_sketch(img_bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.05):
    gray, _ = cv2.pencilSketch(img_bgr, sigma_s=sigma_s, sigma_r=sigma_r,
                               shade_factor=shade_factor)
    return gray


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    src = Path(args.image)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        print(f"failed to read {src}")
        sys.exit(1)

    variants = {
        "canny": canny_edges(img),
        "xdog": xdog(img),
        "threshold": adaptive_threshold(img),
        "pencil": pencil_sketch(img),
    }
    for name, out in variants.items():
        path = out_dir / f"{stem}_{name}.png"
        cv2.imwrite(str(path), out)
        print(f"  wrote {path}")
