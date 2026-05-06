#!/usr/bin/env python3
"""Build per-view block appendix figures as SVG.

The output is a vector SVG canvas with embedded PNG panels. This means the
layout, borders, and optional text are vector, while the rendered block images
remain raster because the source renders are PNG files.

Edit `RENDER_ROOT` and `FIGURE_SPECS` below, then run:

    python3 make_block_appendix_svg.py
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


RENDER_ROOT = Path(
    "/data/jian/output/CapGS/mip360/nurips26_fix_drjohnson_B/"
    "bicycle/rendered_blocks/nurips26_fix_drjohnson_B/test/ours_30000"
)


@dataclass(frozen=True)
class FigureSpec:
    view_name: str
    image_types: Tuple[str, ...]
    block_ids: Optional[Tuple[int, ...]] = None
    output_name: Optional[str] = None


FIGURE_SPECS: Tuple[FigureSpec, ...] = (
    FigureSpec(
        view_name="_DSC8679.JPG",
        image_types=("RGB", "Depth", "residual_transmitance_raw"),
    ),
)


DEFAULT_OUTPUT_SUBDIRS = {
    "rgb": "rgb",
    "depth": "depth",
    "depth_raw": "depth_raw",
    "opacity": "opacity",
    "opacity_raw": "opacity_raw",
    "residual_transmitance": "residual_transmitance",
    "residual_transmitance_raw": "residual_transmitance_raw",
}

TYPE_ALIASES = {
    "rgb": "rgb",
    "color": "rgb",
    "depth": "depth",
    "depthraw": "depth_raw",
    "opacity": "opacity",
    "opacityraw": "opacity_raw",
    "residualtransmitance": "residual_transmitance",
    "residualtransmitanceraw": "residual_transmitance_raw",
    "residualtransmittance": "residual_transmitance",
    "residualtransmittanceraw": "residual_transmitance_raw",
}

TYPE_LABELS = {
    "rgb": "RGB",
    "depth": "Depth",
    "depth_raw": "Depth Raw",
    "opacity": "Opacity",
    "opacity_raw": "Opacity Raw",
    "residual_transmitance": "Residual Transmitance",
    "residual_transmitance_raw": "Residual Transmitance Raw",
}

BLOCK_FILE_RE = re.compile(r"block_(\d+)\.png$")

CELL_WIDTH = 320
OUTER_PAD = 24
ROW_GAP = 20
COL_GAP = 16
BORDER_COLOR = "#000000"
BORDER_WIDTH = "1pt"
BG_COLOR = "#ffffff"
MISSING_FILL = "#f7f8fa"
TEXT_COLOR = "#202124"
SHOW_TYPE_LABELS = False
SHOW_BLOCK_LABELS = False
STRICT_MODALITY_MATCH = True
TYPE_LABEL_HEIGHT = 30 if SHOW_TYPE_LABELS else 0
BLOCK_LABEL_WIDTH = 84 if SHOW_BLOCK_LABELS else 0


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest(render_root: Path) -> dict:
    manifest_path = render_root / "render_manifest.json"
    if not manifest_path.exists():
        return {}
    return read_json(manifest_path)


def normalize_type_key(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", name.lower())
    if cleaned not in TYPE_ALIASES:
        supported = ", ".join(sorted(TYPE_LABELS))
        raise KeyError(f"Unsupported image type '{name}'. Supported keys: {supported}")
    return TYPE_ALIASES[cleaned]


def resolve_output_subdirs(manifest: dict) -> Dict[str, str]:
    output_subdirs = dict(DEFAULT_OUTPUT_SUBDIRS)
    manifest_subdirs = manifest.get("output_subdirs")
    if isinstance(manifest_subdirs, dict):
        for key, value in manifest_subdirs.items():
            if key in output_subdirs and isinstance(value, str):
                output_subdirs[key] = value
    return output_subdirs


def list_block_images(view_dir: Path, subdir_name: str) -> Dict[int, Path]:
    image_dir = view_dir / subdir_name
    if not image_dir.exists():
        raise FileNotFoundError(f"Expected image directory does not exist: {image_dir}")

    block_images: Dict[int, Path] = {}
    for path in sorted(image_dir.glob("block_*.png")):
        match = BLOCK_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        block_images[int(match.group(1))] = path
    return block_images


def select_block_ids(
    per_type_images: Dict[str, Dict[int, Path]],
    requested_block_ids: Optional[Sequence[int]],
) -> List[int]:
    available_sets = [set(paths.keys()) for paths in per_type_images.values()]
    if not available_sets:
        return []

    if requested_block_ids is not None:
        return list(requested_block_ids)

    if STRICT_MODALITY_MATCH:
        common = set.intersection(*available_sets)
        return sorted(common)

    combined = set.union(*available_sets)
    return sorted(combined)


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def encode_png_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def iter_existing_image_paths(per_type_images: Dict[str, Dict[int, Path]]) -> Iterable[Path]:
    for block_images in per_type_images.values():
        for path in block_images.values():
            yield path


def build_svg(
    *,
    view_name: str,
    block_ids: Sequence[int],
    type_keys: Sequence[str],
    per_type_images: Dict[str, Dict[int, Path]],
    output_path: Path,
) -> None:
    first_path = next(iter(iter_existing_image_paths(per_type_images)), None)
    if first_path is None:
        raise RuntimeError(f"No images found for {view_name}")

    src_w, src_h = image_size(first_path)
    cell_w = CELL_WIDTH
    cell_h = int(round(cell_w * (src_h / src_w)))

    label_band_h = TYPE_LABEL_HEIGHT if SHOW_TYPE_LABELS else 0
    content_x0 = OUTER_PAD + BLOCK_LABEL_WIDTH
    content_y0 = OUTER_PAD + label_band_h
    total_w = OUTER_PAD * 2 + BLOCK_LABEL_WIDTH + len(type_keys) * cell_w + max(0, len(type_keys) - 1) * COL_GAP
    total_h = OUTER_PAD * 2 + label_band_h + len(block_ids) * cell_h + max(0, len(block_ids) - 1) * ROW_GAP

    svg_lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{total_w}" height="{total_h}" viewBox="0 0 {total_w} {total_h}">'
        ),
        f'  <rect x="0" y="0" width="{total_w}" height="{total_h}" fill="{BG_COLOR}"/>',
    ]

    if SHOW_TYPE_LABELS:
        label_y = OUTER_PAD + TYPE_LABEL_HEIGHT - 8
        for col_idx, type_key in enumerate(type_keys):
            x0 = content_x0 + col_idx * (cell_w + COL_GAP)
            x_mid = x0 + cell_w / 2.0
            svg_lines.append(
                f'  <text x="{x_mid:.1f}" y="{label_y}" text-anchor="middle" '
                f'font-family="DejaVu Sans, Arial, sans-serif" font-size="20" '
                f'font-weight="700" fill="{TEXT_COLOR}">{escape(TYPE_LABELS[type_key])}</text>'
            )

    for row_idx, block_id in enumerate(block_ids):
        y0 = content_y0 + row_idx * (cell_h + ROW_GAP)

        if SHOW_BLOCK_LABELS:
            block_label_x = OUTER_PAD + BLOCK_LABEL_WIDTH - 10
            block_label_y = y0 + cell_h / 2.0 + 7
            svg_lines.append(
                f'  <text x="{block_label_x}" y="{block_label_y:.1f}" text-anchor="end" '
                f'font-family="DejaVu Sans, Arial, sans-serif" font-size="19" '
                f'font-weight="700" fill="{TEXT_COLOR}">Block {block_id}</text>'
            )

        for col_idx, type_key in enumerate(type_keys):
            x0 = content_x0 + col_idx * (cell_w + COL_GAP)
            svg_lines.append(
                f'  <rect x="{x0}" y="{y0}" width="{cell_w}" height="{cell_h}" '
                f'fill="{MISSING_FILL}" stroke="{BORDER_COLOR}" stroke-width="{BORDER_WIDTH}"/>'
            )

            image_path = per_type_images.get(type_key, {}).get(block_id)
            if image_path is None:
                svg_lines.append(
                    f'  <text x="{x0 + cell_w / 2.0:.1f}" y="{y0 + cell_h / 2.0:.1f}" '
                    f'text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif" '
                    f'font-size="18" fill="{TEXT_COLOR}">missing</text>'
                )
                continue

            svg_lines.append(
                f'  <image x="{x0}" y="{y0}" width="{cell_w}" height="{cell_h}" '
                f'preserveAspectRatio="xMidYMid meet" href="{encode_png_data_uri(image_path)}"/>'
            )

    svg_lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")
    print(f"Saved SVG to {output_path}")


def build_figure(spec: FigureSpec, render_root: Path, manifest: dict) -> None:
    output_subdirs = resolve_output_subdirs(manifest)
    type_keys = [normalize_type_key(name) for name in spec.image_types]

    view_dir = render_root / spec.view_name
    if not view_dir.exists():
        raise FileNotFoundError(f"View directory does not exist: {view_dir}")

    per_type_images = {
        type_key: list_block_images(view_dir, output_subdirs[type_key])
        for type_key in type_keys
    }
    block_ids = select_block_ids(per_type_images, spec.block_ids)
    if not block_ids:
        raise RuntimeError(f"No matching blocks found for {spec.view_name}")

    missing_summary = []
    for block_id in block_ids:
        missing = [
            TYPE_LABELS[type_key]
            for type_key in type_keys
            if block_id not in per_type_images[type_key]
        ]
        if missing:
            missing_summary.append((block_id, missing))

    if STRICT_MODALITY_MATCH and missing_summary:
        details = "; ".join(
            f"block {block_id}: {', '.join(missing)}"
            for block_id, missing in missing_summary
        )
        raise RuntimeError(
            f"Found incomplete block rows under {view_dir} while STRICT_MODALITY_MATCH=True: {details}"
        )

    output_name = spec.output_name or f"{spec.view_name}_appendix.svg"
    output_path = render_root / output_name
    build_svg(
        view_name=spec.view_name,
        block_ids=block_ids,
        type_keys=type_keys,
        per_type_images=per_type_images,
        output_path=output_path,
    )


def main() -> None:
    manifest = load_manifest(RENDER_ROOT)
    for spec in FIGURE_SPECS:
        build_figure(spec, RENDER_ROOT, manifest)


if __name__ == "__main__":
    main()
