"""Build a deterministic cell/image index for CT diffusion training.

This script scans a dataset folder structure containing one or more directories
named "slices" at arbitrary depth, e.g.:

  BASE_PATH/.../<cell_format>/.../slices/<cell_id>/radial_images/*.png

where <cell_format> contains one of: 18650 / 2170. (4680 is excluded.)

It groups images by cell folder ("cell_id") and filters slices by relative depth
in [min_rel_depth, max_rel_depth].

Outputs a JSON file that is later consumed by:
- precompute_geometry.py
- build_splits.py
- train_ct_ddpm.py

Run (PowerShell, with venv):
  . "C:/Users/larsr/Documents/PythonVenv/Scripts/Activate.ps1"; \
  python -m CT_scan_model.scripts.build_cell_index --out CT_scan_model/cell_index.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import config
    import battery_metadata
except ImportError:  # pragma: no cover
    import config  # type: ignore
    import battery_metadata  # type: ignore


@dataclass(frozen=True)
class ImageEntry:
    image_relpath: str
    abs_depth: float
    rel_depth: float


def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def build_index(
    base_path: str,
    min_rel_depth: float = 0.1,
    max_rel_depth: float = 0.9,
) -> Dict[str, dict]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"BASE_PATH not found: '{base_path}'. Please set model/config.py::BASE_PATH or pass --base-path."
        )

    cells: Dict[str, dict] = {}

    # We accept additional intermediate directory levels (e.g. manufacturer/voltage).
    # 4680 is intentionally excluded so it is never indexed / trained on.
    valid_formats = ("18650", "2170")

    for root, dirs, _files in os.walk(base_path):
        # Only consider directories that directly contain a "slices" directory.
        if "slices" not in dirs:
            continue

        slices_dir = os.path.join(root, "slices")
        rel_slices = os.path.relpath(slices_dir, base_path).replace("\\", "/")
        parts = [p for p in rel_slices.split("/") if p]

        fmt_token = next((p for p in parts if p in valid_formats), None)
        if fmt_token is None:
            continue
        cell_format = battery_metadata.extract_cell_format(fmt_token)

        max_height = battery_metadata.get_max_height_from_format(cell_format)
        if max_height is None or max_height == 0:
            continue

        for cell_dir in sorted(_safe_listdir(slices_dir)):
            cell_path = os.path.join(slices_dir, cell_dir)
            if not os.path.isdir(cell_path):
                continue

            radial_dir = os.path.join(cell_path, "radial_images")
            if not os.path.isdir(radial_dir):
                continue

            # Group key for splitting.
            cell_id = os.path.relpath(cell_path, base_path).replace("\\", "/")

            entries: List[ImageEntry] = []
            for fn in sorted(_safe_listdir(radial_dir)):
                if not fn.lower().endswith(".png"):
                    continue
                abs_depth = float(battery_metadata.extract_absolute_depth(fn))
                rel_depth = abs_depth / float(max_height)
                if not (min_rel_depth <= rel_depth <= max_rel_depth):
                    continue

                image_relpath = os.path.join(cell_id, "radial_images", fn).replace("\\", "/")
                entries.append(ImageEntry(image_relpath=image_relpath, abs_depth=abs_depth, rel_depth=rel_depth))

            if not entries:
                continue

            cells[cell_id] = {
                "cell_format": cell_format,
                "images": [
                    {"relpath": e.image_relpath, "abs_depth": e.abs_depth, "rel_depth": e.rel_depth}
                    for e in entries
                ],
            }

    return {
        "base_path": base_path.replace("\\", "/"),
        "min_rel_depth": float(min_rel_depth),
        "max_rel_depth": float(max_rel_depth),
        "cells": cells,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build CT cell index (10-90% slices) for training.")
    p.add_argument("--base-path", default=config.BASE_PATH, help="Dataset BASE_PATH (default: model.config.BASE_PATH)")
    p.add_argument("--min-rel-depth", type=float, default=0.1)
    p.add_argument("--max-rel-depth", type=float, default=0.9)
    p.add_argument("--out", default=os.path.join("CT_scan_model", "cell_index.json"), help="Output JSON path")
    args = p.parse_args(argv)

    index = build_index(args.base_path, min_rel_depth=args.min_rel_depth, max_rel_depth=args.max_rel_depth)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    n_cells = len(index["cells"])
    n_images = sum(len(v["images"]) for v in index["cells"].values())
    print(f"Wrote index: {args.out}  (cells={n_cells}, images={n_images})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
