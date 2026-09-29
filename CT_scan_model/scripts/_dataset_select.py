"""Shared cell selection by format / manufacturer, for the sampling and the
real-image normalization scripts.

Both batch_sample_polar (synthetic) and normalize_real_cart (real) import this,
so the SAME cells (format + manufacturer, within a slice-depth range) drive both
sides — that is what makes synthetic and real comparable.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional


def load_meta(index_json: str, geometry_json: str, splits_json: str):
    """Load the three metadata files. Returns (index, geometry_cells, splits)."""
    with open(index_json, "r", encoding="utf-8") as f:
        index = json.load(f)
    with open(geometry_json, "r", encoding="utf-8") as f:
        geometry = json.load(f)["cells"]
    with open(splits_json, "r", encoding="utf-8") as f:
        splits = json.load(f)
    return index, geometry, splits


def parse_list(s: Optional[str]):
    """'18650,2170' -> ['18650','2170']; None/'' -> None (no filter)."""
    if not s:
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def select_cells(geometry, split_cell_ids, formats=None, manufacturers=None):
    """Filter split cell_ids by cell_format and/or manufacturer (from geometry).

    ``formats`` / ``manufacturers`` are lists (or None = no filter on that field).
    Returns the matching cell_ids, preserving split order.
    """
    fmts = set(formats) if formats else None
    mans = set(manufacturers) if manufacturers else None
    out = []
    for c in split_cell_ids:
        g = geometry.get(c, {})
        if fmts is not None and str(g.get("cell_format")) not in fmts:
            continue
        if mans is not None and str(g.get("manufacturer")) not in mans:
            continue
        out.append(c)
    return out


def cell_format_manufacturer(geometry, cell_id):
    g = geometry.get(cell_id, {})
    return str(g.get("cell_format", "Unknown")), str(g.get("manufacturer", "Unknown"))


def uniform_samples(index_cells, cell_ids, n, seed,
                    min_rel=None, max_rel=None):
    """Draw ``n`` slice samples spread uniformly over ``cell_ids`` (mirrors
    BatteryCTPolarUniformCellsMaxPicturesDataset), each a random slice of its cell.

    Only slices within [min_rel, max_rel] (if given) are eligible. The cell_index
    is already depth-filtered at build time, so this is usually a redundant guard.
    Returns a list of {"cell_id", "relpath", "rel_depth"} dicts.
    """
    cell_to_imgs = {}
    for c in cell_ids:
        imgs = [im for im in index_cells.get(c, {}).get("images", [])
                if _in_range(float(im.get("rel_depth", 0.0)), min_rel, max_rel)]
        if imgs:
            cell_to_imgs[c] = imgs
    valid = list(cell_to_imgs.keys())
    if not valid:
        return []

    rng = random.Random(int(seed))
    k, r = n // len(valid), n % len(valid)
    schedule = []
    for c in valid:
        schedule.extend([c] * k)
    if r > 0:
        schedule.extend(rng.sample(valid, r))
    rng.shuffle(schedule)

    samples = []
    for c in schedule:
        imgs = cell_to_imgs[c]
        im = imgs[rng.randrange(len(imgs))]
        samples.append({"cell_id": c, "relpath": im["relpath"],
                        "rel_depth": float(im.get("rel_depth", 0.0))})
    return samples


def balanced_samples(index_cells, cell_ids, n, seed,
                     min_rel=None, max_rel=None):
    """Plan ``n`` conditioning slices, balanced over cells AND slice depths.

    Unlike ``uniform_samples`` (a random slice per draw, with replacement), no
    slice is reused before every slice of its cell has been used:

    * rounds: each round visits every cell exactly once, in a fixed seeded
      order, so the per-cell counts never differ by more than 1;
    * each cell walks through its own depth-sorted slices cyclically, starting
      at a cell-specific offset;
    * the offsets are spread evenly over the depth range, so every round -- and
      every prefix of the plan -- covers the depths evenly.

    With n = cells x slices per cell every real slice is used exactly once, i.e.
    the synthetic set gets the same cells and depths as normalize_real_cart.
    The plan is deterministic for a given seed and its first m entries do not
    depend on n, which is what makes resuming an aborted run possible.
    Returns a list of {"cell_id", "relpath", "rel_depth"} dicts.
    """
    cell_to_imgs = {}
    for c in cell_ids:
        imgs = [im for im in index_cells.get(c, {}).get("images", [])
                if _in_range(float(im.get("rel_depth", 0.0)), min_rel, max_rel)]
        if imgs:
            cell_to_imgs[c] = sorted(
                imgs, key=lambda im: (float(im.get("rel_depth", 0.0)), im["relpath"]))
    cells = sorted(cell_to_imgs)
    if not cells:
        return []
    random.Random(int(seed)).shuffle(cells)

    # Depth offset per cell position: full blocks of d_ref cells take the
    # offsets 0..d_ref-1, the p remaining cells get evenly spaced offsets.
    # Scaled to each cell's own slice count if cells differ in slice count.
    d_ref = max(len(v) for v in cell_to_imgs.values())
    n_cells = len(cells)
    full = (n_cells // d_ref) * d_ref
    p = n_cells - full
    offsets = []
    for j, c in enumerate(cells):
        sigma = j % d_ref if j < full else (j - full) * d_ref // p
        offsets.append(sigma * len(cell_to_imgs[c]) // d_ref)

    samples = []
    for k in range(int(n)):
        rnd, j = divmod(k, n_cells)
        imgs = cell_to_imgs[cells[j]]
        im = imgs[(offsets[j] + rnd) % len(imgs)]
        samples.append({"cell_id": cells[j], "relpath": im["relpath"],
                        "rel_depth": float(im.get("rel_depth", 0.0))})
    return samples


def real_slices(index, cell_ids, min_rel=None, max_rel=None):
    """All real slices of the given cells within [min_rel, max_rel].

    Returns (base_path, [(cell_id, abs_path, relpath, rel_depth), ...]).
    """
    base = index.get("base_path", "")
    cells = index.get("cells", {})
    out = []
    for c in cell_ids:
        for im in cells.get(c, {}).get("images", []):
            rd = float(im.get("rel_depth", 0.0))
            if not _in_range(rd, min_rel, max_rel):
                continue
            out.append((c, os.path.join(base, im["relpath"]), im["relpath"], rd))
    return base, out


def _in_range(v, lo, hi):
    if lo is not None and v < lo:
        return False
    if hi is not None and v > hi:
        return False
    return True
