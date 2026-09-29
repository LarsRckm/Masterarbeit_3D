"""Normalize real CT scans into the SAME format as the synthetic samples.

For a fair comparison (e.g. with a foundation model), real and synthetic images
must differ only in CONTENT, not in preprocessing. A synthetic image produced by
batch_sample_polar is: model output in polar space -> polar_to_cart (centered,
cart_size x cart_size, black background, filtered by the N_r x N_theta polar
grid). A real CT scan has none of that: native size, native position, full
resolution.

This script puts a real image through the EXACT same normalization:

    detect_cell_boundary  ->  cart_to_polar_boundary  ->  polar_to_cart

so afterwards real and synthetic share size, centering and the polar round-trip
filtering. It reuses the identical helpers (_polar_black_bg, DISPLAY_BG_VALUE)
and the identical back-projection scale (r_max = r_valid_rel * cart_size/2) as
batch_sample_polar / sample_ct_ddpm_polar.

Run:
  python -m CT_scan_model.scripts.normalize_real_cart \
    --indir  path/to/real_ct \
    --outdir path/to/real_normalized \
    --cart-size 1024 --save cartesian
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from ..polar_transform import (
    cart_to_polar_boundary, detect_cell_boundary, polar_to_cart,
)
from ._dataset_select import load_meta, parse_list, real_slices, select_cells
from .sample_ct_ddpm_polar import _to_uint8_np, _polar_black_bg, DISPLAY_BG_VALUE

try:
    from model import config as project_config
except ImportError:  # pragma: no cover
    import config as project_config  # type: ignore

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def normalize_real(gray: np.ndarray, N_r: int, N_theta: int, cart_size: int,
                   pad_value: float):
    """Run one real greyscale image through cart -> polar -> cart.

    Returns (polar_disp, cart_norm, info). ``cart_norm`` is the centered,
    cart_size x cart_size, black-background image in the SAME format as the
    synthetic samples.
    """
    # 1. boundary (same detector as the training dataset)
    cx, cy, r_valid_per_angle, _pts, _ang = detect_cell_boundary(gray, N_theta=N_theta)

    # 2. cart -> polar, normalising to [-1, 1] first (training order)
    img_norm = (gray.astype(np.float32) / 255.0) * 2.0 - 1.0
    polar, mask, r_max_orig = cart_to_polar_boundary(
        image=img_norm, cx=cx, cy=cy, r_valid_per_angle=r_valid_per_angle,
        N_r=N_r, N_theta=N_theta, pad_value=pad_value,
    )

    # r_valid_rel exactly as dataset_ct_polar._build_conditioning computes it,
    # so the back-projection scale matches the synthetic images bit-for-bit.
    image_half_size = float(min(gray.shape[0], gray.shape[1])) / 2.0
    r_valid_rel = float(np.clip(r_max_orig / image_half_size, 0.0, 1.0))

    # 3. black out padding, then polar -> cart (identical to batch_sample_polar)
    polar_disp = _polar_black_bg(polar, mask)
    cx_o = cy_o = float(cart_size) / 2.0
    r_max_bp = r_valid_rel * cx_o
    cart_norm = polar_to_cart(polar_disp, cx_o, cy_o, r_max_bp,
                              N_r, N_theta, cart_size, DISPLAY_BG_VALUE)

    info = {
        "cx": float(cx), "cy": float(cy),
        "src_height": int(gray.shape[0]), "src_width": int(gray.shape[1]),
        "r_max_orig_px": float(r_max_orig), "r_valid_rel": r_valid_rel,
    }
    return polar_disp, cart_norm, info


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description="Normalize real CT scans into the synthetic-sample format (cart->polar->cart)."
    )
    # Input source A: INDEX mode — select the real slices of the same cells you
    # generate synthetically (by format/manufacturer, within a slice-depth range).
    p.add_argument("--index", help="cell_index.json (index mode)")
    p.add_argument("--geometry", help="cell_geometry.json (index mode)")
    p.add_argument("--splits", help="splits.json (index mode)")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--formats", default=None,
                   help="Comma-separated formats, e.g. '18650' (index mode). Match the synthetic run.")
    p.add_argument("--manufacturers", default=None,
                   help="Comma-separated manufacturers, e.g. 'EVE' (index mode). Match the synthetic run.")
    p.add_argument("--min-rel-depth", type=float, default=0.1,
                   help="Lower slice-depth bound (index mode). Default 0.1 = training range.")
    p.add_argument("--max-rel-depth", type=float, default=0.9,
                   help="Upper slice-depth bound (index mode). Default 0.9 = training range.")
    # Input source B: ad-hoc folder / single image (no filtering)
    p.add_argument("--indir", help="Folder of real CT images (ad-hoc mode)")
    p.add_argument("--image", help="Single real CT image (ad-hoc mode)")

    p.add_argument("--outdir", required=True)
    p.add_argument("--cart-size", type=int, default=None,
                   help="Output size; MUST match the synthetic images (default: CARTESIAN_SIZE).")
    p.add_argument("--save", choices=["cartesian", "polar", "both"], default="cartesian",
                   help="Which representation to write. Default: cartesian (the round form).")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    if cv2 is None:
        raise SystemExit("OpenCV (cv2) is required.")

    N_r       = int(getattr(project_config, "POLAR_N_R", 512))
    N_theta   = int(getattr(project_config, "POLAR_N_THETA", 1024))
    pad_value = float(getattr(project_config, "POLAR_PAD_VALUE", -2.0))
    cart_size = int(args.cart_size or getattr(project_config, "CARTESIAN_SIZE", 1024))
    save_polar = str(args.save) in ("polar", "both")
    save_cart  = str(args.save) in ("cartesian", "both")

    # --- build the work list: [{path, stem, cell_id, rel_depth, source}, ...] ---
    if args.index:
        if not (args.geometry and args.splits):
            raise SystemExit("Index-Modus braucht --index, --geometry UND --splits.")
        index_j, geometry_j, splits_j = load_meta(args.index, args.geometry, args.splits)
        split_cells = list(splits_j.get(f"{args.split}_cells", []))
        formats = parse_list(args.formats)
        manufs  = parse_list(args.manufacturers)
        cell_ids = select_cells(geometry_j, split_cells, formats, manufs)
        if not cell_ids:
            raise SystemExit(f"Keine Zellen mit format={formats}, manufacturer={manufs} "
                             f"in Split '{args.split}'.")
        _base, slices = real_slices(index_j, cell_ids, args.min_rel_depth, args.max_rel_depth)
        # continuous numbering avoids name clashes (many cells share e.g. '32.0.png')
        items = [{"path": ap, "stem": f"real_{k:05d}", "cell_id": cid,
                  "rel_depth": rd, "source": rp}
                 for k, (cid, ap, rp, rd) in enumerate(slices)]
        print(f"Index-Modus: formats={formats or 'alle'}, manufacturers={manufs or 'alle'}, "
              f"rel_depth in [{args.min_rel_depth}, {args.max_rel_depth}] "
              f"-> {len(cell_ids)} Zellen, {len(items)} Slices")
    elif args.image:
        s = os.path.splitext(os.path.basename(args.image))[0]
        items = [{"path": args.image, "stem": s, "cell_id": None,
                  "rel_depth": None, "source": os.path.basename(args.image)}]
    elif args.indir:
        base = os.path.abspath(args.indir)
        items = [{"path": os.path.join(base, f), "stem": os.path.splitext(f)[0],
                  "cell_id": None, "rel_depth": None, "source": f}
                 for f in sorted(os.listdir(base)) if f.lower().endswith(EXTS)]
    else:
        raise SystemExit("Eingabe waehlen: --index (+ --geometry --splits) ODER --indir/--image.")

    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise SystemExit("Keine Bilder gefunden.")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Normalisiere {len(items)} echte Bilder -> {N_r}x{N_theta} polar -> "
          f"{cart_size}x{cart_size} kartesisch  (save={args.save})")

    meta, failed = [], []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        stem = it["stem"]
        try:
            gray = cv2.imread(it["path"], cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError("nicht lesbar")
            polar_disp, cart_norm, info = normalize_real(gray, N_r, N_theta, cart_size, pad_value)

            files_out = {}
            if save_cart:
                cfn = f"{stem}_norm_cart.png"
                cv2.imwrite(os.path.join(args.outdir, cfn),
                            _to_uint8_np(np.clip(cart_norm, -1.0, 1.0)))
                files_out["cartesian"] = cfn
            if save_polar:
                pfn = f"{stem}_norm_polar.png"
                cv2.imwrite(os.path.join(args.outdir, pfn),
                            _to_uint8_np(np.clip(polar_disp, -1.0, 1.0)))
                files_out["polar"] = pfn

            info.update({"source": it["source"], "cell_id": it["cell_id"],
                         "rel_depth_index": it["rel_depth"], "files": files_out})
            meta.append(info)
            print(f"[{i}/{len(items)}] {stem[:44]:44s} r_valid_rel={info['r_valid_rel']:.4f}  "
                  f"src={info['src_width']}x{info['src_height']}")
        except Exception as e:
            failed.append((it["source"], str(e)))
            print(f"[{i}/{len(items)}] {stem[:44]:44s} FEHLER: {e}")

    with open(os.path.join(args.outdir, "normalization_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n": len(meta), "N_r": N_r, "N_theta": N_theta, "cart_size": cart_size,
            "pad_value": pad_value, "items": meta,
        }, f, indent=2)

    print(f"\nFertig in {(time.time()-t0):.1f}s: {len(meta)} normalisiert, {len(failed)} fehlgeschlagen")
    if failed:
        for fn, err in failed:
            print(f"  {fn}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
