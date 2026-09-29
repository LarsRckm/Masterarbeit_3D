"""Batch-generate a diverse dataset of synthetic polar CT slices.

Unlike sample_ct_ddpm_polar.py (one fixed conditioning, all N images in a single
batch -> OOM for large N), this script draws a DIFFERENT real conditioning per
generated image and runs in memory-safe batches.

Diversity sources
-----------------
1. Base diversity: one conditioning (mask, radial_map, cat=format/manuf/chem,
   cont=slice_depth/r_valid_rel) per image from a real slice of the filtered
   cells, planned by ``balanced_samples``: every cell equally often, and within a
   cell every slice depth once before any repeats. With --n = cells x slices each
   real slice is used exactly once -> same cells and depths as normalize_real_cart.
   The real cell behind every image is recorded as ``cell_id`` in the metadata.
2. theta rotation (--theta-rotation): each image's mask+radial_map is cyclically
   shifted along theta (columns). The cell is ~rotationally symmetric and the UNet
   uses circular padding, so this yields valid new geometries (tabs / winding
   lead-ins at other angles) from every real cell. cat/cont are theta-invariant.

Each generated image gets its own initial noise, so images are distinct even when
two happen to share a conditioning.

Resume
------
dataset_metadata.json is rewritten after every batch. After an abort (e.g. a
SLURM time limit) run the SAME command again with --resume: the script checks
that settings and conditioning plan match the earlier run and generates only
the missing images, continuing the balanced plan (images of an interrupted
batch are overwritten).

Run (PowerShell / SLURM):
  python -m CT_scan_model.scripts.batch_sample_polar \
    --ckpt   runs/.../checkpoint_297.pt \
    --outdir runs/.../samples_10k \
    --n 10000 --batch-size 50 --theta-rotation \
    --index CT_scan_model/cell_index.json \
    --geometry CT_scan_model/cell_geometry.json \
    --splits CT_scan_model/splits.json \
    --prediction-type v --zero-terminal-snr --beta-schedule linear --sampler ddpm \
    --resume   # continue an aborted run in --outdir (starts fresh if none exists)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from ..dataset_ct_polar import BatteryCTPolarSelectedSamplesDataset
from ..diffusion_cartesian import Diffusion
from ..modules_cartesian_ct import UNet_conditional_cartesian
from ..polar_transform import polar_to_cart
from ._dataset_select import balanced_samples, load_meta, parse_list, select_cells
from .sample_ct_ddpm_polar import (
    _sample_ddim, _sample_ddpm, _to_uint8_np, _polar_black_bg, DISPLAY_BG_VALUE,
)

try:
    from model import config as project_config
except ImportError:  # pragma: no cover
    import config as project_config  # type: ignore

# Identifies the conditioning plan in the metadata; --resume refuses JSONs
# written with a different plan (their images would break the balance).
SCHEDULE_VERSION = "balanced_cells_depths_v1"


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _min_max(counts: Counter):
    return [min(counts.values()), max(counts.values())] if counts else None


def _balance(meta: list) -> dict:
    """How evenly the images so far cover the conditioning cells and depths."""
    cells = Counter(m["cell_id"] for m in meta)
    depths = Counter(round(m["source_rel_depth"], 4) for m in meta)
    return {"n_cells": len(cells), "images_per_cell_min_max": _min_max(cells),
            "n_depths": len(depths), "images_per_depth_min_max": _min_max(depths)}


def _depth_mismatches(meta: list) -> list:
    """Indices whose conditioning depth differs from the planned source slice.

    The dataset silently falls back to the next sample if an image file is
    missing; for such images the recorded cell_id would be wrong.
    """
    return [m["index"] for m in meta
            if abs(m["slice_depth_relative"] - m["source_rel_depth"]) > 1e-4]


def _write_metadata(path: str, header: dict, meta: list, complete: bool) -> None:
    """Write dataset_metadata.json atomically (temp file + os.replace).

    Called after every batch, so a run that is aborted (e.g. by a SLURM time
    limit) still leaves a valid JSON for every image written so far. If the
    process dies during the write itself, the previous version stays intact.
    """
    data = {
        "n": len(meta), "complete": bool(complete), **header,
        "format_distribution": dict(Counter(m["cell_format"] for m in meta)),
        "balance": _balance(meta),
        "depth_mismatch_indices": _depth_mismatches(meta),
        "samples": meta,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _load_resume(path: str, header: dict, plan: list, shifts: list) -> list:
    """Samples of an earlier (aborted) run, checked against the current run.

    Resuming is only allowed with the same settings and the same conditioning
    plan: the earlier images must be exactly the first entries of the current
    plan, otherwise the combined set would no longer be balanced.
    """
    with open(path, "r", encoding="utf-8") as f:
        prev = json.load(f)
    if prev.get("schedule") != header["schedule"]:
        raise SystemExit(
            f"--resume nicht moeglich: {path} stammt aus einer aelteren Skriptversion "
            f"(ohne ausgewogenen Konditionierungsplan). Bitte mit neuem --outdir starten.")
    diffs = [k for k in header if k != "n_target" and prev.get(k) != header[k]]
    if diffs:
        raise SystemExit(
            "--resume nicht moeglich, Einstellungen weichen vom bisherigen Lauf ab:\n  "
            + "\n  ".join(f"{k}: bisher {prev.get(k)!r}, jetzt {header[k]!r}" for k in diffs))
    meta = list(prev.get("samples", []))
    if len(meta) > len(plan):
        raise SystemExit(f"--resume: {path} enthaelt bereits {len(meta)} Bilder, "
                         f"aber --n ist nur {len(plan)}.")
    for i, m in enumerate(meta):
        s = plan[i]
        if (m.get("index") != i or m.get("cell_id") != s["cell_id"]
                or m.get("source_relpath") != s["relpath"] or m.get("theta_shift") != shifts[i]):
            raise SystemExit(
                f"--resume nicht moeglich: Bild {i} in {path} passt nicht zum "
                f"Konditionierungsplan (geaenderter Zellindex oder Split?).")
    return meta


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Batch-generate diverse synthetic polar CT slices.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--n", type=int, required=True, help="Total number of images to generate")
    p.add_argument("--batch-size", type=int, default=50, help="Images per GPU batch (lower if OOM)")

    p.add_argument("--index", default=os.path.join("CT_scan_model", "cell_index.json"))
    p.add_argument("--geometry", default=os.path.join("CT_scan_model", "cell_geometry.json"))
    p.add_argument("--splits", default=os.path.join("CT_scan_model", "splits.json"))
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Which split's cells provide the conditioning templates.")
    p.add_argument("--formats", default=None,
                   help="Comma-separated cell formats to generate, e.g. '18650' or '18650,2170'. "
                        "Omit = all formats in the split.")
    p.add_argument("--manufacturers", default=None,
                   help="Comma-separated manufacturers, e.g. 'EVE' or 'EVE,Samsung'. "
                        "Omit = all manufacturers. Cells vary by manufacturer, so usually set both.")
    p.add_argument("--min-rel-depth", type=float, default=0.1,
                   help="Lower slice-depth bound. Default 0.1 = training range "
                        "(must match normalize_real_cart).")
    p.add_argument("--max-rel-depth", type=float, default=0.9,
                   help="Upper slice-depth bound. Default 0.9 = training range "
                        "(must match normalize_real_cart).")

    p.add_argument("--theta-rotation", action="store_true",
                   help="Cyclically shift each image's mask+radial_map along theta (extra diversity).")
    p.add_argument("--save", choices=["polar", "cartesian", "both"], default="both",
                   help="Which representation to save: 'polar' (model output grid), "
                        "'cartesian' (round back-projected image, for foundation-model "
                        "comparison), or 'both'. Default: both.")

    # Must match how --ckpt was trained
    p.add_argument("--prediction-type", choices=["eps", "v"], default="v")
    p.add_argument("--zero-terminal-snr", action="store_true")
    p.add_argument("--beta-schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--no-radial-map", action="store_true")
    p.add_argument("--noise-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)

    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    p.add_argument("--ddim-steps", type=int, default=200)
    p.add_argument("--ddim-eta", type=float, default=0.0)
    p.add_argument("--cfg-scale", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cart-size", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Continue an aborted run from <outdir>/dataset_metadata.json: same "
                        "settings required, only the missing images are generated. "
                        "Starts from scratch if no metadata exists yet.")
    args = p.parse_args(argv)

    if cv2 is None:
        raise SystemExit("OpenCV (cv2) is required.")
    if int(args.n) < 1:
        raise SystemExit("--n must be >= 1")
    if bool(args.zero_terminal_snr) and str(args.prediction_type) != "v":
        raise SystemExit("--zero-terminal-snr requires --prediction-type v.")

    N_r       = int(getattr(project_config, "POLAR_N_R", 512))
    N_theta   = int(getattr(project_config, "POLAR_N_THETA", 1024))
    pad_value = float(getattr(project_config, "POLAR_PAD_VALUE", -2.0))
    cart_size = int(args.cart_size or getattr(project_config, "CARTESIAN_SIZE", 1024))
    use_radial = not bool(args.no_radial_map)

    _seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Device: {device} | Polar: {N_r}x{N_theta} | Ziel: {args.n} Bilder, "
          f"Batch {args.batch_size}")

    # --- Conditioning templates: one real slice per image, over the FILTERED cells ---
    index_j, geometry_j, splits_j = load_meta(args.index, args.geometry, args.splits)
    split_cells = list(splits_j.get(f"{args.split}_cells", []))
    formats = parse_list(args.formats)
    manufs  = parse_list(args.manufacturers)
    cell_ids = select_cells(geometry_j, split_cells, formats, manufs)
    if not cell_ids:
        raise SystemExit(f"Keine Zellen mit format={formats}, manufacturer={manufs} "
                         f"in Split '{args.split}'.")
    # Balanced conditioning plan (same slice-depth window as normalize_real_cart,
    # default 0.1..0.9): every cell equally often, every depth of a cell once
    # before any repeats. Deterministic for --seed, which --resume relies on.
    plan = balanced_samples(index_j["cells"], cell_ids, int(args.n), int(args.seed),
                            min_rel=args.min_rel_depth, max_rel=args.max_rel_depth)
    if not plan:
        raise SystemExit("Gefilterte Zellen haben keine Slices im Index.")
    # theta shift per image index -- independent of batching and of --resume
    rot_rng = random.Random(int(args.seed) + 1)
    all_shifts = ([rot_rng.randrange(N_theta) for _ in range(int(args.n))]
                  if bool(args.theta_rotation) else [0] * int(args.n))

    meta_path = os.path.join(args.outdir, "dataset_metadata.json")
    # every setting that shapes the images; --resume requires them unchanged
    meta_header = {
        "n_target": int(args.n), "schedule": SCHEDULE_VERSION,
        "ckpt": args.ckpt, "split": args.split,
        "formats": formats, "manufacturers": manufs,
        "min_rel_depth": float(args.min_rel_depth), "max_rel_depth": float(args.max_rel_depth),
        "sampler": args.sampler, "cfg_scale": float(args.cfg_scale),
        **({"ddim_steps": int(args.ddim_steps), "ddim_eta": float(args.ddim_eta)}
           if args.sampler == "ddim" else {}),
        "theta_rotation": bool(args.theta_rotation),
        "prediction_type": args.prediction_type,
        "zero_terminal_snr": bool(args.zero_terminal_snr),
        "beta_schedule": args.beta_schedule, "noise_steps": int(args.noise_steps),
        "beta_start": float(args.beta_start), "beta_end": float(args.beta_end),
        "no_radial_map": bool(args.no_radial_map), "no_ema": bool(args.no_ema),
        "save": args.save, "cart_size": cart_size, "seed": int(args.seed),
    }
    meta = []
    if bool(args.resume) and os.path.isfile(meta_path):
        meta = _load_resume(meta_path, meta_header, plan, all_shifts)
    elif bool(args.resume):
        print(f"Resume: keine {meta_path} gefunden -> starte bei Bild 0")
    start = len(meta)
    if start >= int(args.n):
        _write_metadata(meta_path, meta_header, meta, complete=True)
        print(f"Nichts zu tun: alle {start} Bilder liegen bereits vor.")
        return 0
    if start:
        print(f"Resume: {start} von {args.n} Bildern vorhanden -> generiere ab Index {start}")

    ds = BatteryCTPolarSelectedSamplesDataset(
        args.index, args.geometry, samples=plan[start:], use_radial_map=use_radial)
    loader = DataLoader(
        ds, batch_size=int(args.batch_size), shuffle=False,
        num_workers=int(args.num_workers), drop_last=False,
    )
    print(f"Filter: formats={formats or 'alle'}, manufacturers={manufs or 'alle'} "
          f"-> {len(cell_ids)} Zellen, {len(plan)} Konditionierungen geplant, {len(ds)} offen")

    # --- Model + diffusion ---
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    key = "model" if bool(args.no_ema) else "ema_model"
    if key not in ckpt:
        raise SystemExit(f"Key '{key}' not in checkpoint. Available: {list(ckpt.keys())}")
    model = UNet_conditional_cartesian(
        c_in=int(getattr(project_config, "POLAR_UNET_IN_CHANNELS", 3)),
        num_downs=int(getattr(project_config, "UNET_NUM_DOWNS_POLAR", 5)),
        base_channels=int(getattr(project_config, "POLAR_UNET_BASE_CHANNELS",
                                   project_config.UNET_BASE_CHANNELS)),
    ).to(device)
    model.load_state_dict(ckpt[key])
    model.eval()
    diffusion = Diffusion(
        noise_steps=int(args.noise_steps), beta_start=float(args.beta_start),
        beta_end=float(args.beta_end), schedule=str(args.beta_schedule),
        zero_terminal_snr=bool(args.zero_terminal_snr),
    ).to(device)
    print(f"Geladen: {'EMA' if not args.no_ema else 'raw'} aus {args.ckpt}")

    save_polar = str(args.save) in ("polar", "both")
    save_cart  = str(args.save) in ("cartesian", "both")
    # Getrennte Unterordner pro Repraesentation (statt alles in einen Ordner).
    polar_dir = os.path.join(args.outdir, "polar")
    cart_dir  = os.path.join(args.outdir, "cartesian")
    if save_polar:
        os.makedirs(polar_dir, exist_ok=True)
    if save_cart:
        os.makedirs(cart_dir, exist_ok=True)
    print(f"Speichere: {args.save}  (polar={save_polar}, cartesian={save_cart})")

    fmt_vocab  = project_config.CELL_FORMAT_VOCAB
    man_vocab  = project_config.MANUFACTURER_VOCAB
    chem_vocab = project_config.CHEMISTRY_VOCAB

    # fresh noise stream per start index: a resumed run must not repeat the
    # noise of the images generated before the abort
    _seed_everything(int(args.seed) + start)
    idx = start
    t0 = time.time()
    n_batches = (len(ds) + int(args.batch_size) - 1) // int(args.batch_size)

    for bi, batch in enumerate(loader, start=1):
        x, (cat, cont), _mask_t, _labels = batch
        Bcur = int(x.shape[0])

        mask       = x[:, 1:2].to(device=device, dtype=torch.float32).contiguous()
        radial_map = x[:, 2:3].to(device=device, dtype=torch.float32).contiguous()
        if bool(args.no_radial_map):
            radial_map = torch.zeros_like(radial_map)
        cat  = cat.to(device=device, dtype=torch.long)
        cont = cont.to(device=device, dtype=torch.float32)

        # theta rotation: per-image cyclic shift of mask + radial_map
        # (all shifts are 0 without --theta-rotation)
        shifts = all_shifts[idx: idx + Bcur]
        for b in range(Bcur):
            s = int(shifts[b])
            if s:
                mask[b] = torch.roll(mask[b], s, dims=-1)
                radial_map[b] = torch.roll(radial_map[b], s, dims=-1)

        with torch.no_grad():
            if args.sampler == "ddim":
                samples = _sample_ddim(
                    model, diffusion, (cat, cont), n=Bcur, N_r=N_r, N_theta=N_theta,
                    r_valid_row=N_r - 1, pad_value=pad_value, cfg_scale=float(args.cfg_scale),
                    device=device, ddim_steps=int(args.ddim_steps), ddim_eta=float(args.ddim_eta),
                    mask=mask, radial_map=radial_map, prediction_type=str(args.prediction_type),
                )
            else:
                samples = _sample_ddpm(
                    model, diffusion, (cat, cont), n=Bcur, N_r=N_r, N_theta=N_theta,
                    r_valid_row=N_r - 1, pad_value=pad_value, cfg_scale=float(args.cfg_scale),
                    device=device, mask=mask, radial_map=radial_map,
                    prediction_type=str(args.prediction_type),
                )

        samples_np = samples[:, 0].cpu().numpy()
        mask_np = mask[:, 0].cpu().numpy()
        cont_np = cont.cpu().numpy()
        cat_np = cat.cpu().numpy()

        for b in range(Bcur):
            polar_disp = _polar_black_bg(samples_np[b], mask_np[b])
            files = {}

            if save_polar:
                pfn = f"sample_{idx:05d}_polar.png"
                cv2.imwrite(os.path.join(polar_dir, pfn),
                            _to_uint8_np(np.clip(polar_disp, -1.0, 1.0)))
                files["polar"] = os.path.join("polar", pfn)

            if save_cart:
                # Round back-projected image (same convention as sample_ct_ddpm_polar).
                cx = cy = float(cart_size) / 2.0
                r_max = float(cont_np[b, 1]) * cx
                cart_np = polar_to_cart(polar_disp, cx, cy, r_max, N_r, N_theta,
                                        cart_size, DISPLAY_BG_VALUE)
                cfn = f"sample_{idx:05d}_cart.png"
                cv2.imwrite(os.path.join(cart_dir, cfn),
                            _to_uint8_np(np.clip(cart_np, -1.0, 1.0)))
                files["cartesian"] = os.path.join("cartesian", cfn)

            # the loader walks plan[start:] in order (shuffle=False) -> image idx = plan[idx]
            src = plan[idx]
            meta.append({
                "index": idx, "files": files,
                "cell_id": src["cell_id"],
                "source_relpath": src["relpath"],
                "source_rel_depth": src["rel_depth"],
                "cell_format":  fmt_vocab[int(cat_np[b, 0])],
                "manufacturer": man_vocab[int(cat_np[b, 1])],
                "chemistry":    chem_vocab[int(cat_np[b, 2])],
                "slice_depth_relative": float(cont_np[b, 0]),
                "r_valid_rel":          float(cont_np[b, 1]),
                "theta_shift": int(shifts[b]),
            })
            idx += 1

        # all images of a batch finish together -> refresh the JSON right after
        # writing them, so an aborted run keeps the metadata of every image
        _write_metadata(meta_path, meta_header, meta, complete=False)

        el = time.time() - t0
        eta = el / bi * (n_batches - bi)
        print(f"[Batch {bi}/{n_batches}] {idx}/{args.n} Bilder | "
              f"{el/60:.1f} min | ETA {eta/60:.1f} min", flush=True)

    # --- final metadata (same content as the per-batch writes, marked complete) ---
    _write_metadata(meta_path, meta_header, meta, complete=True)
    dist = Counter(m["cell_format"] for m in meta)
    bal = _balance(meta)
    mism = _depth_mismatches(meta)

    print(f"\nFertig: {idx} Bilder ({idx - start} in diesem Lauf) in {(time.time()-t0)/60:.1f} min")
    print(f"Format-Verteilung: {dict(dist)}")
    print(f"Konditionierung: {bal['n_cells']} Zellen (Bilder pro Zelle min/max "
          f"{bal['images_per_cell_min_max']}), {bal['n_depths']} Schnitttiefen "
          f"(Bilder pro Tiefe min/max {bal['images_per_depth_min_max']})")
    if mism:
        print(f"WARNUNG: bei {len(mism)} Bildern passt die Schnitttiefe nicht zum geplanten "
              f"Quellschnitt (fehlende Bilddatei?) -- dort ist cell_id unzuverlaessig, "
              f"siehe depth_mismatch_indices.")
    print(f"Metadaten: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
