"""Polar coordinate transform utilities for battery CT scans.

Coordinate convention
---------------------
Row i  → physical radius  r = i * (r_max / (N_r - 1))
         where r_max = max(r_valid_per_angle) is computed per cell
Col j  → angle            θ = j * (2π / N_θ),  θ=0 points in +x direction

Boundary detection
------------------
`detect_cell_boundary` traces the actual cell edge per angle (Outside→Inside on
the Otsu binary image) and returns the real boundary — no circle is fitted.
The result is an array r_valid_per_angle[N_theta] giving the maximum valid
radius for every angular column in the polar image.

Padding
-------
Pixels where r > r_valid_per_angle[θ] receive `pad_value` and are flagged 0 in
the returned padding_mask.  The valid cell region is flagged 1.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from scipy.ndimage import gaussian_filter1d  # type: ignore
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def _smooth_profile(x: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 1-D profile with a Gaussian filter (falls back to moving average)."""
    if _HAS_SCIPY:
        return gaussian_filter1d(x.astype(np.float64), sigma=sigma)
    k = max(1, int(sigma * 3))
    return np.convolve(x.astype(np.float64), np.ones(k) / k, mode="same")


# ---------------------------------------------------------------------------
# Boundary detection (adapted from visualize_cell_mask.py)
# ---------------------------------------------------------------------------

def detect_cell_boundary(
    gray: np.ndarray,
    N_theta: int,
    kernel_size: int = 25,
    n_trace_angles: int = 720,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Detect the actual cell boundary by tracing edge points per angle.

    Replicates the Outside→Inside edge-tracing from visualize_cell_mask.py.
    No circle is fitted — the raw per-angle radius is returned directly.

    Parameters
    ----------
    gray : uint8 greyscale image [H, W]
    N_theta : number of angular bins — r_valid_per_angle will have this length
    kernel_size : morphology kernel size for Otsu cleanup (odd, default 25)
    n_trace_angles : number of angles used during tracing (default 720)

    Returns
    -------
    cx, cy : estimated cell centre (mean of detected edge points)
    r_valid_per_angle : float32 [N_theta] — valid radius for each polar column
    edge_pts_xy : float32 [n_trace_angles, 2] — raw (x, y) edge points
    edge_angles_rad : float64 [n_trace_angles] — corresponding angles in radians
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for boundary detection.")

    h, w = gray.shape[:2]
    cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0

    # --- Otsu threshold ---
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, bw_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = int(max(3, kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    _ = cv2.morphologyEx(bw_otsu, cv2.MORPH_CLOSE, kernel)
    _ = cv2.morphologyEx(_, cv2.MORPH_OPEN, kernel)

    # --- Outside→Inside edge tracing ---
    r_max_trace = int(max(1.0, min(cx0, cy0, (w - 1) - cx0, (h - 1) - cy0)))
    trace_angles = np.linspace(0.0, 2.0 * math.pi, n_trace_angles, endpoint=False)

    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_angles_found: list[float] = []
    edge_r_found: list[float] = []

    for a in trace_angles:
        ca = math.cos(a)
        sa = math.sin(a)
        found = False
        for r in range(r_max_trace, 0, -1):
            xi = int(round(cx0 + r * ca))
            yi = int(round(cy0 + r * sa))
            if 0 <= xi < w and 0 <= yi < h and int(bw_otsu[yi, xi]) > 0:
                edge_x.append(float(xi))
                edge_y.append(float(yi))
                edge_angles_found.append(float(a))
                edge_r_found.append(float(r))
                found = True
                break
        if not found:
            # Fallback: use image-centre distance as safe minimum.
            edge_x.append(float(cx0 + 1.0 * ca))
            edge_y.append(float(cy0 + 1.0 * sa))
            edge_angles_found.append(float(a))
            edge_r_found.append(1.0)

    edge_pts_xy = np.stack([edge_x, edge_y], axis=1).astype(np.float32)
    edge_angles_rad = np.asarray(edge_angles_found, dtype=np.float64)
    edge_r_trace = np.asarray(edge_r_found, dtype=np.float64)

    # --- Cell centre: centroid of edge points ---
    cx = float(np.mean(edge_pts_xy[:, 0]))
    cy = float(np.mean(edge_pts_xy[:, 1]))

    # Recompute per-angle radius relative to the detected centre.
    dx = edge_pts_xy[:, 0] - cx
    dy = edge_pts_xy[:, 1] - cy
    edge_r_from_centre = np.sqrt(dx ** 2 + dy ** 2).astype(np.float64)

    # --- Interpolate to N_theta bins ---
    # Wrap the angle array so np.interp can interpolate periodically.
    theta_out = np.linspace(0.0, 2.0 * math.pi, N_theta, endpoint=False)

    # Wrap trace angles and radii for periodic interpolation (append first point at 2π).
    angles_wrap = np.append(edge_angles_rad, edge_angles_rad[0] + 2.0 * math.pi)
    r_wrap = np.append(edge_r_from_centre, edge_r_from_centre[0])

    r_valid_per_angle = np.interp(theta_out, angles_wrap, r_wrap).astype(np.float32)

    return cx, cy, r_valid_per_angle, edge_pts_xy, edge_angles_rad


# ---------------------------------------------------------------------------
# Polar transform — boundary-aware (non-circular)
# ---------------------------------------------------------------------------

def cart_to_polar_boundary(
    image: np.ndarray,
    cx: float,
    cy: float,
    r_valid_per_angle: np.ndarray,
    N_r: int,
    N_theta: int,
    pad_value: float = -2.0,
    interpolation: int = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Transform a normalised cartesian image to polar coordinates using the
    actual (non-circular) cell boundary.

    The radial scale is determined per-cell from the maximum boundary radius
    (r_max = max(r_valid_per_angle)), so the widest angle always fills row N_r-1.

    Parameters
    ----------
    image : float32 ndarray [H, W], values in [-1, 1]
    cx, cy : cell centre in image pixel coordinates
    r_valid_per_angle : float32 [N_theta] — maximum valid radius per angular column
    N_r : number of radial samples (height of output)
    N_theta : number of angular samples (width of output)
    pad_value : fill value for pixels outside the cell boundary
    interpolation : cv2 interpolation flag; None -> cv2.INTER_LINEAR (default).
        Pass ``cv2.INTER_NEAREST`` for LABEL/segmentation maps. Bilinear
        interpolation averages neighbouring pixels, which is correct for
        intensities but invents non-existent classes for labels (a pixel between
        class 2 and 4 becomes 3, a real but wrong class).

    Returns
    -------
    polar_img    : float32 [N_r, N_theta]
    padding_mask : float32 [N_r, N_theta], 1 = inside cell boundary, 0 = padding
    r_max        : float — per-cell radial scale (= max boundary radius)
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for polar transform.")

    if len(r_valid_per_angle) != N_theta:
        raise ValueError(
            f"r_valid_per_angle length {len(r_valid_per_angle)} must match N_theta={N_theta}"
        )

    r_max = float(np.max(r_valid_per_angle))
    r_scale = r_max / max(1.0, float(N_r - 1))
    theta_scale = 2.0 * math.pi / float(N_theta)

    rows = np.arange(N_r, dtype=np.float32)
    cols = np.arange(N_theta, dtype=np.float32)
    R_idx, C_idx = np.meshgrid(rows, cols, indexing="ij")  # [N_r, N_theta]

    r_phys = R_idx * r_scale                                # [N_r, N_theta]
    theta_phys = C_idx * theta_scale                        # [N_r, N_theta]

    # Source pixel coordinates in the cartesian image.
    map_x = (float(cx) + r_phys * np.cos(theta_phys)).astype(np.float32)
    map_y = (float(cy) + r_phys * np.sin(theta_phys)).astype(np.float32)

    polar = cv2.remap(
        image.astype(np.float32),
        map_x,
        map_y,
        interpolation=(cv2.INTER_LINEAR if interpolation is None else int(interpolation)),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(pad_value),
    )

    # Per-column cutoff: r_valid_per_angle[j] is the valid radius for column j.
    # Broadcasting: r_phys [N_r, N_theta] vs r_valid_per_angle [N_theta].
    valid = r_phys <= r_valid_per_angle[np.newaxis, :]      # [N_r, N_theta]
    padding_mask = valid.astype(np.float32)
    polar[~valid] = float(pad_value)

    return polar, padding_mask, r_max


# ---------------------------------------------------------------------------
# Inverse transform (shared for both circle and boundary variants)
# ---------------------------------------------------------------------------

def polar_to_cart(
    polar_img: np.ndarray,
    cx: float,
    cy: float,
    r_max: float,
    N_r: int,
    N_theta: int,
    out_size: int,
    pad_value: float = -2.0,
) -> np.ndarray:
    """Inverse transform: polar [N_r, N_theta] → cartesian [out_size, out_size].

    Pixels at radius > r_max receive pad_value.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for polar transform.")

    r_scale = float(r_max) / max(1.0, float(N_r - 1))
    theta_scale = 2.0 * math.pi / float(N_theta)

    yy, xx = np.mgrid[0:out_size, 0:out_size]
    dx = xx.astype(np.float32) - float(cx)
    dy = yy.astype(np.float32) - float(cy)
    r_phys = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx) % (2.0 * math.pi)

    map_y_p = (r_phys / r_scale).astype(np.float32)     # row  (radial)
    map_x_p = (theta / theta_scale).astype(np.float32)  # col  (angular)

    # Wrap-pad: append column 0 at position N_theta so bilinear interpolation
    # near θ=0/360° finds the correct neighbour instead of BORDER_CONSTANT.
    polar_padded = np.concatenate(
        [polar_img, polar_img[:, :1]], axis=1
    ).astype(np.float32)  # [N_r, N_theta+1]

    cart = cv2.remap(
        polar_padded,
        map_x_p,
        map_y_p,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(pad_value),
    )
    return cart


# ---------------------------------------------------------------------------
# Legacy helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def cart_to_polar(
    image: np.ndarray,
    cx: float,
    cy: float,
    r_valid: float,
    N_r: int,
    N_theta: int,
    pad_value: float = -2.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Circular-mask variant: wraps cart_to_polar_boundary with a uniform radius."""
    r_valid_per_angle = np.full(N_theta, float(r_valid), dtype=np.float32)
    return cart_to_polar_boundary(
        image, cx, cy, r_valid_per_angle, N_r, N_theta, pad_value
    )


def make_padding_mask(N_r: int, N_theta: int, r_valid_row: float) -> np.ndarray:
    """Build a uniform (circular) padding mask — kept for compatibility."""
    rows = np.arange(N_r, dtype=np.float32)
    mask = (rows <= float(r_valid_row)).astype(np.float32)
    return np.broadcast_to(mask[:, None], (N_r, N_theta)).copy()


# ---------------------------------------------------------------------------
# Per-angle Mandrel / Ring (Can) boundary detection
# ---------------------------------------------------------------------------

def detect_regions_per_angle(
    polar_img: np.ndarray,
    padding_mask: np.ndarray,
    r_valid_per_angle: np.ndarray,
    r_scale: float,
    n_trace_angles: int = 720,
    smooth_sigma: float = 5.0,
    ring_search_frac: float = 0.12,
    mandrel_search_frac: float = 0.50,
    min_peak_rel_height: float = 0.10,
    return_debug: bool = False,
):
    """Detect Mandrel- and Ring(Can)-boundary per angle from column gradients.

    For each of ``n_trace_angles`` angles, the corresponding polar column is
    read out, smoothed, and its radial gradient analysed:

    - Ring boundary    : last (outermost) significant gradient peak in the last
                         ``ring_search_frac`` of the valid column range
                         (the bright, high-absorption can wall).
    - Mandrel boundary : first significant gradient peak in the first
                         ``mandrel_search_frac`` of the valid column range
                         (mandrel → jelly-roll transition).

    The ``n_trace_angles`` raw values are then interpolated to ``N_theta``
    bins (periodic), mirroring ``detect_cell_boundary``.

    This is the single source of truth used BOTH by training
    (``dataset_ct_polar._compute_region_labels``) and by the visualisation
    scripts (``test_region_detection`` re-exports this with ``return_debug=True``).

    Returns
    -------
    If ``return_debug`` is False (default, used by training):
        (r_mandrel_per_angle, r_ring_per_angle) : float32 [N_theta] each.
    If ``return_debug`` is True (used by the test scripts):
        a dict additionally containing the raw traces and one example column
        (trace_angles_rad, mandrel_trace, ring_trace, debug_col, debug_profile,
         debug_gradient, debug_r_mandrel, debug_r_ring) for plotting.
    """
    N_r, N_theta = polar_img.shape

    trace_angles_rad = np.linspace(0.0, 2.0 * math.pi, n_trace_angles, endpoint=False)
    trace_cols = (trace_angles_rad / (2.0 * math.pi) * N_theta).astype(int) % N_theta

    mandrel_trace = np.zeros(n_trace_angles, dtype=np.float32)
    ring_trace    = np.zeros(n_trace_angles, dtype=np.float32)

    # Debug capture for one example column (90°), only used when return_debug=True.
    debug_col      = int(trace_cols[n_trace_angles // 4])
    debug_profile  = None
    debug_gradient = None
    debug_r_m      = 0
    debug_r_r      = 0

    for k, j in enumerate(trace_cols):
        r_valid_j   = float(r_valid_per_angle[j])
        r_valid_idx = min(N_r - 1, max(1, int(round(r_valid_j / max(1e-8, r_scale)))))

        profile = polar_img[:r_valid_idx, j].astype(np.float64)
        n = len(profile)
        if n < 10:
            ring_trace[k]    = float(r_valid_idx - 1)
            mandrel_trace[k] = 0.0
            continue

        profile_smooth = _smooth_profile(profile, sigma=smooth_sigma)
        grad           = np.gradient(profile_smooth)

        # --- Ring boundary: LAST significant local peak in the last ring_search_frac ---
        # (matches test_region_detection.detect_regions_per_angle: search backwards
        #  from the outermost row for the first local maximum above threshold, so the
        #  detected can edge equals what test_corridor_weights.py visualises.)
        ring_start = max(1, int(n * (1.0 - ring_search_frac)))
        threshold  = float(np.max(np.abs(grad))) * min_peak_rel_height
        found = False
        for i in range(n - 2, ring_start - 1, -1):
            if (grad[i] > grad[i - 1] and
                    grad[i] > grad[i + 1] and
                    grad[i] > threshold):
                ring_trace[k] = float(i)
                found = True
                break
        if not found:
            ring_local = grad[ring_start:]
            ring_trace[k] = float(ring_start + int(np.argmax(ring_local)))

        # --- Mandrel boundary: first significant peak in the first mandrel_search_frac ---
        mandrel_end = max(2, int(n * mandrel_search_frac))
        mand_region = grad[:mandrel_end]
        threshold   = float(np.max(np.abs(grad))) * min_peak_rel_height

        found = False
        for i in range(1, mandrel_end - 1):
            if (mand_region[i] > mand_region[i - 1] and
                    mand_region[i] > mand_region[i + 1] and
                    mand_region[i] > threshold):
                mandrel_trace[k] = float(i)
                found = True
                break
        if not found:
            mandrel_trace[k] = float(int(np.argmax(mand_region)))

        # Capture the example column for the debug plots.
        if j == debug_col:
            debug_profile  = profile_smooth
            debug_gradient = grad
            debug_r_m      = int(mandrel_trace[k])
            debug_r_r      = int(ring_trace[k])

    # --- Interpolate to N_theta bins (periodic) ---
    theta_out   = np.linspace(0.0, 2.0 * math.pi, N_theta, endpoint=False)
    angles_wrap = np.append(trace_angles_rad, trace_angles_rad[0] + 2.0 * math.pi)

    m_wrap = np.append(mandrel_trace, mandrel_trace[0])
    r_wrap = np.append(ring_trace,    ring_trace[0])

    r_mandrel_per_angle = np.interp(theta_out, angles_wrap, m_wrap).astype(np.float32)
    r_ring_per_angle    = np.interp(theta_out, angles_wrap, r_wrap).astype(np.float32)

    if not return_debug:
        return r_mandrel_per_angle, r_ring_per_angle

    return {
        "r_mandrel_per_angle": r_mandrel_per_angle,
        "r_ring_per_angle":    r_ring_per_angle,
        "trace_angles_rad":    trace_angles_rad,
        "mandrel_trace":       mandrel_trace,
        "ring_trace":          ring_trace,
        "debug_col":           int(debug_col),
        "debug_profile":       debug_profile if debug_profile is not None else np.zeros(10),
        "debug_gradient":      debug_gradient if debug_gradient is not None else np.zeros(10),
        "debug_r_mandrel":     int(debug_r_m),
        "debug_r_ring":        int(debug_r_r),
    }
