"""
Rerun visualizer for the tactile glove + OAK camera capture in this directory.

Layout (rerun blueprint, set manually via the viewer the first time you open it):
    world/cam/rgb              RGB image stream
    world/cam/mono_left        grayscale left
    world/cam/mono_right       grayscale right

    hands/{lh,rh}/silhouette        2D hand silhouette + tactile point cloud (size & color = pressure)
    hands/{lh,rh}/heatfield         splatted continuous "contact heatfield" image (256x256)
    hands/{lh,rh}/force_arrows      2D arrows at finger tips, length = force_N
    hands/{lh,rh}/bend_bars         5 vertical bars showing per-finger bend (normalized 0..1)
    hands/{lh,rh}/imu_pose          a rotated 3D box driven by the hand IMU quaternion

    plots/lh_force_N/*         per-finger force-N scalar timeseries
    plots/rh_force_N/*
    plots/lh_bend/*
    plots/rh_bend/*
    plots/oak_imu/*            high-rate IMU accel/gyro (5880Hz) on its own time axis

All entities share the same `frame_time` timeline, anchored to frames.parquet
`timestamp` (Unix seconds). Top of the rerun viewer lets you scrub it freely.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image


# -----------------------------------------------------------------------------
# Anatomically inspired 2D layout for the tactile points.
# The hand silhouette is purely decorative; coords are in a 1.0 x 1.4 box
# (x in [-0.5, 0.5], y in [0, 1.4]).  Palm pressure has 60 cells laid out in a
# 6x10 grid covering the palm; finger pressure has 12 cells per finger laid
# along 3 columns x 4 rows on each finger pad.
# -----------------------------------------------------------------------------

# Per-finger tip position and orientation (along which the 12 taxels lie).
# Order matters: must match the column order in frames.parquet.
FINGER_LAYOUT = {
    # name      tip_xy            base_xy            width
    "thumb":  ((-0.42, 0.55),    (-0.20, 0.30),     0.10),
    "index":  ((-0.18, 1.32),    (-0.18, 0.78),     0.08),
    "middle": ((-0.02, 1.40),    (-0.02, 0.80),     0.08),
    "ring":   (( 0.14, 1.34),    ( 0.14, 0.78),     0.08),
    "little": (( 0.28, 1.18),    ( 0.26, 0.74),     0.07),
}

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]

# All hand-view vectors and the heatfield image share this pixel canvas so the
# rerun auto-fit doesn't push one of them out of frame.
HAND_VIEW_RES = 480
HAND_BBOX = (-0.55, 0.55, -0.05, 1.95)  # x_lo, x_hi, y_lo, y_hi (world units)


def world_to_pixels(pts: np.ndarray) -> np.ndarray:
    """Convert world (x,y) coords into the hand view's pixel canvas."""
    x_lo, x_hi, y_lo, y_hi = HAND_BBOX
    px = (pts[:, 0] - x_lo) / (x_hi - x_lo) * HAND_VIEW_RES
    py = (1.0 - (pts[:, 1] - y_lo) / (y_hi - y_lo)) * HAND_VIEW_RES   # flip Y
    return np.stack([px, py], axis=1)


def world_radius_to_pixels(r: float | np.ndarray) -> float | np.ndarray:
    """Convert a world-space radius to pixel-space radius (using x scale)."""
    x_lo, x_hi, _, _ = HAND_BBOX
    return r / (x_hi - x_lo) * HAND_VIEW_RES


def finger_taxel_positions(finger: str) -> np.ndarray:
    """12 taxel positions laid out as 3 columns x 4 rows along a finger."""
    (tx, ty), (bx, by), w = FINGER_LAYOUT[finger]
    # Build along-finger axis (base -> tip) and a perpendicular axis.
    axis = np.array([tx - bx, ty - by])
    L = np.linalg.norm(axis)
    axis = axis / max(L, 1e-9)
    perp = np.array([-axis[1], axis[0]])

    # The tactile pad covers roughly the distal half of the finger.
    pts = []
    rows = 4   # along finger
    cols = 3   # across finger
    pad_len = L * 0.55       # pad extends from ~45% to 100% of finger length
    pad_offset = L * 0.45
    for r in range(rows):
        # row 0 closest to palm, row 3 at tip
        along = pad_offset + (r + 0.5) * (pad_len / rows)
        for c in range(cols):
            across = ((c + 0.5) / cols - 0.5) * w
            p = np.array([bx, by]) + along * axis + across * perp
            pts.append(p)
    return np.array(pts)  # (12, 2)


def palm_taxel_positions() -> np.ndarray:
    """60 palm taxel positions laid out as 6 cols x 10 rows on the palm."""
    cols, rows = 6, 10
    pts = []
    # palm box: x in [-0.30, 0.30], y in [0.10, 0.80]
    x0, x1 = -0.30, 0.30
    y0, y1 = 0.10, 0.80
    for r in range(rows):
        for c in range(cols):
            x = x0 + (c + 0.5) / cols * (x1 - x0)
            y = y0 + (r + 0.5) / rows * (y1 - y0)
            pts.append([x, y])
    return np.array(pts)


def hand_silhouette_lines() -> list[np.ndarray]:
    """A simple outline of a palm + 5 fingers, returned as a list of polylines."""
    polylines = []
    # Palm rectangle (slightly rounded approximation as polygon).
    palm = np.array([
        [-0.32, 0.10], [ 0.32, 0.10], [ 0.34, 0.40], [ 0.36, 0.70],
        [ 0.32, 0.80], [-0.32, 0.80], [-0.36, 0.70], [-0.34, 0.40],
        [-0.32, 0.10],
    ])
    polylines.append(palm)

    # Each finger as a tapered capsule (rectangle with rounded tip)
    for name in FINGER_NAMES:
        (tx, ty), (bx, by), w = FINGER_LAYOUT[name]
        axis = np.array([tx - bx, ty - by])
        L = np.linalg.norm(axis)
        axis = axis / max(L, 1e-9)
        perp = np.array([-axis[1], axis[0]])
        # base wider, tip narrower
        w_base = w
        w_tip = w * 0.7
        base_l = np.array([bx, by]) + perp * w_base / 2
        base_r = np.array([bx, by]) - perp * w_base / 2
        # rounded tip = small arc
        tip_center = np.array([tx, ty]) - axis * 0.01
        arc_pts = []
        for theta in np.linspace(-np.pi / 2, np.pi / 2, 9):
            arc_pts.append(tip_center + (np.cos(theta) * perp + np.sin(theta) * axis) * w_tip / 2)
        arc_pts = np.array(arc_pts)
        finger = np.vstack([
            base_l[None, :],
            tip_center + perp * w_tip / 2,
            arc_pts,
            tip_center - perp * w_tip / 2,
            base_r[None, :],
        ])
        polylines.append(finger)
    return polylines


# Color & sizing helpers ------------------------------------------------------

def colormap_inferno(values: np.ndarray) -> np.ndarray:
    """Cheap inferno-ish colormap without depending on matplotlib at runtime."""
    v = np.clip(values, 0.0, 1.0)
    # piecewise rgb stops (black -> purple -> orange -> yellow)
    stops = np.array([
        [0.00, 0.001, 0.000, 0.014],
        [0.20, 0.122, 0.030, 0.281],
        [0.40, 0.371, 0.069, 0.430],
        [0.60, 0.694, 0.166, 0.357],
        [0.80, 0.964, 0.443, 0.117],
        [1.00, 0.988, 0.998, 0.645],
    ])
    out = np.zeros((*v.shape, 3))
    for ch in range(3):
        out[..., ch] = np.interp(v, stops[:, 0], stops[:, 1 + ch])
    return (out * 255).astype(np.uint8)


def _rasterize_hand_mask(silhouette_polylines: list[np.ndarray]) -> np.ndarray:
    """Fill the hand silhouette polygons into a HAND_VIEW_RES x HAND_VIEW_RES uint8 mask (255=hand)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return np.zeros((HAND_VIEW_RES, HAND_VIEW_RES), dtype=np.uint8)
    canvas = Image.new("L", (HAND_VIEW_RES, HAND_VIEW_RES), 0)
    draw = ImageDraw.Draw(canvas)
    for poly_world in silhouette_polylines:
        poly_px = world_to_pixels(poly_world)
        draw.polygon([tuple(p) for p in poly_px], fill=255)
    return np.array(canvas)


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Cheap separable box blur on a 2D float image (used for the glow halo)."""
    if radius <= 0:
        return img
    k = 2 * radius + 1
    kernel_1d = np.ones(k, dtype=np.float32) / k
    # row pass + col pass via convolve in pure numpy
    pad = np.pad(img, ((0, 0), (radius, radius)), mode="edge")
    blurred = np.zeros_like(img)
    for i in range(k):
        blurred += pad[:, i:i + img.shape[1]] * kernel_1d[i]
    pad2 = np.pad(blurred, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(img)
    for i in range(k):
        out += pad2[i:i + img.shape[0], :] * kernel_1d[i]
    return out


_BASE_CACHE: dict[int, np.ndarray] = {}


def _get_skin_base(hand_mask: np.ndarray, skin_color=(56, 52, 64)) -> np.ndarray:
    """Cached base layer (skin-tone fill inside silhouette, dark elsewhere)."""
    key = id(hand_mask)
    if key in _BASE_CACHE:
        return _BASE_CACHE[key]
    mask = (hand_mask > 0).astype(np.float32)
    inset = _box_blur(mask, 6)
    bg = np.array([8, 9, 14], dtype=np.float32)
    skin = np.array(skin_color, dtype=np.float32)
    base = (bg[None, None, :] * (1 - mask[..., None])
            + skin[None, None, :] * mask[..., None] * (0.55 + 0.45 * inset[..., None]))
    _BASE_CACHE[key] = base
    return base


def render_hand_canvas(points: np.ndarray, intensities: np.ndarray,
                        hand_mask: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Render the fancy hand canvas: skin base + inferno heatfield + bloom halo.

    Returns (rgb_image, pressure_grid).  pressure_grid feeds the contour overlay.
    """
    H = W = HAND_VIEW_RES
    x_lo, x_hi, y_lo, y_hi = HAND_BBOX

    grid = np.zeros((H, W), dtype=np.float32)
    if np.any(intensities > 0):
        px = ((points[:, 0] - x_lo) / (x_hi - x_lo) * W).astype(int)
        py = ((1.0 - (points[:, 1] - y_lo) / (y_hi - y_lo)) * H).astype(int)
        sigma_px = sigma * W / (x_hi - x_lo)
        r = int(max(3, sigma_px * 2))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_px ** 2))
        for cx, cy, val in zip(px, py, intensities):
            if val <= 0:
                continue
            x_lo_i, x_hi_i = max(0, cx - r), min(W, cx + r + 1)
            y_lo_i, y_hi_i = max(0, cy - r), min(H, cy + r + 1)
            kx_lo, kx_hi = x_lo_i - (cx - r), kernel.shape[1] - ((cx + r + 1) - x_hi_i)
            ky_lo, ky_hi = y_lo_i - (cy - r), kernel.shape[0] - ((cy + r + 1) - y_hi_i)
            grid[y_lo_i:y_hi_i, x_lo_i:x_hi_i] += kernel[ky_lo:ky_hi, kx_lo:kx_hi] * float(val)

    norm = np.clip(grid / 30.0, 0.0, 1.0)
    base = _get_skin_base(hand_mask)

    if norm.max() < 1e-3:
        return np.clip(base, 0, 255).astype(np.uint8), norm

    bloom = np.clip(_box_blur(norm, radius=9) * 1.4, 0.0, 1.0)
    bloom_rgb = colormap_inferno(bloom).astype(np.float32)
    core_rgb = colormap_inferno(norm).astype(np.float32)
    core_alpha = norm[..., None]
    bloom_alpha = bloom[..., None] * 0.55

    out = base + bloom_rgb * bloom_alpha
    out = out * (1 - core_alpha) + core_rgb * core_alpha
    return np.clip(out, 0, 255).astype(np.uint8), norm


def overlay_contours(canvas: np.ndarray, grid: np.ndarray,
                     levels=(0.20, 0.45, 0.75)) -> np.ndarray:
    """Burn faint white iso-pressure contour lines onto the canvas in-place."""
    out = canvas.copy()
    for lvl in levels:
        # detect cells where adjacent samples straddle the level
        above = grid >= lvl
        edge = np.zeros_like(above)
        edge[1:, :] |= above[1:, :] != above[:-1, :]
        edge[:, 1:] |= above[:, 1:] != above[:, :-1]
        alpha = min(0.20 + 0.25 * lvl, 0.5)
        out[edge] = (out[edge] * (1 - alpha) + np.array([245, 245, 245]) * alpha).astype(np.uint8)
    return out


def stamp_text(canvas: np.ndarray, items: list[tuple[tuple[int, int], str, tuple[int, int, int]]],
               anchor: str = "mm") -> np.ndarray:
    """Burn text directly into the canvas using PIL.

    Accepts either an RGB (H,W,3) or RGBA (H,W,4) numpy array. Each label
    draws with a small semi-transparent rounded background pill so it's
    readable on any color.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return canvas
    mode = "RGBA" if canvas.ndim == 3 and canvas.shape[2] == 4 else "RGB"
    img = Image.fromarray(canvas, mode=mode)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    for (x, y), text, color in items:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # anchor: top-left of text
        if anchor == "mm":
            tx, ty = x - tw // 2, y - th // 2 - bbox[1]
        elif anchor == "lt":
            tx, ty = x, y - bbox[1]
        else:
            tx, ty = x, y - bbox[1]
        pad = 3
        draw.rounded_rectangle(
            (tx - pad, ty + bbox[1] - pad, tx + tw + pad, ty + bbox[1] + th + pad),
            radius=4, fill=(15, 15, 22, 200))
        draw.text((tx, ty), text, fill=color + (255,), font=font)
    return np.array(img)


def render_colorbar(width: int = 24, height: int = 200, vmax: float = 40.0) -> np.ndarray:
    """A vertical colorbar (high pressure on top) labeled 0..vmax."""
    vals = np.linspace(1.0, 0.0, height)
    col = colormap_inferno(vals)            # (H, 3)
    bar = np.tile(col[:, None, :], (1, width, 1))
    # add a thin dark border
    bar[:, 0] = 30
    bar[:, -1] = 30
    bar[0, :] = 30
    bar[-1, :] = 30
    return bar


# Force calibration polynomial ------------------------------------------------

def force_N_from_pressure(p_sum: float, coeffs: dict) -> float:
    """Apply the c3/c2/c1/c0 polynomial used in calib.json glove_force entries."""
    c3, c2, c1, c0 = coeffs["c3"], coeffs["c2"], coeffs["c1"], coeffs["c0"]
    return max(0.0, c3 * p_sum ** 3 + c2 * p_sum ** 2 + c1 * p_sum + c0)


# Main logging ----------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="tactile-reader-viz",
        description="Rerun-based viewer for a tactile-glove capture directory.")
    ap.add_argument("data_dir", nargs="?", default=".",
                    help="capture directory (default: current working dir). Must contain "
                         "frames.parquet, rgb/, oak_imu.parquet, calib.json")
    ap.add_argument("--no-images", action="store_true", help="skip RGB images for a faster log")
    ap.add_argument("--no-imu", action="store_true", help="skip high-rate oak_imu.parquet")
    ap.add_argument("--mirror-rh", action="store_true",
                    help="render right hand mirrored (so palms face the viewer). Default: side-by-side.")
    ap.add_argument("--memory-limit", default="25%")
    return ap.parse_args(argv)


def build_blueprint() -> rrb.BlueprintLike:
    """Force a sensible default layout the first time the viewer opens."""
    # Lock the hand views to the full pixel canvas so they don't auto-fit
    # to the per-frame arrow bbox (which would make the view rescale during
    # playback as fingertip forces grow/shrink).
    hand_bounds = rrb.VisualBounds2D(
        x_range=[-8, HAND_VIEW_RES + 8],
        y_range=[-8, HAND_VIEW_RES + 8],
    )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(origin="hands/lh", name="Left hand",
                                  visual_bounds=hand_bounds),
                rrb.Spatial2DView(origin="world/cam/rgb", name="RGB"),
                rrb.Spatial2DView(origin="hands/rh", name="Right hand",
                                  visual_bounds=hand_bounds),
                column_shares=[1, 1.4, 1],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="plots/force_N", name="Finger force (N)"),
                rrb.TimeSeriesView(origin="plots/bend", name="Finger bend (deg)"),
                rrb.TimeSeriesView(origin="plots/oak_imu", name="OAK IMU"),
            ),
            row_shares=[5, 2],
        ),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
    )


def log_static(calib: dict, mirror_rh: bool) -> dict[str, np.ndarray]:
    """Log everything that doesn't change with time. Returns the rasterized
    hand silhouette masks (one per side) so the per-frame renderer can use them
    to do skin-tone fill without re-rasterizing."""
    masks: dict[str, np.ndarray] = {}
    for side in ("lh", "rh"):
        base = f"hands/{side}"
        sil_color = [205, 220, 230] if side == "lh" else [230, 200, 200]

        sil_lines_world = hand_silhouette_lines()
        if side == "rh" and mirror_rh:
            sil_lines_world = [np.column_stack([-p[:, 0], p[:, 1]]) for p in sil_lines_world]

        masks[side] = _rasterize_hand_mask(sil_lines_world)

        sil_lines_px = [world_to_pixels(p) for p in sil_lines_world]
        rr.log(f"{base}/silhouette",
               rr.LineStrips2D(sil_lines_px, colors=sil_color,
                               radii=world_radius_to_pixels(0.005),
                               draw_order=10.0),
               static=True)

        # (Hand titles + colorbar tick labels are stamped into the per-frame canvas.)
    return masks


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.data_dir).resolve()
    if not (root / "frames.parquet").exists():
        raise SystemExit(f"frames.parquet not in {root}. "
                         f"Pass the path to a capture directory: tactile-reader-viz <dir>")

    import json
    calib = json.loads((root / "calib.json").read_text())

    df = pd.read_parquet(root / "frames.parquet")
    duration_s = float(df["timestamp"].max() - df["timestamp"].min())
    print(f"Loaded {len(df)} frames covering "
          f"{duration_s:.2f}s ({len(df) / duration_s:.1f} Hz)")
    # Use seconds-from-start as the timeline value so the viewer shows a clean
    # 0..29s axis (and doesn't choke on 1.78e9-sized Unix timestamps).
    t0 = float(df["timestamp"].min())

    os.environ.setdefault("RERUN_FLUSH_NUM_BYTES", "8000")
    rr.init("tactile_glove_052211", spawn=True)
    # Force-override any blueprint the viewer cached from a previous run.
    # default_blueprint only applies if the user hasn't saved their own; send_blueprint
    # with make_active+make_default overwrites whatever was there.
    rr.send_blueprint(build_blueprint(), make_active=True, make_default=True)
    # Memory cap so we don't blow up the viewer
    try:
        rr.spawn(memory_limit=args.memory_limit)  # idempotent
    except Exception:
        pass

    hand_masks = log_static(calib, args.mirror_rh)
    colorbar = render_colorbar(width=18, height=160, vmax=40.0)

    # Pre-compute taxel positions
    finger_pts = {f: finger_taxel_positions(f) for f in FINGER_NAMES}  # each (12,2)
    palm_pts = palm_taxel_positions()                                  # (60,2)
    n_finger_taxels = 12
    n_palm_taxels = 60

    # ---- per-frame logging ----
    for i, row in df.iterrows():
        t_rel = float(row["timestamp"]) - t0
        rr.set_time("frame_time", duration=t_rel)
        rr.set_time("frame_idx", sequence=int(row["frame_idx"]))

        # Images (logged as encoded files to keep memory small).
        if not args.no_images:
            p = root / row["rgb_path"]
            if p.exists():
                rr.log("world/cam/rgb", rr.EncodedImage(path=str(p)))

        # ---- both hands tactile ----
        for side in ("lh", "rh"):
            # Gather per-finger taxel pressures (12 each) AND per-taxel force_N (12 each).
            # The arrow at the tip uses the per-finger sum of force_N.
            all_taxel_pos = []  # (5*12 + 60, 2)
            all_taxel_pressure = []
            all_taxel_force = []
            tip_pos = []
            tip_force = []
            tip_labels = []
            for f in FINGER_NAMES:
                p_arr = np.asarray(row[f"{side}_{f}_pressure_cal"], dtype=np.float32)   # (12,)
                fN_arr = np.asarray(row[f"{side}_{f}_force_N_cal"], dtype=np.float32)   # (12,)
                pts = finger_pts[f]
                if side == "rh" and args.mirror_rh:
                    pts = pts.copy(); pts[:, 0] *= -1
                all_taxel_pos.append(pts)
                all_taxel_pressure.append(p_arr)
                all_taxel_force.append(fN_arr)

                tip_xy = pts[-3:].mean(axis=0)
                tip_pos.append(tip_xy)
                f_N_total = float(fN_arr.sum())
                tip_force.append(f_N_total)
                tip_labels.append(f"{f[:1].upper()}: {f_N_total:.2f}N")

            palm = np.asarray(row[f"{side}_palm_pressure_cal"], dtype=np.float32)  # (60,)
            palm_xy = palm_pts.copy()
            if side == "rh" and args.mirror_rh:
                palm_xy[:, 0] *= -1
            all_taxel_pos.append(palm_xy)
            all_taxel_pressure.append(palm)
            # palm has no force_N column, use zeros so the array shapes match
            all_taxel_force.append(np.zeros_like(palm))

            all_taxel_pos = np.vstack(all_taxel_pos)              # (120, 2)
            all_taxel_pressure = np.concatenate(all_taxel_pressure)  # (120,)
            all_taxel_force = np.concatenate(all_taxel_force)        # (120,)

            # --- 0. fancy hand canvas: skin fill + bloom + heatfield + contours + colorbar ---
            canvas, pressure_grid = render_hand_canvas(
                all_taxel_pos, all_taxel_pressure,
                hand_mask=hand_masks[side], sigma=0.035,
            )
            canvas = overlay_contours(canvas, pressure_grid, levels=(0.20, 0.45, 0.75))

            # stamp colorbar in bottom-right
            cb_h, cb_w = colorbar.shape[:2]
            margin = 14
            y0 = HAND_VIEW_RES - cb_h - margin
            x0 = HAND_VIEW_RES - cb_w - margin
            canvas[y0:y0 + cb_h, x0:x0 + cb_w] = colorbar

            # All text (title, hand name, T/I/M/R/L, colorbar ticks, fingertip N values)
            # is burned into the image so nothing can be occluded by rerun labels.
            text_items: list[tuple[tuple[int, int], str, tuple[int, int, int]]] = []

            # Hand title
            title_px = world_to_pixels(np.array([[0.0, 1.85]]))[0]
            text_items.append(((int(title_px[0]), int(title_px[1])),
                               "LEFT HAND" if side == "lh" else "RIGHT HAND",
                               (235, 235, 245)))

            # Finger initials below the bend bars
            bar_x = np.linspace(-0.45, 0.45, 5)
            for x, name in zip(bar_x, FINGER_NAMES):
                p = world_to_pixels(np.array([[x, 1.50]]))[0]
                text_items.append(((int(p[0]), int(p[1])), name[0].upper(),
                                   (220, 220, 230)))

            # Colorbar title above the bar, ticks to its left
            cb_title_x = x0 + cb_w // 2
            cb_title_y = y0 - 10
            text_items.append(((cb_title_x, cb_title_y), "pressure", (210, 210, 220)))
            for v_frac, lbl in ((0.0, "0"), (0.5, "20"), (1.0, "40")):
                ty = int(y0 + (1 - v_frac) * cb_h)
                text_items.append(((x0 - 14, ty), lbl, (210, 210, 220)))

            # Compute fingertip + arrow geometry (used by force_arrows below)
            tip_pos_w = np.array(tip_pos)
            tip_force_arr = np.array(tip_force)
            palm_center = np.array([0.0, 0.45])
            dirs_w = tip_pos_w - palm_center
            dirs_w = dirs_w / (np.linalg.norm(dirs_w, axis=1, keepdims=True) + 1e-9)
            arrow_len_w = np.clip(tip_force_arr, 0, 5.0) * 0.08
            vectors_w = dirs_w * arrow_len_w[:, None]
            tip_origins_px = world_to_pixels(tip_pos_w)
            tip_targets_px = world_to_pixels(tip_pos_w + vectors_w)

            canvas = stamp_text(canvas, text_items, anchor="mm")
            rr.log(f"hands/{side}/heatfield", rr.Image(canvas, draw_order=0.0))

            # --- top-most overlay: N values, painted onto a transparent RGBA
            # layer with draw_order=100 so they sit above EVERY other entity
            # (silhouette / taxels / bars / arrows). ---
            overlay = np.zeros((HAND_VIEW_RES, HAND_VIEW_RES, 4), dtype=np.uint8)
            label_pos_w = tip_pos_w - dirs_w * 0.12   # inside the palm, away from bars
            label_px = world_to_pixels(label_pos_w)
            n_items = [((int(lx), int(ly)), f"{f_N:.2f}N", (255, 230, 180))
                       for (lx, ly), f_N in zip(label_px, tip_force_arr)]
            overlay = stamp_text(overlay, n_items, anchor="mm")
            rr.log(f"hands/{side}/n_labels",
                   rr.Image(overlay, color_model="RGBA", draw_order=100.0))

            # --- 1. tactile points (pixel space) ---
            taxel_px = world_to_pixels(all_taxel_pos)
            v_norm = np.clip(all_taxel_pressure / 40.0, 0.0, 1.0)
            colors = colormap_inferno(v_norm)
            alpha = (90 + 165 * v_norm).astype(np.uint8)
            colors = np.concatenate([colors, alpha[:, None]], axis=1)
            radii_px = world_radius_to_pixels(0.010 + 0.030 * v_norm)
            rr.log(f"hands/{side}/taxels",
                   rr.Points2D(taxel_px, colors=colors, radii=radii_px,
                               draw_order=20.0))

            # --- 2. force arrows at fingertips (geometry already computed above) ---
            arrow_vectors_px = tip_targets_px - tip_origins_px
            arrow_colors = colormap_inferno(np.clip(tip_force_arr / 3.0, 0, 1))
            rr.log(f"hands/{side}/force_arrows",
                   rr.Arrows2D(origins=tip_origins_px, vectors=arrow_vectors_px,
                               colors=arrow_colors,
                               radii=world_radius_to_pixels(0.012),
                               draw_order=40.0))

            # --- 3. bend bars at top of view ---
            bend = np.asarray(row[f"{side}_finger_bend_cal"], dtype=np.float32)
            bend = np.clip(bend, 0, 1)
            bar_x = np.linspace(-0.45, 0.45, 5)
            bar_lines_w = []
            bar_colors = []
            for x, b in zip(bar_x, bend):
                y0, y1 = 1.55, 1.55 + 0.30 * float(b)
                bar_lines_w.append(np.array([[x, y0], [x, y1]]))
                bar_colors.append(colormap_inferno(np.array([b]))[0].tolist())
            bar_lines_px = [world_to_pixels(p) for p in bar_lines_w]
            rr.log(f"hands/{side}/bend_bars",
                   rr.LineStrips2D(bar_lines_px, colors=bar_colors,
                                   radii=world_radius_to_pixels(0.030),
                                   draw_order=20.0))
            guide_lines_w = [np.array([[x, 1.55], [x, 1.88]]) for x in bar_x]
            guide_lines_px = [world_to_pixels(p) for p in guide_lines_w]
            rr.log(f"hands/{side}/bend_guides",
                   rr.LineStrips2D(guide_lines_px, colors=[80, 80, 90],
                                   radii=world_radius_to_pixels(0.008),
                                   draw_order=15.0))
            # (T/I/M/R/L initials are baked into the canvas above)

            # --- 4. per-finger scalar timeseries (incl. thumb) ---
            for f, f_N in zip(FINGER_NAMES, tip_force):
                rr.log(f"plots/force_N/{side}/{f}", rr.Scalars(f_N))
            for f, b in zip(FINGER_NAMES, bend.tolist()):
                rr.log(f"plots/bend/{side}/{f}", rr.Scalars(float(b)))

        if (i + 1) % 50 == 0:
            print(f"  logged frame {i + 1}/{len(df)}")

    # ---- high-rate OAK IMU ----
    # oak_imu.parquet uses a different clock (host monotonic, ~4428s) than
    # frames.parquet (Unix epoch, ~1.78e9). Remap it onto frame_time so it
    # shares the same timeline as everything else, using the same start/end
    # bracket as the frames.
    if not args.no_imu and (root / "oak_imu.parquet").exists():
        imu = pd.read_parquet(root / "oak_imu.parquet")
        print(f"Logging {len(imu)} OAK IMU samples...")
        imu_t = imu["timestamp"].to_numpy()
        imu_t0, imu_t1 = float(imu_t.min()), float(imu_t.max())
        # Affine-remap OAK IMU's host-monotonic clock to the same
        # seconds-from-start axis as frames.
        if imu_t1 > imu_t0:
            imu_rel = (imu_t - imu_t0) * (duration_s / (imu_t1 - imu_t0))
        else:
            imu_rel = np.zeros_like(imu_t)
        rr.send_columns(
            "plots/oak_imu/accel",
            indexes=[rr.TimeColumn("frame_time", duration=imu_rel)],
            columns=rr.Scalars.columns(scalars=imu[["accel_x", "accel_y", "accel_z"]].to_numpy()),
        )
        rr.send_columns(
            "plots/oak_imu/gyro",
            indexes=[rr.TimeColumn("frame_time", duration=imu_rel)],
            columns=rr.Scalars.columns(scalars=imu[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()),
        )

    print("Done. Use the rerun viewer to scrub the timeline.")
    print("Tip: when you've found your crop range, run:")
    print(f"  python {Path(__file__).parent}/crop.py --start <frame_idx> --end <frame_idx>")


if __name__ == "__main__":
    main()
