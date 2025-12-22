#!/usr/bin/env python3
"""
slice_sprite.py

Slice a sprite sheet into per-note PNGs.

Features:
- Content-aware grid detection (no need for perfect margins/borders).
- Transparent background removal that DOES NOT punch holes in subjects:
  ✅ Only removes background that is CONNECTED to the tile edge (flood fill).
- Dehalo/unblend to remove white fringe.
- Auto-tunes background thresholds per sprite (default).

Dependencies:
- Pillow
- numpy
"""

import argparse
import math
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from PIL import Image, ImageFilter
import numpy as np


# ---------------------------
# TSV
# ---------------------------
def parse_tsv(tsv_text: str) -> List[Dict[str, str]]:
    lines = [l.rstrip("\n") for l in tsv_text.splitlines()]
    lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    rows = []
    for line in lines:
        cols = line.split("\t")
        _id = cols[0] if len(cols) > 0 else ""
        if _id.strip():
            rows.append({"id": _id.strip()})
    return rows


def compute_grid(n: int) -> Tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


# ---------------------------
# Content-aware grid detection
# ---------------------------
def smooth_1d(a: np.ndarray, win: int) -> np.ndarray:
    """Simple moving average smoothing for 1D arrays."""
    if win <= 1:
        return a
    kernel = np.ones(int(win), dtype=np.float32) / float(win)
    return np.convolve(a.astype(np.float32), kernel, mode="same")


def find_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True segments in a boolean 1D array."""
    segs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    return segs


def pick_background_threshold(gray: np.ndarray, user_white: Optional[int]) -> int:
    """
    Choose a 'white threshold' so pixels >= threshold are treated as background.
    If user provided, use it. Otherwise choose a robust percentile-based threshold.
    """
    if user_white is not None:
        return int(user_white)
    p95 = float(np.percentile(gray, 95))
    return int(min(252, max(230, round(p95) - 2)))


def detect_content_bands(
    img_rgb: Image.Image,
    *,
    white_thr: Optional[int] = None,
    blur: float = 0.8,
    smooth_win: int = 9,
    empty_frac: float = 0.0025,
):
    """
    Detect 'content bands' along X and Y.

    Returns:
      x_content_segments: list of (x0, x1) that contain tile columns
      y_content_segments: list of (y0, y1) that contain tile rows
      used_white_thr: the white threshold used
    """
    gray_img = img_rgb.convert("L")
    if blur > 0:
        gray_img = gray_img.filter(ImageFilter.GaussianBlur(radius=blur))
    gray = np.array(gray_img, dtype=np.uint8)

    thr = pick_background_threshold(gray, white_thr)
    ink = (gray < thr).astype(np.uint8)

    col_ink = smooth_1d(ink.sum(axis=0), smooth_win)
    row_ink = smooth_1d(ink.sum(axis=1), smooth_win)

    col_empty_thr = max(1.0, gray.shape[0] * empty_frac)
    row_empty_thr = max(1.0, gray.shape[1] * empty_frac)

    return (
        find_segments(col_ink > col_empty_thr),
        find_segments(row_ink > row_empty_thr),
        thr,
    )


def split_into_n_segments_by_whitespace(segs, n):
    """
    Sometimes content segments come as one large segment (if gutters contain some ink).
    If we already have >= n segments, keep the largest n by width.
    If we have fewer than n, return as-is; caller can fallback.
    """
    if len(segs) == n:
        return segs
    if len(segs) > n:
        picked = sorted(segs, key=lambda s: s[1] - s[0], reverse=True)[:n]
        return sorted(picked, key=lambda s: s[0])
    return segs


def infer_grid_boxes(
    img_rgb: Image.Image,
    wanted_cols: int,
    wanted_rows: int,
    *,
    white_thr: Optional[int] = None,
    blur: float = 0.8,
    smooth_win: int = 9,
    empty_frac: float = 0.0025,
    pad: int = 2,
):
    """
    Infer grid tile bounding boxes from content bands.

    Strategy:
      - detect content segments along x and y
      - try to get exactly wanted_cols and wanted_rows
      - if detection fails, fallback to even-splitting across the whole image
    """
    x_segs, y_segs, _ = detect_content_bands(
        img_rgb,
        white_thr=white_thr,
        blur=blur,
        smooth_win=smooth_win,
        empty_frac=empty_frac,
    )

    x_cols = split_into_n_segments_by_whitespace(x_segs, wanted_cols)
    y_rows = split_into_n_segments_by_whitespace(y_segs, wanted_rows)

    # If we got exactly the right number, build boxes directly
    if len(x_cols) == wanted_cols and len(y_rows) == wanted_rows:
        boxes = []
        for r in range(wanted_rows):
            for c in range(wanted_cols):
                x0, x1 = x_cols[c]
                y0, y1 = y_rows[r]
                # small padding inward/outward to counter faint borders
                boxes.append((
                    max(0, x0 - pad),
                    max(0, y0 - pad),
                    min(img_rgb.width, x1 + pad),
                    min(img_rgb.height, y1 + pad),
                ))
        return boxes

    # fallback: even split
    w, h = img_rgb.size
    cell_w = w / wanted_cols
    cell_h = h / wanted_rows

    boxes = []
    for r in range(wanted_rows):
        for c in range(wanted_cols):
            boxes.append((
                int(round(c * cell_w)),
                int(round(r * cell_h)),
                int(round((c + 1) * cell_w)),
                int(round((r + 1) * cell_h)),
            ))
    return boxes


# ---------------------------
# Transparency + dehalo (edge-connected background)
# ---------------------------
def _estimate_bg_color_and_scale(arr_u8: np.ndarray, frame: int = 10) -> Tuple[np.ndarray, float]:
    """
    Estimate background RGB from the border pixels of the tile.
    Returns (bg_rgb_float32[3], mad_float).
    """
    h, w, _ = arr_u8.shape
    f = max(1, int(frame))
    # Border ring (top, bottom, left, right)
    top = arr_u8[:f, :, :]
    bot = arr_u8[h - f:h, :, :]
    lef = arr_u8[:, :f, :]
    rig = arr_u8[:, w - f:w, :]
    border = np.concatenate([top.reshape(-1, 3), bot.reshape(-1, 3), lef.reshape(-1, 3), rig.reshape(-1, 3)], axis=0)

    bg = np.median(border.astype(np.float32), axis=0)  # robust background estimate
    # Median absolute deviation over per-pixel RGB distances
    dist = np.sqrt(np.sum((border.astype(np.float32) - bg[None, :]) ** 2, axis=1))
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))  # robust noise estimate
    return bg, mad


def _floodfill_edge_connected_bg(dist: np.ndarray, bg_cut: float) -> np.ndarray:
    """
    Flood-fill background starting from the tile edge:
    A pixel is considered "background-like" if dist <= bg_cut.
    Returns bg_conn mask (H, W) True for background connected to the border.
    """
    h, w = dist.shape
    bg_like = dist <= bg_cut
    bg_conn = np.zeros((h, w), dtype=np.bool_)

    q = deque()

    def push(y, x):
        if 0 <= y < h and 0 <= x < w and (not bg_conn[y, x]) and bg_like[y, x]:
            bg_conn[y, x] = True
            q.append((y, x))

    # Seed with border pixels
    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    # 4-neighborhood flood fill (safer against thin gaps than 8-neighborhood)
    while q:
        y, x = q.popleft()
        push(y - 1, x)
        push(y + 1, x)
        push(y, x - 1)
        push(y, x + 1)

    return bg_conn


def rgba_with_transparent_bg_edge_connected(
    tile_rgb: Image.Image,
    *,
    frame: int = 10,
    k_bg: float = 3.0,
    dehalo: bool = True,
    shrink: int = 1,
    feather: int = 1,
) -> Tuple[Image.Image, Dict[str, float]]:
    """
    Convert background to transparency WITHOUT punching holes:

    Key idea:
      ✅ Only pixels that are background-like AND CONNECTED TO TILE EDGE become transparent.
      => Whites inside the subject stay opaque.

    Steps:
    - Estimate background color (from border).
    - Compute per-pixel RGB distance to background.
    - Flood fill from the border through "background-like" pixels.
    - Alpha = 0 for edge-connected background, 255 elsewhere.
    - Optional feather + shrink + dehalo.

    Returns (tile_rgba, debug_info).
    """
    rgb_img = tile_rgb.convert("RGB")
    arr = np.array(rgb_img, dtype=np.uint8)  # (H, W, 3)
    h, w, _ = arr.shape

    bg_rgb, mad = _estimate_bg_color_and_scale(arr, frame=frame)
    sigma = max(1.0, mad * 1.4826)  # convert MAD to approx stddev
    bg_cut = max(4.0, float(k_bg) * sigma)

    arr_f = arr.astype(np.float32)
    dist = np.sqrt(np.sum((arr_f - bg_rgb[None, None, :]) ** 2, axis=2)).astype(np.float32)

    bg_conn = _floodfill_edge_connected_bg(dist, bg_cut)

    # Alpha: fully transparent only for edge-connected background
    alpha_u8 = np.full((h, w), 255, dtype=np.uint8)
    alpha_u8[bg_conn] = 0

    # Optional: feather alpha edges (helps jaggies)
    feather = max(0, int(feather))
    if feather > 0:
        a_img = Image.fromarray(alpha_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=feather))
        alpha_u8 = np.array(a_img, dtype=np.uint8)

    # Optional: shrink alpha a bit to remove last thin halo
    shrink = max(0, int(shrink))
    if shrink > 0:
        a_img = Image.fromarray(alpha_u8, mode="L")
        for _ in range(shrink):
            a_img = a_img.filter(ImageFilter.MinFilter(3))  # 3x3 erosion
        alpha_u8 = np.array(a_img, dtype=np.uint8)

    # Dehalo: unblend against estimated background color
    if dehalo:
        a = (alpha_u8.astype(np.float32) / 255.0)  # (H, W)
        a3 = np.maximum(a, 1e-6)[..., None]         # (H, W, 1)
        rgb = arr_f / 255.0                         # (H, W, 3)
        bg = (bg_rgb / 255.0)[None, None, :]        # (1, 1, 3)

        # rgb = a*orig + (1-a)*bg  => orig = (rgb - (1-a)*bg)/a
        orig = (rgb - (1.0 - a)[..., None] * bg) / a3
        orig = np.clip(orig, 0.0, 1.0)
        out_rgb = (orig * 255.0 + 0.5).astype(np.uint8)
    else:
        out_rgb = arr

    out = Image.fromarray(out_rgb, mode="RGB").convert("RGBA")
    out.putalpha(Image.fromarray(alpha_u8, mode="L"))

    dbg = {
        "bg_r": float(bg_rgb[0]),
        "bg_g": float(bg_rgb[1]),
        "bg_b": float(bg_rgb[2]),
        "sigma": float(sigma),
        "bg_cut": float(bg_cut),
        "k_bg": float(k_bg),
        "frame": float(frame),
        "feather": float(feather),
        "shrink": float(shrink),
    }
    return out, dbg


def auto_tune_alpha_params_for_tile(
    tile_rgb: Image.Image,
    *,
    frame: int,
    k_bg_candidates: List[float],
    shrink_candidates: List[int],
    feather: int,
    dehalo: bool,
) -> Tuple[Image.Image, Dict[str, float]]:
    """
    Auto-tune parameters for one tile.

    Heuristic:
    - Prefer settings that remove MORE edge background (more transparency),
      but keep a reasonable amount of foreground (alpha > 0).
    - Because we only remove edge-connected background, "holes inside subject"
      are already prevented. This tuning is mainly about: remove as much bg as possible
      without eating into the subject edge.

    Returns best (tile_rgba, debug_info).
    """
    best = None
    best_score = None

    # Evaluate each candidate combination
    for k_bg in k_bg_candidates:
        for shrink in shrink_candidates:
            rgba, dbg = rgba_with_transparent_bg_edge_connected(
                tile_rgb,
                frame=frame,
                k_bg=k_bg,
                dehalo=dehalo,
                shrink=shrink,
                feather=feather,
            )
            a = np.array(rgba.split()[-1], dtype=np.uint8)
            frac_transparent = float((a <= 8).mean())
            frac_opaque = float((a >= 200).mean())

            # Score:
            # - reward transparency (background removal)
            # - penalize if too little opaque area (subject got eaten)
            # Note: you may tune these constants; they work well for “character on white bg”.
            if frac_opaque < 0.12:
                score = -999.0  # reject: almost everything got transparent
            else:
                score = (frac_transparent * 2.0) + (frac_opaque * 0.5)

            if best_score is None or score > best_score:
                best_score = score
                dbg = dict(dbg)
                dbg.update({
                    "frac_transparent": frac_transparent,
                    "frac_opaque": frac_opaque,
                    "score": float(score),
                })
                best = (rgba, dbg)

    # Fallback (shouldn’t happen)
    if best is None:
        rgba, dbg = rgba_with_transparent_bg_edge_connected(
            tile_rgb,
            frame=frame,
            k_bg=3.0,
            dehalo=dehalo,
            shrink=1,
            feather=feather,
        )
        dbg = dict(dbg)
        dbg.update({"frac_transparent": 0.0, "frac_opaque": 1.0, "score": -1.0})
        return rgba, dbg

    return best


def letterbox_rgba_to_square(tile_rgba: Image.Image, out_size: int) -> Image.Image:
    """
    Center tile on a transparent square canvas, then resize to out_size×out_size.
    """
    tile_rgba = tile_rgba.convert("RGBA")
    w, h = tile_rgba.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(tile_rgba, ((side - w) // 2, (side - h) // 2), tile_rgba)
    return canvas.resize((out_size, out_size), Image.LANCZOS)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Slice a sprite sheet into per-note PNGs (content-aware grid detection + edge-connected transparent background removal)."
    )

    ap.add_argument("--tsv", default="image-data.tsv",
                    help="Path to TSV containing note IDs in the first column. Lines starting with # are ignored. (default: image-data.tsv)")
    ap.add_argument("--sprite", required=True,
                    help="Path to the sprite sheet PNG to slice. (required)")
    ap.add_argument("--out", default="media/images",
                    help="Output directory for extracted tiles. (default: media/images)")
    ap.add_argument("--tile_out", type=int, default=256,
                    help="Final output tile size in pixels (square). (default: 256)")

    ap.add_argument("--cols", type=int, default=0,
                    help="Grid columns. 0 = auto (ceil(sqrt(N))). (default: 0)")
    ap.add_argument("--rows", type=int, default=0,
                    help="Grid rows. 0 = auto (ceil(N/cols)). (default: 0)")
    ap.add_argument("--count", type=int, default=0,
                    help="How many tiles to export. 0 = number of TSV rows. (default: 0)")

    # detection knobs (usually you do NOT need to change these)
    ap.add_argument("--white_thr", type=int, default=None,
                    help="Background threshold for grid detection (0..255). Higher = treat more as background. None = auto. (default: auto)")
    ap.add_argument("--blur", type=float, default=0.8,
                    help="Gaussian blur radius used for detection. (default: 0.8)")
    ap.add_argument("--smooth", type=int, default=9,
                    help="Smoothing window for projection profiles used in detection. (default: 9)")
    ap.add_argument("--empty_frac", type=float, default=0.0025,
                    help="Empty threshold as fraction of image dimension. Lower = more sensitive. (default: 0.0025)")
    ap.add_argument("--pad", type=int, default=2,
                    help="Padding in pixels around detected cell boxes. (default: 2)")

    # transparency knobs (auto-tuned by default)
    ap.add_argument("--alpha_auto", action="store_true",
                    help="Auto-tune transparency parameters per tile. (default: enabled)")
    ap.add_argument("--no_alpha_auto", action="store_true",
                    help="Disable auto-tuning; use fixed values.")
    ap.add_argument("--alpha_frame", type=int, default=10,
                    help="How many pixels from the tile edge to sample for background estimation. (default: 10)")
    ap.add_argument("--alpha_k_bg", type=float, default=3.0,
                    help="Fixed: background distance multiplier (higher = more conservative). Used when auto-tuning is off. (default: 3.0)")
    ap.add_argument("--alpha_k_bg_candidates", type=str, default="2.4,2.8,3.2,3.6,4.0",
                    help="Auto: comma-separated k_bg candidates to try. (default: 2.4,2.8,3.2,3.6,4.0)")
    ap.add_argument("--alpha_shrink", type=int, default=1,
                    help="Fixed: shrink (erode) alpha mask by N pixels to remove edge halo. Used when auto-tuning is off. (default: 1)")
    ap.add_argument("--alpha_shrink_candidates", type=str, default="0,1,2",
                    help="Auto: comma-separated shrink candidates to try. (default: 0,1,2)")
    ap.add_argument("--alpha_feather", type=int, default=1,
                    help="Feather (Gaussian blur) for alpha edges in px. 0 = off. (default: 1)")
    ap.add_argument("--no_dehalo", action="store_true",
                    help="Disable dehalo (unblending) step. Use only for debugging.")

    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output PNGs.")
    ap.add_argument("--debug_boxes", default="",
                    help="If set, write a debug PNG showing detected boxes to this path.")
    ap.add_argument("--debug_alpha", default="",
                    help="If set, write a debug PNG for ONE tile alpha mask (first exported tile). Useful for tuning.")

    args = ap.parse_args()

    # alpha_auto default: ON unless explicitly disabled
    alpha_auto = True
    if args.no_alpha_auto:
        alpha_auto = False

    # Parse candidates
    try:
        k_bg_candidates = [float(x.strip()) for x in args.alpha_k_bg_candidates.split(",") if x.strip()]
    except Exception:
        raise SystemExit("Invalid --alpha_k_bg_candidates. Example: 2.4,2.8,3.2")

    try:
        shrink_candidates = [int(x.strip()) for x in args.alpha_shrink_candidates.split(",") if x.strip()]
    except Exception:
        raise SystemExit("Invalid --alpha_shrink_candidates. Example: 0,1,2")

    tsv_path = Path(args.tsv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = parse_tsv(tsv_path.read_text(encoding="utf-8"))
    if not notes:
        raise SystemExit(f"No IDs found in TSV: {tsv_path}")

    n = args.count if args.count > 0 else len(notes)

    cols, rows = args.cols, args.rows
    if cols <= 0 or rows <= 0:
        cols, rows = compute_grid(n)

    # Load sprite as RGB for detection
    img = Image.open(args.sprite).convert("RGB")

    boxes = infer_grid_boxes(
        img,
        wanted_cols=cols,
        wanted_rows=rows,
        white_thr=args.white_thr,
        blur=args.blur,
        smooth_win=args.smooth,
        empty_frac=args.empty_frac,
        pad=args.pad,
    )

    saved = skipped = 0
    limit = min(n, len(notes), len(boxes))

    first_tile_debug_written = False

    for i in range(limit):
        note_id = notes[i]["id"]
        out_path = out_dir / f"{note_id}.png"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        tile_rgb = img.crop(boxes[i])

        # Make background transparent (edge-connected), auto-tuned by default
        if alpha_auto:
            tile_rgba, dbg = auto_tune_alpha_params_for_tile(
                tile_rgb,
                frame=args.alpha_frame,
                k_bg_candidates=k_bg_candidates,
                shrink_candidates=shrink_candidates,
                feather=args.alpha_feather,
                dehalo=not args.no_dehalo,
            )
        else:
            tile_rgba, dbg = rgba_with_transparent_bg_edge_connected(
                tile_rgb,
                frame=args.alpha_frame,
                k_bg=args.alpha_k_bg,
                dehalo=not args.no_dehalo,
                shrink=args.alpha_shrink,
                feather=args.alpha_feather,
            )

        # Optional: write alpha debug for first exported tile
        if args.debug_alpha and not first_tile_debug_written:
            a = tile_rgba.split()[-1]
            dbg_path = Path(args.debug_alpha)
            dbg_path.parent.mkdir(parents=True, exist_ok=True)
            a.save(dbg_path)
            print(f"Debug alpha mask written: {dbg_path}")
            print(f"Alpha debug info (tile 1): {dbg}")
            first_tile_debug_written = True

        # Letterbox to square on transparent canvas + resize
        tile_out = letterbox_rgba_to_square(tile_rgba, args.tile_out)
        tile_out.save(out_path, "PNG")
        saved += 1

    print(f"Sprite: {args.sprite}")
    print(f"Detected grid: {cols} cols × {rows} rows (exporting {limit} tiles)")
    print(f"Transparency: edge-connected flood-fill | auto_tune={'ON' if alpha_auto else 'OFF'} | dehalo={'OFF' if args.no_dehalo else 'ON'}")
    print(f"Saved: {saved} | Skipped: {skipped} | Out: {out_dir}")

    # Optional: write debug image with boxes (kept as RGB; transparency not relevant)
    if args.debug_boxes:
        dbg_img = img.copy()
        from PIL import ImageDraw
        d = ImageDraw.Draw(dbg_img)
        for j, (x0, y0, x1, y1) in enumerate(boxes):
            d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
            if j < limit:
                d.text((x0 + 3, y0 + 3), str(j + 1), fill=(255, 0, 0))
        dbg_path = Path(args.debug_boxes)
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        dbg_img.save(dbg_path)
        print(f"Debug boxes image written: {dbg_path}")


if __name__ == "__main__":
    main()
