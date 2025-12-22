#!/usr/bin/env python3
import argparse
import math
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
# Transparency + dehalo
# ---------------------------
def rgba_with_transparent_bg(
    tile_rgb: Image.Image,
    bg_thr: int,
    soft: int = 12,
    dehalo: bool = True,
    shrink: int = 1,
) -> Image.Image:
    """
    Convert near-white background to transparency + optionally remove white edge halo.

    bg_thr: whiteness threshold (min(R,G,B) >= bg_thr => transparent)
    soft: ramp width for alpha (0 = hard cut)
    dehalo: remove white fringing by unblending against white using the computed alpha
    shrink: erode/shrink alpha by N pixels to remove last thin halo (0 = off)
    """
    rgb_img = tile_rgb.convert("RGB")
    arr = np.array(rgb_img, dtype=np.uint8)  # (H, W, 3)

    # whiteness in 0..255, robust for slightly tinted whites
    # IMPORTANT: reduce over channel axis (axis=2) to get (H, W)
    whiteness = np.min(arr, axis=2).astype(np.int16)

    soft = max(0, int(soft))
    if soft == 0:
        alpha = (whiteness < bg_thr).astype(np.uint8) * 255
    else:
        # alpha ramps from 255 (opaque) to 0 (transparent) near bg_thr
        a = (bg_thr - whiteness) * 255.0 / float(soft)
        alpha = np.clip(a, 0, 255).astype(np.uint8)

    # Optional: shrink alpha a bit to remove the last 1px white halo
    # (AI images are often anti-aliased against white)
    shrink = max(0, int(shrink))
    if shrink > 0:
        a_img = Image.fromarray(alpha, mode="L")
        for _ in range(shrink):
            a_img = a_img.filter(ImageFilter.MinFilter(3))  # 3x3 erosion
        alpha = np.array(a_img, dtype=np.uint8)

    if dehalo:
        # Unblend from WHITE background to remove white fringe on edges.
        # Work in float 0..1
        a = (alpha.astype(np.float32) / 255.0)  # (H, W)
        a3 = np.maximum(a, 1e-6)[..., None]     # (H, W, 1) avoid divide by zero

        rgb = arr.astype(np.float32) / 255.0    # (H, W, 3)
        white = 1.0

        # rgb = a*orig + (1-a)*white  => orig = (rgb - (1-a)*white)/a
        orig = (rgb - (1.0 - a)[..., None] * white) / a3
        orig = np.clip(orig, 0.0, 1.0)

        out_rgb = (orig * 255.0 + 0.5).astype(np.uint8)
    else:
        out_rgb = arr

    out = Image.fromarray(out_rgb, mode="RGB").convert("RGBA")
    out.putalpha(Image.fromarray(alpha, mode="L"))
    return out


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
        description="Slice a sprite sheet into per-note PNGs (content-aware grid detection + transparent background)."
    )

    ap.add_argument("--tsv", default="image-data.tsv",
                    help="Path to TSV containing note IDs in the first column. Lines starting with # are ignored. (default: image-data.tsv)")
    ap.add_argument("--sprite", required=True,
                    help="Path to the sprite sheet PNG to slice (required).")
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

    # transparency knobs
    ap.add_argument("--alpha_white_thr", type=int, default=0,
                    help="White threshold for transparency (0 = auto from sprite). Pixels with min(R,G,B) >= thr become transparent. (default: auto)")
    ap.add_argument("--alpha_soft", type=int, default=12,
                    help="Softness for background->alpha ramp in pixels. 0 = hard cut. (default: 12)")
    ap.add_argument("--alpha_shrink", type=int, default=1,
                    help="Shrink (erode) alpha mask by N pixels to remove edge halo. 0 = off. (default: 1)")
    ap.add_argument("--no_dehalo", action="store_true",
                    help="Disable dehalo (unblending) step. Use only for debugging.")

    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output PNGs.")
    ap.add_argument("--debug_boxes", default="",
                    help="If set, write a debug PNG showing detected boxes to this path.")

    args = ap.parse_args()

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

    # Choose transparency threshold
    if args.alpha_white_thr > 0:
        alpha_thr = int(args.alpha_white_thr)
    else:
        gray = np.array(img.convert("L"), dtype=np.uint8)
        p99 = float(np.percentile(gray, 99.0))
        alpha_thr = int(min(254, max(235, round(p99) - 1)))

    saved = skipped = 0
    limit = min(n, len(notes), len(boxes))

    for i in range(limit):
        note_id = notes[i]["id"]
        out_path = out_dir / f"{note_id}.png"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        tile_rgb = img.crop(boxes[i])

        # Make background transparent + dehalo + optional alpha shrink
        tile_rgba = rgba_with_transparent_bg(
            tile_rgb,
            bg_thr=alpha_thr,
            soft=args.alpha_soft,
            dehalo=not args.no_dehalo,
            shrink=args.alpha_shrink,
        )

        # Letterbox to square on transparent canvas + resize
        tile_out = letterbox_rgba_to_square(tile_rgba, args.tile_out)

        tile_out.save(out_path, "PNG")
        saved += 1

    print(f"Sprite: {args.sprite}")
    print(f"Detected grid: {cols} cols × {rows} rows (exporting {limit} tiles)")
    print(f"Alpha threshold used: {alpha_thr} (min(R,G,B) >= thr => transparent)")
    print(f"Saved: {saved} | Skipped: {skipped} | Out: {out_dir}")

    # Optional: write debug image with boxes (kept as RGB; transparency not relevant)
    if args.debug_boxes:
        dbg = img.copy()
        from PIL import ImageDraw
        d = ImageDraw.Draw(dbg)
        for j, (x0, y0, x1, y1) in enumerate(boxes):
            d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
            if j < limit:
                d.text((x0 + 3, y0 + 3), str(j + 1), fill=(255, 0, 0))
        dbg_path = Path(args.debug_boxes)
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        dbg.save(dbg_path)
        print(f"Debug boxes image written: {dbg_path}")


if __name__ == "__main__":
    main()
