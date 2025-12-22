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
    win = int(win)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(a.astype(np.float32), kernel, mode="same")


def find_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find contiguous True segments in a boolean 1D array.
    Returns list of (start, end) with end exclusive.
    """
    segs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
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

    # gray is 0..255. We want a high threshold near whites.
    # Use 95th percentile, then clamp to a sane range.
    p95 = float(np.percentile(gray, 95))
    # Push it upward a bit; AI sheets often have near-white background.
    thr = int(min(252, max(230, round(p95) - 2)))
    return thr


def detect_content_bands(
    img_rgb: Image.Image,
    *,
    white_thr: Optional[int] = None,
    blur: float = 0.8,
    smooth_win: int = 9,
    empty_frac: float = 0.0025,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """
    Detect 'content bands' along X and Y.

    Returns:
      x_content_segments: list of (x0, x1) that contain tile columns
      y_content_segments: list of (y0, y1) that contain tile rows
      used_white_thr: the white threshold used
    """
    # Convert to grayscale array
    gray_img = img_rgb.convert("L")
    if blur and blur > 0:
        gray_img = gray_img.filter(ImageFilter.GaussianBlur(radius=blur))
    gray = np.array(gray_img, dtype=np.uint8)

    thr = pick_background_threshold(gray, white_thr)

    # "Ink" = pixels darker than thr
    ink = (gray < thr).astype(np.uint8)

    # Projection: how many ink pixels per column/row
    col_ink = ink.sum(axis=0)
    row_ink = ink.sum(axis=1)

    # Smooth projections to remove small noise / faint lines
    col_ink_s = smooth_1d(col_ink, smooth_win)
    row_ink_s = smooth_1d(row_ink, smooth_win)

    # Decide what counts as "empty"
    # Use fraction of full dimension so it scales with image size.
    # Example: if width=2000 and empty_frac=0.0025 -> ~5 pixels of ink is still "empty".
    col_empty_thr = max(1.0, float(gray.shape[0]) * empty_frac)
    row_empty_thr = max(1.0, float(gray.shape[1]) * empty_frac)

    col_has_content = col_ink_s > col_empty_thr
    row_has_content = row_ink_s > row_empty_thr

    x_content_segments = find_segments(col_has_content)
    y_content_segments = find_segments(row_has_content)

    return x_content_segments, y_content_segments, thr


def split_into_n_segments_by_whitespace(
    content_segments: List[Tuple[int, int]],
    n: int,
) -> List[Tuple[int, int]]:
    """
    Sometimes content segments come as one large segment (if gutters contain some ink).
    If we already have >= n segments, keep the largest n by width.
    If we have 1 segment but need many, we cannot split reliably here.
    """
    if len(content_segments) == n:
        return content_segments

    # If there are more than n, pick the n widest (then sort by position)
    if len(content_segments) > n:
        picked = sorted(content_segments, key=lambda s: (s[1] - s[0]), reverse=True)[:n]
        return sorted(picked, key=lambda s: s[0])

    # If fewer than n, return as-is; caller can fallback.
    return content_segments


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
) -> List[Tuple[int, int, int, int]]:
    """
    Infer grid tile bounding boxes from content bands.

    Strategy:
      - detect content segments along x and y
      - try to get exactly wanted_cols and wanted_rows
      - if detection fails (e.g., gutters not empty), fallback to even-splitting
        based on the "content bounding box" (non-empty area) and then split evenly.
    """
    x_segs, y_segs, used_thr = detect_content_bands(
        img_rgb,
        white_thr=white_thr,
        blur=blur,
        smooth_win=smooth_win,
        empty_frac=empty_frac,
    )

    # First attempt: content segments represent columns/rows
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

    # Fallback: compute the overall content bounding box and split evenly
    # This handles cases where gutters have ink/noise and segments collapse.
    gray = np.array(img_rgb.convert("L"), dtype=np.uint8)
    thr = pick_background_threshold(gray, white_thr)
    ink = gray < thr

    ys, xs = np.where(ink)
    if len(xs) == 0 or len(ys) == 0:
        # totally blank image; fallback to whole image split
        bbox = (0, 0, img_rgb.width, img_rgb.height)
    else:
        x0 = int(xs.min()); x1 = int(xs.max()) + 1
        y0 = int(ys.min()); y1 = int(ys.max()) + 1
        # pad bbox a bit
        bbox = (
            max(0, x0 - 4),
            max(0, y0 - 4),
            min(img_rgb.width, x1 + 4),
            min(img_rgb.height, y1 + 4),
        )

    bx0, by0, bx1, by1 = bbox
    bw = bx1 - bx0
    bh = by1 - by0

    # Split the content bbox evenly into grid
    cell_w = bw / wanted_cols
    cell_h = bh / wanted_rows

    boxes = []
    for r in range(wanted_rows):
        for c in range(wanted_cols):
            x0 = int(round(bx0 + c * cell_w))
            x1 = int(round(bx0 + (c + 1) * cell_w))
            y0 = int(round(by0 + r * cell_h))
            y1 = int(round(by0 + (r + 1) * cell_h))
            boxes.append((
                max(0, x0),
                max(0, y0),
                min(img_rgb.width, x1),
                min(img_rgb.height, y1),
            ))
    return boxes


def normalize_tile_to_square(tile: Image.Image, out_size: int, bg=(255, 255, 255)) -> Image.Image:
    """
    Crop tight around content (optional), then letterbox to square, then resize.
    We keep it simple/robust:
      - just letterbox to square then resize.
    """
    tile = tile.convert("RGBA")

    # Convert to RGB against white background (avoid alpha weirdness)
    bg_img = Image.new("RGB", tile.size, bg)
    bg_img.paste(tile, mask=tile.split()[-1])
    tile_rgb = bg_img

    # Letterbox to square
    w, h = tile_rgb.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), bg)
    ox = (side - w) // 2
    oy = (side - h) // 2
    canvas.paste(tile_rgb, (ox, oy))
    return canvas.resize((out_size, out_size), Image.LANCZOS)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="image-data.tsv", help="Path to image-data.tsv")
    ap.add_argument("--sprite", required=True, help="Path to generated sprite PNG")
    ap.add_argument("--out", default="media/images", help="Output dir (default: media/images)")

    ap.add_argument("--tile_out", type=int, default=256, help="Output tile size (default: 256)")
    ap.add_argument("--cols", type=int, default=0, help="Grid columns (0=auto sqrt like prompt)")
    ap.add_argument("--rows", type=int, default=0, help="Grid rows (0=auto sqrt like prompt)")
    ap.add_argument("--count", type=int, default=0, help="How many tiles to export (0=use TSV count)")

    # detection knobs (usually you do NOT need to change these)
    ap.add_argument("--white_thr", type=int, default=None, help="Background threshold 0..255 (default=auto)")
    ap.add_argument("--blur", type=float, default=0.8, help="Gaussian blur radius for detection (default=0.8)")
    ap.add_argument("--smooth", type=int, default=9, help="Smoothing window for projections (default=9)")
    ap.add_argument("--empty_frac", type=float, default=0.0025, help="Empty threshold as fraction (default=0.0025)")
    ap.add_argument("--pad", type=int, default=2, help="Padding around detected cells (default=2)")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing tiles")
    ap.add_argument("--debug_boxes", type=str, default="", help="Write debug image with boxes to this path")
    args = ap.parse_args()

    tsv_path = Path(args.tsv)
    sprite_path = Path(args.sprite)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = parse_tsv(tsv_path.read_text(encoding="utf-8"))
    if not notes:
        raise SystemExit(f"No IDs found in TSV: {tsv_path}")

    n = args.count if args.count > 0 else len(notes)

    cols = args.cols
    rows = args.rows
    if cols <= 0 or rows <= 0:
        cols, rows = compute_grid(n)

    # Load sprite
    img = Image.open(sprite_path).convert("RGB")

    # Infer grid boxes
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

    # Export tiles in TSV order (left->right, top->bottom)
    saved = 0
    skipped = 0

    for i in range(min(n, len(notes), len(boxes))):
        note_id = notes[i]["id"]
        out_path = out_dir / f"{note_id}.png"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        x0, y0, x1, y1 = boxes[i]
        tile = img.crop((x0, y0, x1, y1))
        tile = normalize_tile_to_square(tile, args.tile_out)
        tile.save(out_path, "PNG")
        saved += 1

    print(f"Sprite: {sprite_path}")
    print(f"Detected grid: {cols} cols × {rows} rows (exporting {min(n, len(notes), len(boxes))} tiles)")
    print(f"Saved: {saved} | Skipped: {skipped} | Out: {out_dir}")

    # Optional: write debug image with boxes
    if args.debug_boxes:
        dbg = img.copy()
        from PIL import ImageDraw
        d = ImageDraw.Draw(dbg)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
            if i < n:
                d.text((x0 + 3, y0 + 3), str(i + 1), fill=(255, 0, 0))
        dbg_path = Path(args.debug_boxes)
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        dbg.save(dbg_path)
        print(f"Debug boxes image written: {dbg_path}")


if __name__ == "__main__":
    main()
