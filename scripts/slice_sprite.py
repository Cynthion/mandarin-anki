#!/usr/bin/env python3
"""
slice_sprite.py

Slice a sprite sheet into per-note PNGs.

GOAL (robust / no manual tuning):
1) Identify subjects on a near-white background (no grid assumptions).
2) Infer their row/column positions from their centers (top-left → bottom-right).
3) Re-center each subject into a new 256×256 WHITE tile.
4) Make ONLY the NEW tile’s edge-connected background transparent:
   ✅ removes outer background only (flood fill from tile edge)
   ✅ does NOT remove white/near-white inside the subject
   ✅ ALSO removes soft edge-connected shadows (like the pear shadow)

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
    """
    Parse TSV and return rows with at least an 'id' in column 1.
    - Ignores empty lines and comment lines (#...)
    - Also skips a header row if it looks like one ("id" or "ID").
    """
    lines = [l.rstrip("\n") for l in tsv_text.splitlines()]
    lines = [l for l in lines if l.strip() and not l.strip().lstrip().startswith("#")]

    rows: List[Dict[str, str]] = []
    for line in lines:
        cols = line.split("\t")
        _id = (cols[0] if len(cols) > 0 else "").strip()
        if not _id:
            continue
        if _id.lower() == "id":  # header
            continue
        rows.append({"id": _id})
    return rows


# ---------------------------
# Background estimation + flood fill (edge-connected)
# ---------------------------
def _estimate_bg_color_and_border_dist(arr_u8: np.ndarray, frame: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate background RGB from the border pixels of the image.
    Returns (bg_rgb_float32[3], border_distances_float32[N]).
    """
    h, w, _ = arr_u8.shape
    f = max(1, int(frame))

    top = arr_u8[:f, :, :]
    bot = arr_u8[h - f:h, :, :]
    lef = arr_u8[:, :f, :]
    rig = arr_u8[:, w - f:w, :]

    border = np.concatenate(
        [
            top.reshape(-1, 3),
            bot.reshape(-1, 3),
            lef.reshape(-1, 3),
            rig.reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)

    bg = np.median(border, axis=0)
    dist = np.sqrt(np.sum((border - bg[None, :]) ** 2, axis=1)).astype(np.float32)
    return bg.astype(np.float32), dist


def _floodfill_edge_connected_bg(dist_img: np.ndarray, bg_cut: float) -> np.ndarray:
    """
    Flood-fill background starting from the image edge:
    A pixel is "background-like" if dist <= bg_cut.
    Returns bg_conn mask (H, W) True for background connected to the border.
    """
    h, w = dist_img.shape
    bg_like = dist_img <= bg_cut
    bg_conn = np.zeros((h, w), dtype=np.bool_)

    q = deque()

    def push(y: int, x: int):
        if 0 <= y < h and 0 <= x < w and (not bg_conn[y, x]) and bg_like[y, x]:
            bg_conn[y, x] = True
            q.append((y, x))

    # seed with border pixels
    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    # 4-neighborhood flood fill
    while q:
        y, x = q.popleft()
        push(y - 1, x)
        push(y + 1, x)
        push(y, x - 1)
        push(y, x + 1)

    return bg_conn


def build_foreground_mask_autocut(img_rgb: Image.Image, *, frame: int = 12) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a foreground mask for a near-white background WITHOUT user parameters.

    Strategy:
    - Estimate bg color from border
    - Compute per-pixel RGB distance to bg
    - Choose bg_cut automatically from border distance distribution:
      bg_cut = percentile(border_dist, 99.5) + small cushion
    - Flood-fill bg-like pixels connected to outer edge
    - Foreground = NOT(edge-connected background)
    """
    arr = np.array(img_rgb.convert("RGB"), dtype=np.uint8)
    bg_rgb, border_dist = _estimate_bg_color_and_border_dist(arr, frame=frame)

    p995 = float(np.percentile(border_dist, 99.5))
    p99 = float(np.percentile(border_dist, 99.0))
    cushion = max(2.0, 0.25 * (p995 - p99 + 1.0))
    bg_cut = p995 + cushion

    arr_f = arr.astype(np.float32)
    dist_img = np.sqrt(np.sum((arr_f - bg_rgb[None, None, :]) ** 2, axis=2)).astype(np.float32)

    bg_conn = _floodfill_edge_connected_bg(dist_img, bg_cut)
    fg = ~bg_conn

    dbg = {
        "bg_r": float(bg_rgb[0]),
        "bg_g": float(bg_rgb[1]),
        "bg_b": float(bg_rgb[2]),
        "bg_cut": float(bg_cut),
        "border_p99": float(p99),
        "border_p995": float(p995),
        "frame": float(frame),
    }
    return fg, dbg


# ---------------------------
# Simple morphology for mask cleanup
# ---------------------------
def _morph_close(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    """Morphological closing (dilate then erode) with a square kernel derived from radius."""
    r = int(radius)
    if r <= 0:
        return mask_bool
    k = max(3, r * 2 + 1)  # odd
    m = (mask_bool.astype(np.uint8) * 255)
    im = Image.fromarray(m, mode="L")
    im = im.filter(ImageFilter.MaxFilter(size=k))
    im = im.filter(ImageFilter.MinFilter(size=k))
    return (np.array(im, dtype=np.uint8) >= 128)


def _morph_open(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    """Morphological opening (erode then dilate) to remove pepper noise."""
    r = int(radius)
    if r <= 0:
        return mask_bool
    k = max(3, r * 2 + 1)
    m = (mask_bool.astype(np.uint8) * 255)
    im = Image.fromarray(m, mode="L")
    im = im.filter(ImageFilter.MinFilter(size=k))
    im = im.filter(ImageFilter.MaxFilter(size=k))
    return (np.array(im, dtype=np.uint8) >= 128)


# ---------------------------
# Connected components (union-find, 8-connected)
# ---------------------------
class _UF:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def make(self, x: int):
        self.parent[x] = x

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def connected_components_bbox(mask: np.ndarray) -> List[Dict[str, int]]:
    """
    Find connected components (8-connected) in a boolean mask.
    Returns list: {x0,y0,x1,y1,area,cx,cy}
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)

    uf = _UF()
    next_label = 1

    # first pass
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue

            neighbors = []
            if x > 0 and labels[y, x - 1] > 0:
                neighbors.append(labels[y, x - 1])
            if y > 0:
                if x > 0 and labels[y - 1, x - 1] > 0:
                    neighbors.append(labels[y - 1, x - 1])
                if labels[y - 1, x] > 0:
                    neighbors.append(labels[y - 1, x])
                if x + 1 < w and labels[y - 1, x + 1] > 0:
                    neighbors.append(labels[y - 1, x + 1])

            if not neighbors:
                lbl = next_label
                next_label += 1
                uf.make(lbl)
                labels[y, x] = lbl
            else:
                lbl = min(neighbors)
                labels[y, x] = lbl
                for n in neighbors:
                    if n != lbl:
                        uf.union(lbl, n)

    if next_label == 1:
        return []

    # second pass + stats
    stats: Dict[int, List[int]] = {}
    for y in range(h):
        for x in range(w):
            lbl = labels[y, x]
            if lbl == 0:
                continue
            root = uf.find(lbl)
            labels[y, x] = root
            if root not in stats:
                stats[root] = [x, y, x + 1, y + 1, 1, x, y]
            else:
                s = stats[root]
                s[0] = min(s[0], x)
                s[1] = min(s[1], y)
                s[2] = max(s[2], x + 1)
                s[3] = max(s[3], y + 1)
                s[4] += 1
                s[5] += x
                s[6] += y

    comps: List[Dict[str, int]] = []
    for _, s in stats.items():
        x0, y0, x1, y1, area, sumx, sumy = s
        cx = int(round(sumx / max(1, area)))
        cy = int(round(sumy / max(1, area)))
        comps.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "area": area, "cx": cx, "cy": cy})
    return comps


# ---------------------------
# Subject detection (no params; tries to hit exactly N)
# ---------------------------
def detect_subjects(sprite_rgb: Image.Image, want_n: int) -> Tuple[List[Dict[str, int]], Dict[str, float]]:
    """
    Auto-detect subjects on a near-white sprite. No user parameters needed.
    """
    fg0, dbg0 = build_foreground_mask_autocut(sprite_rgb, frame=14)

    open_radii = [0, 1, 2]
    close_radii = [0, 2, 4, 6, 8, 10, 12, 14]

    best: Optional[List[Dict[str, int]]] = None
    best_dbg: Optional[Dict[str, float]] = None
    best_score: Optional[float] = None

    for o in open_radii:
        fg_open = _morph_open(fg0, radius=o)

        for c in close_radii:
            fg = _morph_close(fg_open, radius=c)

            comps = connected_components_bbox(fg)
            if not comps:
                continue

            comps = sorted(comps, key=lambda z: z["area"], reverse=True)

            top_k = comps[: min(len(comps), max(12, want_n))]
            med_area = float(np.median([t["area"] for t in top_k])) if top_k else 0.0
            min_area = max(30, int(med_area * 0.08)) if med_area > 0 else 30
            comps2 = [cc for cc in comps if cc["area"] >= min_area]

            count = len(comps2)
            diff = abs(count - want_n)

            miss_penalty = 200.0 if count < want_n else 0.0
            score = -(diff * 10.0) - miss_penalty - (o * 1.5) - (c * 0.2)

            if best_score is None or score > best_score:
                best_score = score
                best = comps2
                best_dbg = dict(dbg0)
                best_dbg.update(
                    {
                        "open_radius": float(o),
                        "close_radius": float(c),
                        "min_area": float(min_area),
                        "found_components": float(count),
                        "want_n": float(want_n),
                        "score": float(score),
                    }
                )

            if count == want_n:
                return comps2, best_dbg

    if best is None or best_dbg is None:
        raise SystemExit("No subjects detected. Is the sprite background near-white?")

    if len(best) > want_n:
        best = sorted(best, key=lambda z: z["area"], reverse=True)[:want_n]

    if len(best) != want_n:
        raise SystemExit(
            f"Subject detection failed: expected {want_n} subjects, got {len(best)}.\n"
            f"Detection debug: {best_dbg}\n"
            f"Likely causes: subjects overlap/touch, or background not near-white everywhere."
        )

    return best, best_dbg


# ---------------------------
# Infer grid size from subject centers (auto)
# ---------------------------
def infer_grid_from_centers(centers: np.ndarray, want_n: int, img_w: int, img_h: int) -> Tuple[int, int]:
    """Infer (cols, rows) from subject centers."""
    aspect = float(img_w) / float(max(1, img_h))
    est_cols = int(round(math.sqrt(want_n * aspect)))
    est_cols = max(1, min(want_n, est_cols))

    candidates = []
    for dc in range(-3, 4):
        c = max(1, min(want_n, est_cols + dc))
        r = int(math.ceil(want_n / c))
        candidates.append((c, r))
    candidates = sorted(set(candidates))

    xs = centers[:, 0].astype(np.float32)
    ys = centers[:, 1].astype(np.float32)

    def kmeans_1d(values: np.ndarray, k: int, iters: int = 30) -> Tuple[np.ndarray, np.ndarray, float]:
        v = values.astype(np.float32)
        if k <= 1:
            center = np.array([float(np.mean(v))], dtype=np.float32)
            labels = np.zeros((len(v),), dtype=np.int32)
            sse = float(np.sum((v - center[0]) ** 2))
            return center, labels, sse

        qs = np.linspace(0.0, 1.0, k)
        centers_ = np.quantile(v, qs).astype(np.float32)

        for _ in range(iters):
            d = np.abs(v[:, None] - centers_[None, :])
            labels = np.argmin(d, axis=1).astype(np.int32)
            new_centers = centers_.copy()
            for i in range(k):
                m = v[labels == i]
                if len(m) > 0:
                    new_centers[i] = float(np.mean(m))
            if np.allclose(new_centers, centers_):
                centers_ = new_centers
                break
            centers_ = new_centers

        sse = 0.0
        for i in range(k):
            m = v[labels == i]
            if len(m) > 0:
                sse += float(np.sum((m - centers_[i]) ** 2))
        return centers_, labels, float(sse)

    best = None
    best_score = None

    for cols, rows in candidates:
        _, lab_y, sse_y = kmeans_1d(ys, rows)
        _, lab_x, sse_x = kmeans_1d(xs, cols)

        occ_y = np.bincount(lab_y, minlength=rows)
        occ_x = np.bincount(lab_x, minlength=cols)
        empty_penalty = float((occ_y == 0).sum() + (occ_x == 0).sum()) * 1e9

        score = (sse_y + sse_x) + empty_penalty
        if best_score is None or score < best_score:
            best_score = score
            best = (cols, rows)

    assert best is not None
    return best


def assign_order_top_left(comps: List[Dict[str, int]], want_n: int, img_w: int, img_h: int) -> Tuple[List[Dict[str, int]], Dict[str, float]]:
    """Assign each component to (row, col), then sort row-major for TSV order."""
    comps = list(comps)
    if len(comps) != want_n:
        raise SystemExit(f"Internal error: expected {want_n} comps, got {len(comps)}")

    centers = np.array([[c["cx"], c["cy"]] for c in comps], dtype=np.float32)
    cols, rows = infer_grid_from_centers(centers, want_n, img_w, img_h)

    xs = centers[:, 0].astype(np.float32)
    ys = centers[:, 1].astype(np.float32)

    def kmeans_1d(values: np.ndarray, k: int, iters: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        v = values.astype(np.float32)
        if k <= 1:
            centers_ = np.array([float(np.mean(v))], dtype=np.float32)
            labels = np.zeros((len(v),), dtype=np.int32)
            return centers_, labels

        qs = np.linspace(0.0, 1.0, k)
        centers_ = np.quantile(v, qs).astype(np.float32)

        for _ in range(iters):
            d = np.abs(v[:, None] - centers_[None, :])
            labels = np.argmin(d, axis=1).astype(np.int32)
            new_centers = centers_.copy()
            for i in range(k):
                m = v[labels == i]
                if len(m) > 0:
                    new_centers[i] = float(np.mean(m))
            if np.allclose(new_centers, centers_):
                centers_ = new_centers
                break
            centers_ = new_centers

        order = np.argsort(centers_)
        centers_sorted = centers_[order]
        remap = np.zeros_like(order)
        remap[order] = np.arange(k, dtype=np.int32)
        labels_remap = remap[labels]
        return centers_sorted, labels_remap

    row_centers, row_labels = kmeans_1d(ys, rows)
    col_centers, col_labels = kmeans_1d(xs, cols)

    for i, c in enumerate(comps):
        c["row"] = int(row_labels[i])
        c["col"] = int(col_labels[i])

    ordered = sorted(comps, key=lambda c: (c["row"], c["col"], c["cy"], c["cx"]))

    dbg = {
        "inferred_cols": float(cols),
        "inferred_rows": float(rows),
        "row_centers": ",".join([str(int(round(v))) for v in row_centers.tolist()]),
        "col_centers": ",".join([str(int(round(v))) for v in col_centers.tolist()]),
    }
    return ordered, dbg


# ---------------------------
# Crop → center → edge-connected transparency + dehalo (+ stronger shadow removal)
# ---------------------------
def _estimate_bg_color_and_scale(arr_u8: np.ndarray, frame: int = 10) -> Tuple[np.ndarray, float]:
    """
    Estimate background RGB from the border pixels of the image.
    Returns (bg_rgb_float32[3], mad_float).
    """
    h, w, _ = arr_u8.shape
    f = max(1, int(frame))

    top = arr_u8[:f, :, :]
    bot = arr_u8[h - f:h, :, :]
    lef = arr_u8[:, :f, :]
    rig = arr_u8[:, w - f:w, :]

    border = np.concatenate(
        [
            top.reshape(-1, 3),
            bot.reshape(-1, 3),
            lef.reshape(-1, 3),
            rig.reshape(-1, 3),
        ],
        axis=0,
    )

    bg = np.median(border.astype(np.float32), axis=0)
    dist = np.sqrt(np.sum((border.astype(np.float32) - bg[None, :]) ** 2, axis=1))
    med = float(np.median(dist))
    mad = float(np.median(np.abs(dist - med)))
    return bg, mad


def _grow_region_from_mask(seed_mask: np.ndarray, allow_mask: np.ndarray) -> np.ndarray:
    """
    Region-grow (flood fill) starting from an existing True region (seed_mask),
    but only into pixels where allow_mask is True.
    4-neighborhood. Returns the expanded mask.
    """
    h, w = seed_mask.shape
    out = seed_mask.copy()
    q = deque()

    ys, xs = np.nonzero(seed_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        q.append((y, x))

    def try_push(y: int, x: int):
        if 0 <= y < h and 0 <= x < w and (not out[y, x]) and allow_mask[y, x]:
            out[y, x] = True
            q.append((y, x))

    while q:
        y, x = q.popleft()
        try_push(y - 1, x)
        try_push(y + 1, x)
        try_push(y, x - 1)
        try_push(y, x + 1)

    return out


def rgba_with_transparent_bg_edge_connected(
    tile_rgb: Image.Image,
    *,
    frame: int = 10,
    dehalo: bool = True,
    shrink: int = 1,
    feather: int = 1,
) -> Image.Image:
    """
    Convert background to transparency WITHOUT punching holes:

    Stage A (strict): edge flood-fill using tight bg threshold (near-white).
    Stage B (shadow): expand that edge-connected region into "shadow-like" pixels
                      using an auto-derived relaxed threshold *but only while staying
                      connected to the edge background region*.

    This reliably removes soft shadows (like under the pear) without deleting
    white highlights inside the subject.
    """
    rgb_img = tile_rgb.convert("RGB")
    arr = np.array(rgb_img, dtype=np.uint8)
    h, w, _ = arr.shape

    bg_rgb, mad = _estimate_bg_color_and_scale(arr, frame=frame)
    sigma = max(1.0, mad * 1.4826)

    arr_f = arr.astype(np.float32)
    dist_img = np.sqrt(np.sum((arr_f - bg_rgb[None, None, :]) ** 2, axis=2)).astype(np.float32)

    # --- Stage A: strict background ---
    bg_cut_strict = max(4.0, 3.2 * float(sigma))
    bg_conn = _floodfill_edge_connected_bg(dist_img, bg_cut_strict)

    # --- Stage B: shadow absorption (auto) ---
    # Define "shadow-like" = low chroma + not-too-dark.
    r = arr_f[..., 0]
    g = arr_f[..., 1]
    b = arr_f[..., 2]
    vmax = np.maximum(np.maximum(r, g), b)
    vmin = np.minimum(np.minimum(r, g), b)
    chroma = vmax - vmin

    # low chroma = near-neutral; exclude very dark pixels to protect outlines
    neutralish = (chroma <= 28.0) & (vmax >= 85.0)

    # Auto relaxed cutoff:
    # look at distances among neutral pixels that are already in/near background,
    # then choose a high percentile as the maximum "shadow distance" to absorb.
    # If that set is empty, fall back to a conservative multiple.
    neutral_d = dist_img[neutralish & (bg_conn | (dist_img <= (bg_cut_strict * 1.5)))]
    if neutral_d.size >= 32:
        p995 = float(np.percentile(neutral_d, 99.5))
        p99 = float(np.percentile(neutral_d, 99.0))
        cushion = max(3.0, 0.35 * (p995 - p99 + 1.0))
        bg_cut_relaxed = p995 + cushion
    else:
        bg_cut_relaxed = max(18.0, bg_cut_strict * 3.0)

    # Allow mask for region-growing shadows:
    # must be neutral-ish AND within relaxed distance
    allow_shadow = neutralish & (dist_img <= bg_cut_relaxed)

    # Grow background region from bg_conn into allow_shadow pixels.
    bg_final = _grow_region_from_mask(bg_conn, allow_shadow)

    # Alpha: transparent for edge-connected background (incl. absorbed shadows)
    alpha_u8 = np.full((h, w), 255, dtype=np.uint8)
    alpha_u8[bg_final] = 0

    # Feather edges a bit to prevent jaggies
    feather = max(0, int(feather))
    if feather > 0:
        a_img = Image.fromarray(alpha_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=feather))
        alpha_u8 = np.array(a_img, dtype=np.uint8)

    # Shrink alpha to cut remaining fringes
    shrink = max(0, int(shrink))
    if shrink > 0:
        a_img = Image.fromarray(alpha_u8, mode="L")
        for _ in range(shrink):
            a_img = a_img.filter(ImageFilter.MinFilter(3))
        alpha_u8 = np.array(a_img, dtype=np.uint8)

    # Dehalo: unblend against estimated background color
    if dehalo:
        a = (alpha_u8.astype(np.float32) / 255.0)
        a3 = np.maximum(a, 1e-6)[..., None]
        rgb = arr_f / 255.0
        bg = (bg_rgb / 255.0)[None, None, :]

        orig = (rgb - (1.0 - a)[..., None] * bg) / a3
        orig = np.clip(orig, 0.0, 1.0)
        out_rgb = (orig * 255.0 + 0.5).astype(np.uint8)
    else:
        out_rgb = arr

    out = Image.fromarray(out_rgb, mode="RGB").convert("RGBA")
    out.putalpha(Image.fromarray(alpha_u8, mode="L"))
    return out


def crop_and_center_subject(
    sprite_rgb: Image.Image,
    bbox: Tuple[int, int, int, int],
    *,
    out_size: int,
) -> Image.Image:
    """
    Crop subject bbox with an AUTO padding, then paste onto a WHITE out_size×out_size canvas centered.
    """
    x0, y0, x1, y1 = bbox
    w, h = sprite_rgb.size

    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    pad = int(round(max(bw, bh) * 0.06))
    pad = int(max(6, min(24, pad)))

    x0p = max(0, x0 - pad)
    y0p = max(0, y0 - pad)
    x1p = min(w, x1 + pad)
    y1p = min(h, y1 + pad)

    cut = sprite_rgb.crop((x0p, y0p, x1p, y1p)).convert("RGB")
    cw, ch = cut.size

    scale = min(1.0, float(out_size) / float(max(cw, ch)))
    if scale < 1.0:
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch * scale)))
        cut = cut.resize((nw, nh), Image.LANCZOS)
        cw, ch = cut.size

    canvas = Image.new("RGB", (out_size, out_size), (255, 255, 255))
    ox = (out_size - cw) // 2
    oy = (out_size - ch) // 2
    canvas.paste(cut, (ox, oy))
    return canvas


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Slice sprite sheet by detecting subjects, mapping top-left→bottom-right, centering into 256×256 tiles, and applying edge-connected transparency (with shadow cleanup)."
    )

    ap.add_argument("--tsv", default="image-data.tsv",
                    help="TSV with note IDs in first column. (default: image-data.tsv)")
    ap.add_argument("--sprite", required=True,
                    help="Sprite sheet PNG to process. (required)")
    ap.add_argument("--out", default="media/images",
                    help="Output directory for generated PNGs. (default: media/images)")
    ap.add_argument("--tile_out", type=int, default=256,
                    help="Output tile size (square). (default: 256)")
    ap.add_argument("--count", type=int, default=0,
                    help="How many images to produce. 0 = TSV row count. (default: 0)")

    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing PNGs.")
    ap.add_argument("--debug_detect", default="",
                    help="Write a debug PNG showing all detected subjects + order to this path.")
    ap.add_argument("--debug_list", action="store_true",
                    help="Print ordered bbox list (index, bbox, area).")

    args = ap.parse_args()

    notes = parse_tsv(Path(args.tsv).read_text(encoding="utf-8"))
    if not notes:
        raise SystemExit(f"No IDs found in TSV: {args.tsv}")

    want_n = int(args.count) if args.count > 0 else len(notes)

    sprite = Image.open(args.sprite).convert("RGB")
    W, H = sprite.size

    # 1) identify subjects (auto)
    comps, detect_dbg = detect_subjects(sprite, want_n=want_n)

    # 2) map subjects to row/col order
    ordered, map_dbg = assign_order_top_left(comps, want_n=want_n, img_w=W, img_h=H)

    if args.debug_list:
        for i, c in enumerate(ordered, start=1):
            print(f"{i:02d}\t{c['x0']},{c['y0']},{c['x1']},{c['y1']}\tarea={c['area']}")

    if args.debug_detect:
        dbg_img = sprite.copy()
        from PIL import ImageDraw
        d = ImageDraw.Draw(dbg_img)
        for i, c in enumerate(ordered, start=1):
            d.rectangle([c["x0"], c["y0"], c["x1"], c["y1"]], outline=(255, 0, 0), width=3)
            d.text((c["x0"] + 4, c["y0"] + 4), str(i), fill=(255, 0, 0))
        p = Path(args.debug_detect)
        p.parent.mkdir(parents=True, exist_ok=True)
        dbg_img.save(p)
        print(f"Debug detection image written: {p}")
        print(f"Detection debug: {detect_dbg}")
        print(f"Mapping debug: {map_dbg}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = skipped = 0
    limit = min(want_n, len(notes), len(ordered))

    for i in range(limit):
        note_id = notes[i]["id"]
        out_path = out_dir / f"{note_id}.png"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        c = ordered[i]
        bbox = (c["x0"], c["y0"], c["x1"], c["y1"])

        centered = crop_and_center_subject(sprite, bbox, out_size=args.tile_out)
        rgba = rgba_with_transparent_bg_edge_connected(
            centered,
            frame=10,
            dehalo=True,
            shrink=1,
            feather=1,
        )
        rgba.save(out_path, "PNG")
        saved += 1

    print(f"Sprite: {args.sprite}")
    print(f"TSV: {args.tsv}")
    print(f"Wanted: {want_n} | Detected: {len(ordered)} | Exported: {limit}")
    print(f"Inferred grid: {int(map_dbg['inferred_cols'])} cols × {int(map_dbg['inferred_rows'])} rows")
    print(f"Saved: {saved} | Skipped: {skipped} | Out: {out_dir}")


if __name__ == "__main__":
    main()
