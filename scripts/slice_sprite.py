#!/usr/bin/env python3
"""
slice_sprite.py

Slice a sprite sheet into per-note PNGs.

GOAL (robust / no manual tuning):
1) Identify subjects on a near-white background (no grid assumptions).
2) Order subjects strictly top-left → bottom-right (reading order).
3) Re-center each subject into a new 256×256 WHITE tile.
4) Make ONLY the NEW tile’s edge-connected background transparent:
   ✅ removes outer background only (flood fill from tile edge)
   ✅ does NOT remove white/near-white inside the subject
   ✅ ALSO removes soft edge-connected shadows

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
        if _id.lower() == "id":
            continue
        rows.append({"id": _id})
    return rows


# ---------------------------
# Flood fill (edge-connected)
# ---------------------------
def _floodfill_edge_connected_bool(bg_like: np.ndarray) -> np.ndarray:
    """
    Flood-fill starting from the image edge on a boolean bg_like mask.
    Returns bg_conn mask (H, W) True for background connected to the border.
    """
    h, w = bg_like.shape
    bg_conn = np.zeros((h, w), dtype=np.bool_)
    q = deque()

    def push(y: int, x: int):
        if 0 <= y < h and 0 <= x < w and (not bg_conn[y, x]) and bg_like[y, x]:
            bg_conn[y, x] = True
            q.append((y, x))

    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    while q:
        y, x = q.popleft()
        push(y - 1, x)
        push(y + 1, x)
        push(y, x - 1)
        push(y, x + 1)

    return bg_conn

def _floodfill_edge_connected_bg(dist_img: np.ndarray, bg_cut: float) -> np.ndarray:
    """
    Flood-fill background starting from the image edge:
    A pixel is "background-like" if dist <= bg_cut.
    Returns bg_conn mask (H, W) True for background connected to the border.
    """
    bg_like = dist_img <= bg_cut
    return _floodfill_edge_connected_bool(bg_like)

def _floodfill_from_seed(mask: np.ndarray, sy: int, sx: int) -> np.ndarray:
    """
    Flood-fill (4-connected) on a boolean mask starting at (sy, sx).
    Returns a boolean mask for the connected component.
    """
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.bool_)

    if not (0 <= sy < h and 0 <= sx < w):
        return out
    if not mask[sy, sx]:
        return out

    q = deque()
    out[sy, sx] = True
    q.append((sy, sx))

    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and (not out[ny, nx]) and mask[ny, nx]:
                out[ny, nx] = True
                q.append((ny, nx))
    return out

def _find_nearest_true(mask: np.ndarray, sy: int, sx: int, max_r: int = 24) -> Optional[Tuple[int, int]]:
    """
    If seed isn't inside foreground, search a small radius for the nearest True pixel.
    Returns (y, x) or None.
    """
    h, w = mask.shape
    if 0 <= sy < h and 0 <= sx < w and mask[sy, sx]:
        return (sy, sx)

    for r in range(1, max_r + 1):
        y0, y1 = max(0, sy - r), min(h - 1, sy + r)
        x0, x1 = max(0, sx - r), min(w - 1, sx + r)

        # scan perimeter of the square ring (cheap-ish)
        for x in range(x0, x1 + 1):
            if mask[y0, x]:
                return (y0, x)
            if mask[y1, x]:
                return (y1, x)
        for y in range(y0, y1 + 1):
            if mask[y, x0]:
                return (y, x0)
            if mask[y, x1]:
                return (y, x1)

    return None


# ---------------------------
# Foreground mask (FIXED)
# ---------------------------
def build_foreground_mask_autocut(img_rgb: Image.Image, *, frame: int = 12) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build a foreground mask for a near-white background WITHOUT user parameters.

    IMPORTANT FIX:
    ----------------
    Your failure case (big bottom-left portrait gets "missed") is caused by
    background estimation / distance thresholds being polluted when a subject is
    close to the image border. That makes the flood fill treat parts of the subject
    as "background-like" and erase it from the foreground mask.

    This version DOES NOT estimate bg color by RGB distance.
    Instead it detects background by a property that is stable for these sprites:
      - background is bright AND low-chroma (near-white / near-neutral)
    and it calibrates thresholds from the border.

    Result: the large portrait cannot be swallowed just because it sits near the border.
    """
    arr = np.array(img_rgb.convert("RGB"), dtype=np.uint8)
    a = arr.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]

    vmax = np.maximum(np.maximum(r, g), b)
    vmin = np.minimum(np.minimum(r, g), b)
    chroma = vmax - vmin  # 0 for perfect gray/white

    h, w = vmax.shape
    f = max(1, int(frame))

    border_v = np.concatenate(
        [
            vmax[:f, :].reshape(-1),
            vmax[h - f:h, :].reshape(-1),
            vmax[:, :f].reshape(-1),
            vmax[:, w - f:w].reshape(-1),
        ],
        axis=0,
    )
    border_c = np.concatenate(
        [
            chroma[:f, :].reshape(-1),
            chroma[h - f:h, :].reshape(-1),
            chroma[:, :f].reshape(-1),
            chroma[:, w - f:w].reshape(-1),
        ],
        axis=0,
    )

    # Brightness threshold: background is very bright.
    # Use a low percentile to allow subtle anti-aliased edges/shadows.
    tv = float(np.percentile(border_v, 5.0))
    tv = max(170.0, tv - 12.0)

    # Chroma threshold: background is near-neutral.
    # Use high percentile + cushion to tolerate compression / slight tint.
    tc = float(np.percentile(border_c, 97.0))
    tc = min(80.0, max(8.0, tc + 6.0))

    # Background-like pixels (core)
    bg_like = (vmax >= tv) & (chroma <= tc)

    # Allow mild shadows as background too, but ONLY if they are neutral-ish.
    # This helps flood-fill pass through soft gray shadows without "breaking" bg connectivity.
    shadow_like = (vmax >= (tv - 45.0)) & (chroma <= (tc * 1.35))
    bg_like = bg_like | shadow_like

    bg_conn = _floodfill_edge_connected_bool(bg_like)
    fg = ~bg_conn

    dbg = {
        "frame": float(frame),
        "tv_brightness": float(tv),
        "tc_chroma": float(tc),
        "bg_like_pct": float(bg_like.mean() * 100.0),
        "bg_conn_pct": float(bg_conn.mean() * 100.0),
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
    k = max(3, r * 2 + 1)
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


def _fill_holes(mask_bool: np.ndarray) -> np.ndarray:
    """
    Fill holes inside foreground regions without bridging separate subjects.
    Holes are background (False) regions that are NOT connected to the image border
    in the inverse mask.
    """
    h, w = mask_bool.shape
    inv = ~mask_bool

    bg_conn = np.zeros((h, w), dtype=np.bool_)
    q = deque()

    def push(y: int, x: int):
        if 0 <= y < h and 0 <= x < w and (not bg_conn[y, x]) and inv[y, x]:
            bg_conn[y, x] = True
            q.append((y, x))

    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    while q:
        y, x = q.popleft()
        push(y - 1, x)
        push(y + 1, x)
        push(y, x - 1)
        push(y, x + 1)

    holes = inv & (~bg_conn)
    return mask_bool | holes


# ---------------------------
# Connected components (union-find, 4-connected)
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
    Find connected components (4-connected) in a boolean mask.
    Returns list: {x0,y0,x1,y1,area,cx,cy}
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)

    uf = _UF()
    next_label = 1

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue

            neighbors = []
            if x > 0 and labels[y, x - 1] > 0:
                neighbors.append(labels[y, x - 1])
            if y > 0 and labels[y - 1, x] > 0:
                neighbors.append(labels[y - 1, x])

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
# Subject detection
# ---------------------------
def detect_subjects(sprite_rgb: Image.Image, want_n: int) -> Tuple[List[Dict[str, int]], Dict[str, float]]:
    """
    Auto-detect ALL subjects on a near-white sprite.

    - Uses the FIXED foreground mask (brightness/chroma).
    - Avoids trimming to want_n; returns all plausible components.
    - Still uses want_n only as a hint to avoid over-cleaning / under-detecting.
    """
    fg0, dbg0 = build_foreground_mask_autocut(sprite_rgb, frame=14)

    # Keep morphology conservative (avoid bridging gaps)
    open_radii = [0, 1, 2, 3]
    close_radii = [0, 1, 2]

    best: Optional[List[Dict[str, int]]] = None
    best_dbg: Optional[Dict[str, float]] = None
    best_score: Optional[float] = None

    for o in open_radii:
        fg = _morph_open(fg0, radius=o)
        fg = _fill_holes(fg)

        for c in close_radii:
            fg2 = _morph_close(fg, radius=c) if c > 0 else fg

            comps = connected_components_bbox(fg2)
            if not comps:
                continue

            # Sort by area desc for robust stats
            comps = sorted(comps, key=lambda z: z["area"], reverse=True)

            # Robust min-area based on top candidates (avoid tiny specks)
            top_k = comps[: min(len(comps), max(30, want_n * 2, 12))]
            med_area = float(np.median([t["area"] for t in top_k])) if top_k else 0.0
            # Slightly more permissive than before: we want ALL subjects
            min_area = max(60, int(med_area * 0.04)) if med_area > 0 else 60

            comps2 = [cc for cc in comps if cc["area"] >= min_area]
            count = len(comps2)

            # Score: prefer higher count but punish obvious noise explosions
            # (we don't know the true count; we only have want_n as a weak hint)
            # - Encourage >= want_n
            # - Penalize huge overshoots (often due to noise)
            miss_penalty = 600.0 if count < want_n else 0.0
            overshoot = max(0, count - max(want_n, 1))
            overshoot_penalty = 1.2 * float(overshoot)

            # Morphology complexity penalty (prefer simpler)
            score = (count * 2.0) - miss_penalty - overshoot_penalty - (o * 1.1) - (c * 0.7)

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
                        "want_n_hint": float(want_n),
                        "score": float(score),
                    }
                )

    if best is None or best_dbg is None or len(best) == 0:
        raise SystemExit("No subjects detected. Is the sprite background near-white?")

    return best, best_dbg

def _cluster_1d(values: np.ndarray, eps: float) -> List[Tuple[float, float, int]]:
    """
    Cluster sorted 1D values into groups where consecutive points differ by <= eps.
    Returns list of clusters as (center, spread, count).
    """
    if values.size == 0:
        return []
    v = np.sort(values.astype(np.float32))
    clusters = []
    start = 0
    for i in range(1, len(v)):
        if float(v[i] - v[i - 1]) > float(eps):
            chunk = v[start:i]
            center = float(np.median(chunk))
            spread = float(np.percentile(chunk, 90) - np.percentile(chunk, 10)) if chunk.size > 1 else 0.0
            clusters.append((center, spread, int(chunk.size)))
            start = i
    chunk = v[start:]
    center = float(np.median(chunk))
    spread = float(np.percentile(chunk, 90) - np.percentile(chunk, 10)) if chunk.size > 1 else 0.0
    clusters.append((center, spread, int(chunk.size)))
    return clusters


def assign_order_virtual_grid(
    comps: List[Dict[str, int]],
    want_n: int,
    img_w: int,
    img_h: int,
) -> Tuple[List[Dict[str, int]], Dict[str, float]]:
    """
    Assign ALL detected subjects into a "virtual grid" inferred from stats,
    then output subjects in strict top-left → bottom-right grid order.
    If there are too many subjects, bottom-right ones get dropped.

    Strategy:
    1) Infer row clusters from cy, using eps ~ median component height * 0.65
    2) Infer col clusters from cx, using eps ~ median component width  * 0.65
    3) Assign each component to nearest (row, col) cell
    4) Resolve cell collisions by choosing best (largest area + closest to cell center)
    5) Read cells row-major and take first want_n
    """
    if not comps:
        return [], {"rows": 0.0, "cols": 0.0}

    # Basic size stats for adaptive clustering tolerances
    hs = np.array([max(1, c["y1"] - c["y0"]) for c in comps], dtype=np.float32)
    ws = np.array([max(1, c["x1"] - c["x0"]) for c in comps], dtype=np.float32)
    med_h = float(np.median(hs))
    med_w = float(np.median(ws))

    eps_row = max(10.0, med_h * 0.65)
    eps_col = max(10.0, med_w * 0.65)

    cys = np.array([c["cy"] for c in comps], dtype=np.float32)
    cxs = np.array([c["cx"] for c in comps], dtype=np.float32)

    row_clusters = _cluster_1d(cys, eps=eps_row)
    col_clusters = _cluster_1d(cxs, eps=eps_col)

    # Sort clusters top->bottom / left->right
    row_centers = np.array(sorted([rc[0] for rc in row_clusters]), dtype=np.float32)
    col_centers = np.array(sorted([cc[0] for cc in col_clusters]), dtype=np.float32)

    # Fallback if clustering degenerates
    if row_centers.size == 0:
        row_centers = np.array([float(np.median(cys))], dtype=np.float32)
    if col_centers.size == 0:
        col_centers = np.array([float(np.median(cxs))], dtype=np.float32)

    def nearest_index(arr: np.ndarray, v: float) -> int:
        return int(np.argmin(np.abs(arr - float(v))))

    # Build cell map
    # cell[(r,c)] = chosen_component
    # extras = components that collided and lost
    cell: Dict[Tuple[int, int], Dict[str, int]] = {}
    extras: List[Dict[str, int]] = []

    collisions = 0
    for comp in comps:
        r = nearest_index(row_centers, comp["cy"])
        c = nearest_index(col_centers, comp["cx"])
        key = (r, c)

        # "Fit" score: prefer larger, more central in its cell band
        dy = abs(float(comp["cy"]) - float(row_centers[r]))
        dx = abs(float(comp["cx"]) - float(col_centers[c]))
        # normalize by eps to make comparable
        center_penalty = (dy / max(1.0, eps_row)) + (dx / max(1.0, eps_col))
        fit = float(comp["area"]) - (center_penalty * (0.15 * float(comp["area"]) + 50.0))

        if key not in cell:
            comp2 = dict(comp)
            comp2["_grid_r"] = r
            comp2["_grid_c"] = c
            comp2["_fit"] = fit
            cell[key] = comp2
        else:
            collisions += 1
            cur = cell[key]
            if fit > float(cur.get("_fit", cur["area"])):
                extras.append(cur)
                comp2 = dict(comp)
                comp2["_grid_r"] = r
                comp2["_grid_c"] = c
                comp2["_fit"] = fit
                cell[key] = comp2
            else:
                extras.append(comp)

    # Emit in row-major cell order
    ordered: List[Dict[str, int]] = []
    rows_n = int(row_centers.size)
    cols_n = int(col_centers.size)
    for r in range(rows_n):
        for c in range(cols_n):
            key = (r, c)
            if key in cell:
                ordered.append(cell[key])

    # If grid produced too few (e.g., odd layouts), fill remaining by pure reading order of unused
    used_ids = set(id(x) for x in ordered)
    unused = [c for c in comps if id(c) not in used_ids]
    unused_sorted = sorted(unused, key=lambda z: (z["cy"], z["cx"]))
    for u in unused_sorted:
        ordered.append(u)

    # Now apply the strict "if too many, drop bottom-right"
    # We already built row-major; just take the first want_n.
    ordered_final = ordered[:want_n]

    dbg = {
        "median_h": float(med_h),
        "median_w": float(med_w),
        "eps_row": float(eps_row),
        "eps_col": float(eps_col),
        "rows_detected": float(rows_n),
        "cols_detected": float(cols_n),
        "grid_capacity": float(rows_n * cols_n),
        "components_in": float(len(comps)),
        "collisions": float(collisions),
        "extras_count": float(len(extras)),
        "ordered_before_trim": float(len(ordered)),
        "ordered_final": float(len(ordered_final)),
        "want_n": float(want_n),
    }
    return ordered_final, dbg


# ---------------------------
# Crop → center → edge-connected transparency + dehalo
# ---------------------------
def _estimate_bg_color_and_scale(arr_u8: np.ndarray, frame: int = 10) -> Tuple[np.ndarray, float]:
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
    rgb_img = tile_rgb.convert("RGB")
    arr = np.array(rgb_img, dtype=np.uint8)
    h, w, _ = arr.shape

    bg_rgb, mad = _estimate_bg_color_and_scale(arr, frame=frame)
    sigma = max(1.0, mad * 1.4826)

    arr_f = arr.astype(np.float32)
    dist_img = np.sqrt(np.sum((arr_f - bg_rgb[None, None, :]) ** 2, axis=2)).astype(np.float32)

    bg_cut_strict = max(4.0, 3.2 * float(sigma))
    bg_conn = _floodfill_edge_connected_bg(dist_img, bg_cut_strict)

    r = arr_f[..., 0]
    g = arr_f[..., 1]
    b = arr_f[..., 2]
    vmax = np.maximum(np.maximum(r, g), b)
    vmin = np.minimum(np.minimum(r, g), b)
    chroma = vmax - vmin
    neutralish = (chroma <= 28.0) & (vmax >= 85.0)

    neutral_d = dist_img[neutralish & (bg_conn | (dist_img <= (bg_cut_strict * 1.5)))]
    if neutral_d.size >= 32:
        p995 = float(np.percentile(neutral_d, 99.5))
        p99 = float(np.percentile(neutral_d, 99.0))
        cushion = max(3.0, 0.35 * (p995 - p99 + 1.0))
        bg_cut_relaxed = p995 + cushion
    else:
        bg_cut_relaxed = max(18.0, bg_cut_strict * 3.0)

    allow_shadow = neutralish & (dist_img <= bg_cut_relaxed)
    bg_final = _grow_region_from_mask(bg_conn, allow_shadow)

    alpha_u8 = np.full((h, w), 255, dtype=np.uint8)
    alpha_u8[bg_final] = 0

    feather = max(0, int(feather))
    if feather > 0:
        a_img = Image.fromarray(alpha_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=feather))
        alpha_u8 = np.array(a_img, dtype=np.uint8)

    shrink = max(0, int(shrink))
    if shrink > 0:
        a_img = Image.fromarray(alpha_u8, mode="L")
        for _ in range(shrink):
            a_img = a_img.filter(ImageFilter.MinFilter(3))
        alpha_u8 = np.array(a_img, dtype=np.uint8)

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


def crop_and_center_subject_masked(
    sprite_rgb: Image.Image,
    bbox: Tuple[int, int, int, int],
    seed_xy: Tuple[int, int],
    *,
    out_size: int,
) -> Image.Image:
    """
    Crop a region around bbox, then isolate ONLY the connected component that contains seed_xy
    (in sprite coordinates). This prevents neighbor-subject pixels from leaking into the crop.

    Returns an RGB tile on WHITE background (so your existing edge-connected background remover works).
    """
    x0, y0, x1, y1 = bbox
    W, H = sprite_rgb.size
    cx, cy = seed_xy

    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    # Keep your padding, but it’s safe now because we’ll mask by component
    pad = int(round(max(bw, bh) * 0.06))
    pad = int(max(6, min(24, pad)))

    x0p = max(0, x0 - pad)
    y0p = max(0, y0 - pad)
    x1p = min(W, x1 + pad)
    y1p = min(H, y1 + pad)

    cut = sprite_rgb.crop((x0p, y0p, x1p, y1p)).convert("RGB")

    # Build foreground mask on the cut
    fg_cut, _ = build_foreground_mask_autocut(cut, frame=10)

    # Seed in cut coordinates
    sx = int(cx - x0p)
    sy = int(cy - y0p)

    # If seed isn't on fg (can happen due to thresholds), find nearest fg pixel
    nearest = _find_nearest_true(fg_cut, sy, sx, max_r=32)
    if nearest is None:
        # Fallback: no fg found near seed — just use bbox crop without masking (rare)
        # (still centered below)
        masked_rgb = cut
        comp_mask = None
    else:
        sy2, sx2 = nearest
        comp_mask = _floodfill_from_seed(fg_cut, sy2, sx2)

        # Tight-bbox the component inside the cut (so we remove surrounding neighbor areas entirely)
        ys, xs = np.nonzero(comp_mask)
        if ys.size == 0:
            masked_rgb = cut
            comp_mask = None
        else:
            yy0, yy1 = int(ys.min()), int(ys.max()) + 1
            xx0, xx1 = int(xs.min()), int(xs.max()) + 1

            # small safety pad in the cut frame (NOT large enough to hit neighbors usually)
            tight_pad = 2
            yy0 = max(0, yy0 - tight_pad)
            xx0 = max(0, xx0 - tight_pad)
            yy1 = min(comp_mask.shape[0], yy1 + tight_pad)
            xx1 = min(comp_mask.shape[1], xx1 + tight_pad)

            cut = cut.crop((xx0, yy0, xx1, yy1))
            comp_mask = comp_mask[yy0:yy1, xx0:xx1]

            # Paint everything outside the component white (so later bg removal works)
            arr = np.array(cut, dtype=np.uint8)
            inv = ~comp_mask
            arr[inv] = np.array([255, 255, 255], dtype=np.uint8)
            masked_rgb = Image.fromarray(arr, mode="RGB")

    # Scale down to fit out_size (same behavior as your old function)
    cw, ch = masked_rgb.size
    scale = min(1.0, float(out_size) / float(max(cw, ch)))
    if scale < 1.0:
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch * scale)))
        masked_rgb = masked_rgb.resize((nw, nh), Image.LANCZOS)
        cw, ch = masked_rgb.size

    # Center on WHITE (your transparency step expects this)
    canvas = Image.new("RGB", (out_size, out_size), (255, 255, 255))
    ox = (out_size - cw) // 2
    oy = (out_size - ch) // 2
    canvas.paste(masked_rgb, (ox, oy))
    return canvas

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Slice sprite sheet by detecting subjects, ordering top-left→bottom-right, centering into tiles, and applying edge-connected transparency (with shadow cleanup)."
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

    comps_all, detect_dbg = detect_subjects(sprite, want_n=want_n)
    ordered, map_dbg = assign_order_virtual_grid(comps_all, want_n=want_n, img_w=W, img_h=H)


    if args.debug_list:
        for i, c in enumerate(ordered, start=1):
            print(f"{i:02d}\t{c['x0']},{c['y0']},{c['x1']},{c['y1']}\tarea={c['area']}")

    if args.debug_detect:
        dbg_img = sprite.copy()
        from PIL import ImageDraw
        d = ImageDraw.Draw(dbg_img)

        # draw all comps faint
        for c in comps_all:
            d.rectangle([c["x0"], c["y0"], c["x1"], c["y1"]], outline=(0, 120, 255), width=2)

        # draw chosen ordered strong + index
        for i, c in enumerate(ordered, start=1):
            d.rectangle([c["x0"], c["y0"], c["x1"], c["y1"]], outline=(255, 0, 0), width=4)
            d.text((c["x0"] + 4, c["y0"] + 4), str(i), fill=(255, 0, 0))

        p = Path(args.debug_detect)
        p.parent.mkdir(parents=True, exist_ok=True)
        dbg_img.save(p)
        print(f"Debug detection image written: {p}")
        print(f"Detection debug: {detect_dbg}")
        print(f"Grid mapping debug: {map_dbg}")

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

        centered = crop_and_center_subject_masked(
            sprite,
            bbox,
            seed_xy=(c["cx"], c["cy"]),
            out_size=args.tile_out,
        )

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
    print(f"Wanted: {want_n} | Detected(all): {len(comps_all)} | Ordered: {len(ordered)} | Exported: {limit}")
    print(
        f"Grid: {int(map_dbg['rows_detected'])} rows × {int(map_dbg['cols_detected'])} cols "
        f"(cap {int(map_dbg['grid_capacity'])}) | collisions: {int(map_dbg['collisions'])}"
    )
    print(f"Saved: {saved} | Skipped: {skipped} | Out: {out_dir}")

if __name__ == "__main__":
    main()
