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
    Auto-detect subjects on a near-white sprite.

    - Uses the FIXED foreground mask (brightness/chroma) to avoid missing the large portrait.
    - Avoids aggressive closing that would bridge white gaps.
    - If more than want_n components exist, selects the first want_n by TL→BR order.
    """
    fg0, dbg0 = build_foreground_mask_autocut(sprite_rgb, frame=14)

    open_radii = [0, 1, 2, 3]
    close_radii = [0, 1, 2]

    best: Optional[List[Dict[str, int]]] = None
    best_dbg: Optional[Dict[str, float]] = None
    best_score: Optional[float] = None

    def take_top_left_n(comps_list: List[Dict[str, int]]) -> List[Dict[str, int]]:
        return sorted(comps_list, key=lambda c: (c["cy"], c["cx"]))[:want_n]

    for o in open_radii:
        fg = _morph_open(fg0, radius=o)
        fg = _fill_holes(fg)

        for c in close_radii:
            fg2 = _morph_close(fg, radius=c) if c > 0 else fg

            comps = connected_components_bbox(fg2)
            if not comps:
                continue

            comps = sorted(comps, key=lambda z: z["area"], reverse=True)

            top_k = comps[: min(len(comps), max(12, want_n))]
            med_area = float(np.median([t["area"] for t in top_k])) if top_k else 0.0
            min_area = max(80, int(med_area * 0.06)) if med_area > 0 else 80
            comps2 = [cc for cc in comps if cc["area"] >= min_area]

            count = len(comps2)
            diff = abs(count - want_n)

            miss_penalty = 500.0 if count < want_n else 0.0
            extra_penalty = 4.0 * max(0, count - want_n)

            score = -(diff * 15.0) - miss_penalty - extra_penalty - (o * 1.2) - (c * 0.6)

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

            if count >= want_n:
                out = take_top_left_n(comps2)
                if len(out) == want_n:
                    best_dbg = dict(best_dbg or dbg0)
                    best_dbg["trimmed_from"] = float(count)
                    return out, best_dbg

    if best is None or best_dbg is None:
        raise SystemExit("No subjects detected. Is the sprite background near-white?")

    if len(best) > want_n:
        best_dbg = dict(best_dbg)
        best_dbg["trimmed_from"] = float(len(best))
        best = sorted(best, key=lambda c: (c["cy"], c["cx"]))[:want_n]

    if len(best) != want_n:
        raise SystemExit(
            f"Subject detection failed: expected {want_n} subjects, got {len(best)}.\n"
            f"Detection debug: {best_dbg}\n"
        )

    return best, best_dbg


# ---------------------------
# STRICT top-left → bottom-right ordering (row bucketing)
# ---------------------------
def assign_order_top_left(
    comps: List[Dict[str, int]],
    want_n: int,
    img_w: int,
    img_h: int,
) -> Tuple[List[Dict[str, int]], Dict[str, float]]:
    comps = list(comps)
    if len(comps) != want_n:
        raise SystemExit(f"Internal error: expected {want_n} comps, got {len(comps)}")

    comps_y = sorted(comps, key=lambda c: (c["cy"], c["cx"]))

    ys = np.array([c["cy"] for c in comps_y], dtype=np.float32)
    if len(ys) >= 2:
        dy = np.diff(ys)
        dy_pos = dy[dy > 0]
        if dy_pos.size > 0:
            p25 = float(np.percentile(dy_pos, 25.0))
            p75 = float(np.percentile(dy_pos, 75.0))
            thr = max(8.0, (p25 + p75) * 0.5)
        else:
            thr = 12.0
    else:
        thr = 12.0

    rows: List[List[Dict[str, int]]] = []
    cur: List[Dict[str, int]] = []
    cur_mean_y: float = float(comps_y[0]["cy"]) if comps_y else 0.0

    for c in comps_y:
        if not cur:
            cur = [c]
            cur_mean_y = float(c["cy"])
            continue

        if float(c["cy"]) - cur_mean_y > thr:
            rows.append(cur)
            cur = [c]
            cur_mean_y = float(c["cy"])
        else:
            cur.append(c)
            cur_mean_y = (cur_mean_y * (len(cur) - 1) + float(c["cy"])) / float(len(cur))

    if cur:
        rows.append(cur)

    ordered: List[Dict[str, int]] = []
    max_cols = 0
    for r in rows:
        r_sorted = sorted(r, key=lambda c: (c["cx"], c["cy"]))
        max_cols = max(max_cols, len(r_sorted))
        ordered.extend(r_sorted)

    dbg = {
        "row_break_threshold": float(thr),
        "rows_detected": float(len(rows)),
        "max_cols_in_a_row": float(max_cols),
    }
    return ordered, dbg


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


def crop_and_center_subject(
    sprite_rgb: Image.Image,
    bbox: Tuple[int, int, int, int],
    *,
    out_size: int,
) -> Image.Image:
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

    comps, detect_dbg = detect_subjects(sprite, want_n=want_n)
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
    print(f"Row grouping: {int(map_dbg['rows_detected'])} rows | max cols in a row: {int(map_dbg['max_cols_in_a_row'])}")
    print(f"Saved: {saved} | Skipped: {skipped} | Out: {out_dir}")


if __name__ == "__main__":
    main()
