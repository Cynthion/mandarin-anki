import fs from "node:fs/promises";

function parseTSV(tsv) {
  const lines = tsv
    .split(/\r?\n/)
    .map((l) => l.replace(/\s+$/g, "")) // trimEnd but keep leading tabs if any
    .filter((l) => l && !l.trimStart().startsWith("#"));

  const rows = [];
  for (const line of lines) {
    const cols = line.split("\t");
    rows.push({
      id: (cols[0] ?? "").trim(),
      hanzi: (cols[1] ?? "").trim(),
      pinyin: (cols[2] ?? "").trim(),
      meaning: (cols[3] ?? "").trim(),
      tags: (cols[10] ?? "").trim(),
    });
  }
  return rows.filter((r) => r.id);
}

function toSubject(row) {
  // meaning = what to draw; hanzi = hint only
  const base = (row.meaning || row.hanzi || row.id).trim();
  const hint = row.hanzi ? `(${row.hanzi})` : "";
  return `${base} ${hint}`.trim();
}

/**
 * Compute a pleasant grid for any batch size.
 * - By default: near-square
 * - Also caps columns so we don't get super-wide sheets.
 */
function computeGrid(n, maxCols = 6) {
  if (n <= 0) return { cols: 1, rows: 1 };

  let cols = Math.ceil(Math.sqrt(n));
  cols = Math.max(1, Math.min(cols, maxCols));
  let rows = Math.ceil(n / cols);

  // If last row is extremely empty, try shifting to fewer columns (more compact)
  // Example: n=17, sqrt=>5, fine. n=10 -> 4 cols => 3 rows (ok)
  while (cols > 1 && (rows - 1) * cols >= n) {
    rows -= 1;
  }

  return { cols, rows };
}

function expectedSize(tile, gutter, margin, cols, rows) {
  const width = 2 * margin + cols * tile + (cols - 1) * gutter;
  const height = 2 * margin + rows * tile + (rows - 1) * gutter;
  return { width, height };
}

function buildPrompt({
  label,
  cols,
  rows,
  tile,
  gutter,
  margin,
  expectedWidth,
  expectedHeight,
  subjects,
}) {
  return `
Create EXACTLY ONE PNG sprite sheet image.

CANVAS SIZE (ABSOLUTE, MUST MATCH EXACTLY):
- The final PNG must be exactly ${expectedWidth}×${expectedHeight} pixels.
- Do NOT add any extra padding, border, frame, shadow, or whitespace outside this canvas.
- Do NOT crop smaller than this size. Do NOT export larger than this size.

INVISIBLE GRID LAYOUT (DO NOT DRAW THE GRID):
- Grid is ${cols} columns × ${rows} rows.
- Each tile is exactly ${tile}×${tile}px.
- Horizontal gutter between tiles is exactly ${gutter}px, pure #FFFFFF.
- Vertical gutter between tiles is exactly ${gutter}px, pure #FFFFFF.
- Outer margin around the entire grid is exactly ${margin}px on all 4 sides, pure #FFFFFF.
- Background everywhere (including gutters and margins) is flat pure white #FFFFFF.
- IMPORTANT: The grid is only for placement; do NOT draw borders, lines, boxes, frames, or separators.

FORBIDDEN (DO NOT INCLUDE ANY OF THESE):
- No table/grid lines, no tile borders, no separators, no outlines, no strokes
- No outer border/frame around the whole sheet
- No gradients or shadows in gutters/margins (must be perfectly flat #FFFFFF)
- No text, no numbers, no labels, no watermark, no logo

ART STYLE (lock this style for all future sheets):
- clean minimal digital illustration, friendly, modern
- soft shading, simple shapes, slightly rounded forms
- consistent lighting from top-left
- consistent palette across the whole sheet
- subtle shadow ONLY under the subject (inside the tile), never in gutters/margins

COMPOSITION (strict, for EVERY tile):
- One main subject per tile.
- Subject must be centered horizontally AND vertically.
- Subject scale must be consistent across tiles.
- Subject should fill about 85% of tile height and about 85% of tile width.
- Keep at least 24px empty white padding inside each tile (nothing touches edges).
- Use consistent portrait framing for people: head + shoulders, same baseline and head position.
- If you add small icons (e.g., a heart), they must remain near the subject and inside the centered composition area (not near tile edges).

BATCH LABEL (for humans only, not drawn): ${label}

TILES (exact order, left-to-right then top-to-bottom):
${subjects.join("\n")}
`.trim();
}

async function main() {
  const tsvPath = process.argv[2] ?? "image-data.tsv";
  const label = process.argv[3] ?? "batch";

  // Stable layout lock
  const TILE = 256;
  const GUTTER = 24;
  const MARGIN = 24;

  // Optional: allow overriding maxCols for layout via env var (nice for big batches)
  const MAX_COLS = Number(process.env.SPRITE_MAX_COLS ?? "6") || 6;

  const tsv = await fs.readFile(tsvPath, "utf-8");
  const rows = parseTSV(tsv);

  if (rows.length === 0) {
    throw new Error(`No note IDs found in ${tsvPath}`);
  }

  const { cols, rows: gridRows } = computeGrid(rows.length, MAX_COLS);
  const { width: expected_width, height: expected_height } = expectedSize(
    TILE,
    GUTTER,
    MARGIN,
    cols,
    gridRows
  );

  const subjects = rows.map(
    (r, i) => `${String(i + 1).padStart(2, "0")}. ${toSubject(r)}`
  );

  const prompt = buildPrompt({
    label,
    cols,
    rows: gridRows,
    tile: TILE,
    gutter: GUTTER,
    margin: MARGIN,
    expectedWidth: expected_width,
    expectedHeight: expected_height,
    subjects,
  });

  // Print prompt
  console.log(prompt);

  // Print machine-readable layout config (useful for slicing/debugging/logs)
  const config = {
    TILE,
    GUTTER,
    MARGIN,
    cols,
    rows: gridRows,
    count: rows.length,
    expected_width,
    expected_height,
  };

  console.log("\n---\n");
  console.log("SPRITE_LAYOUT_CONFIG_JSON:");
  console.log(JSON.stringify(config, null, 2));
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
