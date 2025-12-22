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
  safePad,
  maxFill,
  gutterGuard, // extra “no-ink” band inside each tile near edges
  subjectTargetPct,
}) {
  const contentBox = `${maxFill}×${maxFill}px`;
  const contentPct = `${Math.round((maxFill / tile) * 100)}%`;

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

EXTREME SPACING REQUIREMENT (THIS IS CRITICAL — DO NOT VIOLATE):
- The gutters (${gutter}px) MUST remain completely empty white space (#FFFFFF).
- NOTHING from a subject may enter the gutters: no outline, no shading, no anti-aliasing, no whiskers, no tails, no “soft edges”.
- Inside EACH tile there must be a DOUBLE SAFETY ZONE:
  1) SAFE PAD: Keep at least ${safePad}px of pure white padding INSIDE the tile on all 4 sides.
  2) GUTTER GUARD: Additionally keep a further ${gutterGuard}px “no-ink” band INSIDE the tile edges.
- This means the entire subject must fit inside a centered ${contentBox} content box (${contentPct} of the tile).
- If uncertain, make subjects smaller. Whitespace is REQUIRED for correct slicing.

VERY IMPORTANT ANTI-OVERLAP RULE:
- The subject silhouette must not touch or approach the tile edge.
- Target subject size is only ${subjectTargetPct}% of tile width/height (NOT 85%).
- Prefer smaller + well-centered over bigger.

GRID-LINE PREVENTION (READ CAREFULLY):
- Do NOT draw ANY grid lines, even faint.
- Do NOT draw a table, separators, dividers, guides, frames, borders, crop marks, bleed marks.
- Do NOT outline tiles.
- Do NOT add shading/noise in gutters or margins; they must be perfectly flat #FFFFFF.

FORBIDDEN (DO NOT INCLUDE ANY OF THESE):
- No table/grid lines, no tile borders, no separators, no outlines, no strokes in gutters/margins
- No outer border/frame around the whole sheet
- No gradients or shadows in gutters/margins (must be perfectly flat #FFFFFF)
- No text, no numbers, no labels, no watermark, no logo

ART STYLE (adult comic-based, keep as-is):
- modern editorial comic illustration for adults (NOT childish, NOT kawaii)
- clean inked linework: thin-to-medium uniform outlines on the subject only
- subtle cel shading (2–3 tone blocks), minimal gradients
- muted, mature palette; avoid neon / toy-like colors
- consistent lighting direction (top-left), subtle
- no shadows below or around the subject

TRANSPARENCY-SAFE RULES:
- absolutely NO drop shadow below or around the subject
- NO ground shadow, NO contact shadow, NO ambient shadow
- NO glow, NO halo, NO vignette
- background is pure white #FFFFFF
- do NOT use pure white (#FFFFFF) inside the subject (use lightly tinted highlights instead)

COMPOSITION (strict, for EVERY tile):
- One main subject per tile.
- Subject must be centered horizontally AND vertically.
- Subject scale must be consistent across tiles.
- Subject MUST stay entirely inside the centered ${contentBox} content box.
- The tile edges must remain pure white with no marks, no outline, no shading.
- Keep silhouettes compact; avoid thin protrusions that might approach edges.

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

  /**
   * IMPORTANT CHANGE:
   * The model is “cheating” by letting outlines / anti-aliasing drift into the gutters.
   * So we make the *default* subject smaller and add a second buffer band.
   *
   * SAFE_PAD: inner white padding inside each tile
   * GUTTER_GUARD: extra band to ensure nothing goes near tile edges
   *
   * Together they force a clearly-visible whitespace gap even if the model draws a bit large.
   */
  const SAFE_PAD = Number(process.env.SPRITE_SAFE_PAD ?? "56"); // was 44; now stricter (52–72 works well)
  const GUTTER_GUARD = Number(process.env.SPRITE_GUTTER_GUARD ?? "12"); // extra no-ink band inside edges

  const safePad = Math.max(24, Math.min(96, SAFE_PAD));
  const gutterGuard = Math.max(0, Math.min(32, GUTTER_GUARD));

  // effective pad = safePad + gutterGuard
  const effectivePad = safePad + gutterGuard;
  const maxFill = Math.max(64, TILE - 2 * effectivePad);

  // Tell the generator explicitly to aim smaller than before (was 85%)
  const subjectTargetPct = Math.max(55, Math.min(78, Math.round((maxFill / TILE) * 100)));

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
    safePad,
    maxFill,
    gutterGuard,
    subjectTargetPct,
  });

  console.log(prompt);

  const config = {
    TILE,
    GUTTER,
    MARGIN,
    SAFE_PAD: safePad,
    GUTTER_GUARD: gutterGuard,
    EFFECTIVE_PAD: effectivePad,
    MAX_FILL: maxFill,
    SUBJECT_TARGET_PCT: subjectTargetPct,
    cols,
    rows: gridRows,
    count: rows.length,
    expected_width,
    expected_height,
  };
  console.log("\n---\nSPRITE_LAYOUT_CONFIG_JSON:");
  console.log(JSON.stringify(config, null, 2));
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
