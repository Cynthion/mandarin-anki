import fs from "node:fs/promises";

function parseTSV(tsv) {
  const lines = tsv
    .split(/\r?\n/)
    .map(l => l.trimEnd())
    .filter(l => l && !l.trimStart().startsWith("#"));

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
  return rows.filter(r => r.id);
}

function toSubject(row) {
  // Meaning is what to draw; hanzi is a small hint.
  const base = (row.meaning || row.hanzi || row.id).trim();
  const hint = row.hanzi ? `(${row.hanzi})` : "";
  return `${base} ${hint}`.trim();
}

function computeGrid(n) {
  const cols = Math.ceil(Math.sqrt(n));
  const rows = Math.ceil(n / cols);
  return { cols, rows };
}

function expectedSize(tile, gutter, margin, cols, rows) {
  return {
    width: 2 * margin + cols * tile + (cols - 1) * gutter,
    height: 2 * margin + rows * tile + (rows - 1) * gutter,
  };
}

async function main() {
  const tsvPath = process.argv[2] ?? "image-data.tsv";
  const label = process.argv[3] ?? "batch";

  // Keep these stable across all runs
  const TILE = 256;
  const GUTTER = 24;
  const MARGIN = 24;

  const tsv = await fs.readFile(tsvPath, "utf-8");
  const rows = parseTSV(tsv);

  if (rows.length === 0) {
    throw new Error(`No note IDs found in ${tsvPath}`);
  }

  const { cols, rows: gridRows } = computeGrid(rows.length);
  const { width: expected_width, height: expected_height } =
    expectedSize(TILE, GUTTER, MARGIN, cols, gridRows);

  const subjects = rows.map((r, i) => `${String(i + 1).padStart(2, "0")}. ${toSubject(r)}`);

  const prompt = `
Create ONE single PNG sprite sheet for Anki flashcards.

HARD REQUIREMENTS (must match exactly):
- Pixel size: ${expected_width}×${expected_height}px EXACT (no extra padding, no cropping)
- Grid: ${cols} columns × ${gridRows} rows
- Tile size: ${TILE}×${TILE}px
- Gutter between tiles: ${GUTTER}px (horizontal AND vertical), pure #FFFFFF
- Outer margin around the entire grid: ${MARGIN}px, pure #FFFFFF
- Background everywhere is pure white #FFFFFF

ABSOLUTELY FORBIDDEN (do NOT add these):
- No outer border/frame around the whole sheet
- No cell borders, no grid lines, no separators, no outlines
- No shadows/gradients/lines in gutters or margins (must be flat #FFFFFF)
- No text, no numbers, no labels, no watermark

ART STYLE (keep consistent across future sheets):
- clean minimal digital illustration, friendly, modern
- soft shading, simple shapes, slightly rounded forms
- consistent lighting direction (top-left)
- centered subject, big and readable
- keep a safe border inside each tile (don’t touch tile edges)

BATCH LABEL: ${label}

TILES (in exact order, left-to-right, top-to-bottom):
${subjects.join("\n")}
`.trim();

  console.log(prompt);
  console.log("\n---\n");
  console.log("Optional config (FYI / debugging only):");
  console.log(
    JSON.stringify(
      { TILE, GUTTER, MARGIN, cols, rows: gridRows, count: rows.length, expected_width, expected_height },
      null,
      2
    )
  );
}

main().catch(err => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
