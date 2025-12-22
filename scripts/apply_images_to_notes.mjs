// scripts/apply_images_to_notes.mjs
// Update deck/notes.tsv "image" field for IDs listed in image-data.tsv.
// - Matches rows by the stable `id` (col 1)
// - Writes: <img src="<ID>.png"> into the image column (col 10, index 9)
// - Optionally checks that media/images/<ID>.png exists before writing
//
// Usage examples:
//   node scripts/apply_images_to_notes.mjs
//   node scripts/apply_images_to_notes.mjs --dry-run
//   node scripts/apply_images_to_notes.mjs --notes deck/notes.tsv --imageData image-data.tsv --imagesDir media/images
//
// Exit code:
//   0 = success
//   1 = failure (e.g., files missing)

import fs from "node:fs/promises";
import path from "node:path";

const DEFAULTS = {
  notesPath: "deck/notes.tsv",
  imageDataPath: "image-data.tsv",
  imagesDir: "media/images",
  // Your deck schema is:
  // 1 id, 2 hanzi, 3 pinyin, 4 meaning, 5 example-hanzi, 6 example-pinyin,
  // 7 example-meaning, 8 audio, 9 audio-example, 10 image, 11 tags
  imageColIndex: 9, // 0-based (10th column)
  minCols: 11,
};

function parseArgs(argv) {
  const args = { ...DEFAULTS, dryRun: false, force: false, quiet: false };

  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--dry-run") args.dryRun = true;
    else if (a === "--force") args.force = true; // write even if png missing
    else if (a === "--quiet") args.quiet = true;
    else if (a === "--notes") args.notesPath = argv[++i] ?? args.notesPath;
    else if (a === "--imageData") args.imageDataPath = argv[++i] ?? args.imageDataPath;
    else if (a === "--imagesDir") args.imagesDir = argv[++i] ?? args.imagesDir;
    else if (a === "--help" || a === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${a}`);
    }
  }

  return args;
}

function printHelp() {
  console.log(`
apply_images_to_notes.mjs

Updates the "image" field in notes.tsv for all IDs listed in image-data.tsv.

Options:
  --notes <path>        Path to notes.tsv (default: ${DEFAULTS.notesPath})
  --imageData <path>    Path to image-data.tsv (default: ${DEFAULTS.imageDataPath})
  --imagesDir <dir>     Directory containing <ID>.png (default: ${DEFAULTS.imagesDir})
  --dry-run             Print what would change, do not write files
  --force               Write <img ...> even if the PNG is missing
  --quiet               Less output
  -h, --help            Show help

Notes:
  - Keeps comments (#...) and blank lines untouched
  - Ensures each row has at least ${DEFAULTS.minCols} TSV columns (pads with empty cells)
  - Writes: <img src="<ID>.png"> into column ${DEFAULTS.imageColIndex + 1}
`.trim());
}

function parseTsvLinesKeepFormatting(tsvText) {
  // Keep original line endings reasonably stable (we'll re-join with \n).
  // We preserve blank lines and comment lines verbatim.
  return tsvText.split(/\r?\n/);
}

function getIdFromLine(line) {
  const trimmed = line.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith("#")) return null;
  const cols = line.split("\t");
  const id = (cols[0] ?? "").trim();
  return id || null;
}

function parseIdsFromImageData(tsvText) {
  const lines = tsvText.split(/\r?\n/);
  const ids = [];
  for (const line of lines) {
    const l = line.trimEnd();
    if (!l.trim()) continue;
    if (l.trimStart().startsWith("#")) continue;
    const cols = l.split("\t");
    const id = (cols[0] ?? "").trim();
    if (id) ids.push(id);
  }
  // Deduplicate, but keep order stable (first occurrence wins)
  const seen = new Set();
  return ids.filter((id) => (seen.has(id) ? false : (seen.add(id), true)));
}

async function exists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

function ensureCols(cols, minCols) {
  if (cols.length >= minCols) return cols;
  const out = cols.slice();
  while (out.length < minCols) out.push("");
  return out;
}

async function main() {
  const args = parseArgs(process.argv);

  const [notesText, imageDataText] = await Promise.all([
    fs.readFile(args.notesPath, "utf-8"),
    fs.readFile(args.imageDataPath, "utf-8"),
  ]);

  const wantedIds = parseIdsFromImageData(imageDataText);
  if (wantedIds.length === 0) {
    throw new Error(`No IDs found in ${args.imageDataPath}`);
  }

  // Build quick lookup set
  const wantedSet = new Set(wantedIds);

  const lines = parseTsvLinesKeepFormatting(notesText);

  let changed = 0;
  let matched = 0;
  let missingInNotes = 0;
  let missingPng = 0;

  const foundInNotes = new Set();

  const outLines = [];
  for (const line of lines) {
    const id = getIdFromLine(line);
    if (!id || !wantedSet.has(id)) {
      outLines.push(line);
      continue;
    }

    foundInNotes.add(id);
    matched += 1;

    const pngPath = path.join(args.imagesDir, `${id}.png`);
    const hasPng = await exists(pngPath);

    if (!hasPng && !args.force) {
      missingPng += 1;
      if (!args.quiet) {
        console.warn(`[skip] ${id}: PNG not found at ${pngPath} (use --force to write anyway)`);
      }
      outLines.push(line);
      continue;
    }

    const cols = ensureCols(line.split("\t"), args.minCols);
    const newValue = `<img src="${id}.png">`;

    if ((cols[args.imageColIndex] ?? "") !== newValue) {
      cols[args.imageColIndex] = newValue;
      changed += 1;

      if (!args.quiet) {
        console.log(`[update] ${id} -> ${newValue}${hasPng ? "" : " (png missing, forced)"}`);
      }
    } else if (!args.quiet) {
      console.log(`[ok] ${id} already set`);
    }

    outLines.push(cols.join("\t"));
  }

  // IDs in image-data.tsv but not present in notes.tsv
  for (const id of wantedIds) {
    if (!foundInNotes.has(id)) {
      missingInNotes += 1;
      if (!args.quiet) console.warn(`[warn] ${id}: not found in ${args.notesPath}`);
    }
  }

  if (!args.quiet) {
    console.log(
      `\nSummary: matched=${matched}, changed=${changed}, missing_png_skipped=${missingPng}, missing_in_notes=${missingInNotes}`
    );
  }

  if (args.dryRun) {
    if (!args.quiet) console.log(`\n[dry-run] Not writing ${args.notesPath}`);
    return;
  }

  // Write back
  await fs.writeFile(args.notesPath, outLines.join("\n"), "utf-8");
  if (!args.quiet) console.log(`\nWrote: ${args.notesPath}`);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(1);
});
