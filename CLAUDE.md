# CLAUDE.md — Mandarin Anki Deck

This file provides guidance for AI assistants (Claude Code and similar tools) working in this repository.

---

## Project Overview

This repository is the **single source of truth** for a custom Mandarin Chinese Anki flashcard deck. It manages:

- Flashcard note data in a Git-friendly TSV file (`deck/notes.tsv`)
- Anki card templates and CSS (`anki/note-type/`)
- Optional media assets: audio MP3s and card images (`media/`)
- A browser-based live preview system (`preview/`)
- Utility scripts for image generation and media sync (`scripts/`)

**Stack:** Node.js (npm scripts), Python 3 (utility scripts), HTML/CSS (Anki templates)

---

## Repository Structure

```
mandarin-anki/
├── deck/
│   ├── notes.tsv           # Master flashcard data (source of truth)
│   └── notes.schema.md     # Field schema documentation
├── anki/
│   └── note-type/
│       ├── front1.html     # Card 1 front (Mandarin → English)
│       ├── back1.html      # Card 1 back
│       ├── front2.html     # Card 2 front (English → Mandarin)
│       ├── back2.html      # Card 2 back
│       └── styling.css     # Shared card styling
├── media/
│   ├── audio/              # MP3 audio files
│   ├── images/             # Generated card images (PNG)
│   └── sprites/            # Sprite sheets and image-data.tsv
├── preview/
│   ├── index.html          # Preview page
│   ├── preview.js          # Card renderer for browser preview
│   └── mock.json           # Sample notes for preview
├── scripts/
│   ├── sync_media_to_anki.py       # Copy media to Anki collection.media
│   ├── make_sprite_prompt.mjs      # Generate ChatGPT sprite prompt
│   ├── apply_images_to_notes.mjs   # Add image tags to TSV
│   └── slice_sprite.py             # Slice sprite sheets into PNGs
├── .github/
│   └── copilot-instructions.md     # Detailed AI editing rules (authoritative)
├── package.json
└── README.md
```

---

## Development Commands

Install dependencies first (only needed once):

```bash
npm install
```

| Command | Description |
|---|---|
| `npm run preview` | Serve card preview at http://localhost:5173/preview/ |
| `npm run sprite:prompt` | Generate ChatGPT sprite prompt from `media/sprites/image-data.tsv` |
| `npm run sprite:slice` | Slice `media/sprites/image.png` into PNGs in `media/images/` |
| `npm run sprite:apply-images` | Update `deck/notes.tsv` image field with generated image tags |
| `npm run anki:sync-media:mac` | Copy media to Anki on macOS |
| `npm run anki:sync-media:windows` | Copy media to Anki on Windows |

---

## TSV Format — Critical Rules

`deck/notes.tsv` is the source of truth. Edits here must be exact.

### Column order (11 fields, tab-separated)

| # | Field | Description |
|---|---|---|
| 1 | `id` | Unique stable identifier — **never change** |
| 2 | `hanzi` | Simplified Chinese |
| 3 | `pinyin` | With tone diacritics |
| 4 | `meaning` | English, short/dictionary style |
| 5 | `example-hanzi` | Example sentence in hanzi |
| 6 | `example-pinyin` | Example sentence in pinyin (tone marks) |
| 7 | `example-meaning` | Example sentence English meaning |
| 8 | `audio` | Optional: `[sound:ATTS <id>.mp3]` |
| 9 | `audio-example` | Optional: `[sound:ATTS <id>-ex.mp3]` |
| 10 | `image` | Optional: `<img src="filename.png">` |
| 11 | `tags` | Space-separated Anki tags |

### Format rules

- **Tab-separated only** — exactly 10 tab characters per data row (11 fields)
- **No quotes** around any field
- **No trailing spaces**
- Empty fields are allowed but must still be present (tab placeholders)
- Blank lines are allowed for human readability
- One header row at the top

### ID policy

IDs use the format `<POS>-<YYYYMMDD>-<NNNN>`:

- `YYYYMMDD` = creation date of the entry
- `NNNN` = zero-padded sequential counter within that POS+date group
- **Never change an existing ID** — Anki uses IDs to match existing notes during re-import
- **Never reuse an ID** — even if an entry is deleted

Valid POS prefixes: `NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`, `PART`, `CLF`, `PREP`, `Q`

Examples: `NOUN-20251221-0001`, `VERB-20251221-0003`, `ADJ-20251221-0012`

### Tags policy

Tags live in the `tags` column as space-separated lowercase tokens:

- Every entry **must** include its POS tag as the final token:
  `noun`, `verb`, `adj`, `adv`, `pron`, `part`, `clf`, `prep`, `question`
- Add one category tag before the POS tag when applicable (e.g., `work noun`, `modal verb`, `degree adv`)
- Keep tags lowercase, avoid duplicates
- Question particles (吗/呢/吧) use `part`; question words (什么/谁/哪) use `question`

### Pinyin rules

- Always use tone diacritics: `péngyou`, `Zhōngguó` — not `pengyou`, `Zhongguo`
- Compounds typically unspaced: `péngyou`, `huǒguō`
- Multi-word phrases spaced: `Zhōngguó rén`, `zhè ge`
- Use apostrophes to avoid ambiguity: `nǚ'ér`

### Hanzi rules

- Simplified Chinese only
- Hanzi must match the pinyin and meaning exactly

### Example sentence rules

- Short, simple, beginner-friendly
- Must use the target word showing typical usage
- Pinyin with full tone marks

---

## Card Template Conventions

Templates use Anki's Mustache-like syntax:

```
{{field-name}}           — field substitution
{{#field}}...{{/field}} — render block if field is truthy
{{^field}}...{{/field}} — render block if field is falsy/empty
{{tts zh_CN:hanzi}}     — TTS fallback (desktop only, not iOS)
[sound:filename.mp3]    — Anki audio player tag
<img src="file.png">    — image display
```

### Platform notes

- **macOS/Windows:** TTS fallback (`{{tts ...}}`) works for Chinese if voices are installed
- **iOS (AnkiMobile):** TTS fallback is **not supported** — pre-generated MP3s are required
- **Preview system** (`preview/preview.js`): strips TTS tags; converts `[sound:X]` → `<audio>` elements; rewrites image paths to `../media/images/`

### CSS conventions

- POS-based color classes: `.noun` (brown), `.verb` (blue), `.adj` (yellow), etc.
- Responsive layout with flexbox
- Supports `.nightMode` for dark theme
- Chinese font stack: Source Han Sans → Noto Sans CJK → system fallback
- Card images: 256×256px source, `max-height: 150px` on card

---

## Anki Import Workflow

1. In Anki: `File → Import` → select `deck/notes.tsv`
2. Settings:
   - Type: **Notes**, Separator: **Tab**
   - Allow HTML in fields: ✅
   - Note Type: `Mandarin (TSV)`
   - Existing notes: **Update existing notes** ✅
3. Field mapping: map `tags` column → **Tags** (not a regular field)
4. Click **Import**

Re-importing with the same IDs is safe and preserves review history and scheduling.

---

## Image Generation Workflow

1. Create `media/sprites/image-data.tsv` — one line per note (ID, hanzi, pinyin, meaning)
2. `npm run sprite:prompt` — generates `media/sprites/sprite-prompt.txt`
3. Paste prompt into ChatGPT → save the output sprite PNG as `media/sprites/image.png`
4. `npm run sprite:slice` — detects grid, crops tiles, removes white background, saves as `media/images/<ID>.png`
5. `npm run sprite:apply-images` — updates `deck/notes.tsv` with `<img src="...">` tags
6. `npm run anki:sync-media:mac` — copies images to Anki's `collection.media/`

---

## AI Assistant Rules

These rules apply when generating or editing any file in this repository. The canonical source is `.github/copilot-instructions.md`.

### When editing `deck/notes.tsv`

- **Never change existing IDs** — this breaks Anki scheduling
- **Never reorder or reuse IDs**
- **Always maintain exactly 10 tab separators** per data row
- Append new rows; do not restructure existing rows
- Validate before outputting: column count, ID format, pinyin diacritics, tag format
- **Output TSV as a TSV code block** — never as a markdown table
- Include header row only if explicitly requested
- Do not invent audio or image filenames — leave fields blank if files don't exist
- When adding entries: allocate sequential `NNNN` within the same POS+date group

### When editing templates (`anki/note-type/`)

- Preserve existing Anki Mustache syntax
- Test logic paths for both truthy and falsy fields
- Do not break the TTS fallback pattern (`{{^audio}}{{tts ...}}{{/audio}}`)

### General

- Do not create new files unless explicitly requested
- Code and scripts must be non-destructive by default
- Cross-reference `deck/notes.schema.md` for field order and rules
- Cross-reference `.github/copilot-instructions.md` for the full canonical ruleset

---

## Linting and Formatting

Prettier is configured (`.prettierrc.json`) for scripts and HTML/CSS files:

```bash
npx prettier --write .
```

No automated test suite exists. Validation is manual via the preview system and Anki import.
