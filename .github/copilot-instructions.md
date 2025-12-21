# Copilot Instructions (Repository Rules)

This repository contains a custom Mandarin Anki deck maintained via:
- a TSV source file (`deck/notes.tsv`)
- optional media assets (`media/audio/`, `media/images/`)
- versioned note type templates (`anki/note-type/`)

Copilot must follow these rules when generating or editing content.

---

## 1) Source of truth

- `deck/notes.tsv` is the source of truth for note content.
- Do not reorder, renumber, or reuse IDs unless explicitly instructed.
- Changes should be stable and should not create duplicates.

---

## 2) TSV format (strict)

`deck/notes.tsv` columns are exactly:

id	hanzi	pinyin	meaning	example	audio	image	tags

Rules:
- Tab-separated values only (TSV). No commas/semicolons as separators.
- Keep a single header row.
- Every data row must have exactly **8 tab separators** (9 fields total).
- Do not introduce quotes around fields.
- Avoid trailing spaces.
- Blank lines are allowed for human readability, but not required.

---

## 3) ID policy (critical)

IDs must be unique and stable forever.

### Canonical ID format
`<POS>-<YYYYMMDD>-<NNNN>`

Examples:
- `NOUN-20251221-0001`
- `VERB-20251221-0003`
- `ADJ-20251221-0012`
- `ADV-20251221-0020`
- `PRON-20251221-0006`
- `PART-20251221-0004`
- `CLF-20251221-0009`
- `PREP-20251221-0008`
- `Q-20251221-0002`

Rules:
- `<POS>` must be one of: NOUN, VERB, ADJ, ADV, PRON, PART, CLF, PREP, Q
- `YYYYMMDD` is the creation date of the entry
- `NNNN` is a zero-padded counter within that POS+date group
- Never change existing IDs; never reuse IDs
- When adding multiple entries, allocate sequential `NNNN`

---

## 4) Tags policy (Anki tags)

Tags live in the TSV `tags` column as space-separated tokens (no commas).
During import, this column is mapped to Anki “Tags” (not a field).

### Must-have POS tags
Every entry must include its POS tag as the final tag token:

- noun, verb, adj, adv, pron, part, clf, prep, question

### Category tags
Add one category tag before the POS tag when applicable.

Examples:
- `work noun`
- `modal verb`
- `degree adv`
- `question` (question words)
- `clf` (classifiers)
- `prep` (prepositions/coverbs)
- `part` (particles)

Rules:
- Keep tags lowercase
- Avoid duplicates
- Question particles (吗/呢/吧) are `part`
- Question words (什么/谁/哪个/几) are `question` (category optional)

---

## 5) Pinyin rules

- Use tone marks (diacritics): `Zhōngguó`, not `Zhongguo`.
- Use spacing consistently:
  - Compounds typically unspaced: `péngyou`, `huǒguō`
  - Multi-word phrases spaced: `Zhōngguó rén`, `zhè ge`, `wǒmen de`
- Use apostrophes where needed to avoid ambiguity (e.g., `nǚ’ér`).

---

## 6) Meaning rules

- Meanings are in English.
- Keep meanings short, dictionary style.
- Prefer consistent wording across the deck.

---

## 7) Example sentence rules

- Examples must be short, simple, beginner-friendly.
- Examples are in **pinyin with tone marks**.
- Should match the part of speech and show typical usage.

Special cases:
- Coverbs/prepositions (给/到/用/etc.) must show coverb use clearly.
- If a word can be multiple POS (e.g., 在 verb vs prep), only create separate entries if explicitly requested.

---

## 8) Hanzi rules

- Use simplified Chinese.
- Ensure hanzi matches meaning and pinyin.
- For classifiers/measure words, use correct hanzi (个/本/只/张/杯...).

---

## 9) Media fields (audio/image)

Media fields are optional.

### Audio field
- Use Anki sound tags:
  - `[sound:filename.mp3]`
- Prefer ID-based filenames when configured:
  - `[sound:<id>.mp3]`
- Do not invent filenames unless explicitly instructed.
- Do not include external URLs.

### Image field
- Use HTML:
  - `<img src="filename.png">`
- Files should exist under repo media folder if referenced.

---

## 10) Editing discipline

When asked to “add entries”:
- Append new rows (or place near related entries for readability) but never change existing IDs.
- Ensure no duplicate IDs.
- Ensure every row has all 9 fields.

When asked to “fix the TSV”:
- Validate: column count, ID format, tags, pinyin diacritics, beginner-appropriate examples.

---

## 11) Output expectations

- TSV content must be emitted as a TSV code block.
- Include header only if requested; otherwise output only the new rows to paste.
- Do not output markdown tables for TSV content.
- Code/scripts should be repo-focused and non-destructive unless explicitly requested.
