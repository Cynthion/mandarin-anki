# Copilot Instructions (Repository Rules)

This repository contains an Anki deck maintained via a TSV source file (`notes.tsv`) and media assets (images/audio).
Copilot must follow these rules when generating or editing content.

---

## 1) Source of truth

- `notes.tsv` is the source of truth for notes content.
- Do not reorder or renumber IDs unless explicitly instructed.
- Changes should be additive and stable: updates should not create duplicates.

---

## 2) TSV format (strict)

`notes.tsv` columns are exactly:

id	hanzi	pinyin	meaning	example	audio	image	tags

Rules:
- Tab-separated values only (TSV). No commas, no semicolons, no extra columns.
- Keep a single header row.
- Every row must have exactly 8 tab separators (9 fields total).
- Keep `audio` and `image` fields empty unless explicitly provided.
- Do not introduce quotes around fields.
- Do not include trailing spaces.

---

## 3) ID policy (critical)

IDs must be unique and stable.

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
- `<POS>` must match the part-of-speech group:
  - NOUN, VERB, ADJ, ADV, PRON, PART, CLF, PREP, Q
- `YYYYMMDD` is the creation date of the entry.
- `NNNN` is a zero-padded counter within that POS+date group.
- Never reuse IDs, never change existing IDs.
- When adding multiple entries at once, allocate sequential `NNNN`.

---

## 4) Tags policy

Tags live in the `tags` column as space-separated tokens (no commas).

### Must-have POS tags
Every entry must include its POS tag as the final tag token:
- `noun`, `verb`, `adj`, `adv`, `pron`, `part`, `clf`, `prep`, `question`

### Category tags
Include one category tag before the POS tag when applicable:
- Nouns: `family`, `people`, `animal`, `place`, `geography`, `food`, `object`, `work`, `time`, `money`, `measure`, `number`, etc.
- Verbs: `core`, `motion`, `daily`, `modal`, `mental`, `emotion`, `communication`, `perception`, `commerce`, `social`, `origin`, `learning`, `life`, etc.
- Adjectives: `feeling`, `evaluation`, `state`, `quality`, `temperature`, `food`, `appearance`, `size`, `trait`, `quantity`, `utility`, etc.
- Adverbs: `degree`, `time`, `negation`, `scope`, `manner`, `sequence`, `attitude`, `discourse`, `question`, etc.

Example tags:
- `work noun`
- `modal verb`
- `degree adv`
- `question`
- `clf`
- `prep`
- `part`

Rules:
- Keep tags lowercase.
- Avoid duplicates (e.g., don’t add `question` twice).
- For question particles (吗/呢/吧), tag as `part` (not `adv`).
- Question words like 什么/谁/哪个/几 should be tagged `question` (category optional).

---

## 5) Pinyin rules

- Use tone marks (diacritics) for all syllables: `Zhōngguó`, not `Zhongguo`.
- Use proper spacing for multi-word pinyin:
  - Compounds that are one word in Chinese usually remain together: `péngyou`, `huǒguō`
  - Phrases can be spaced: `Zhōngguó rén`, `zhè ge`, `wǒmen de`
- Use `nǚ’ér` with apostrophe where needed to separate syllables; otherwise standard spacing.

---

## 6) Meaning rules

- Meanings are in English.
- Keep meanings short and “dictionary style”.
- Prefer consistent wording across the deck (e.g., “mobile phone” not sometimes “cellphone”).

---

## 7) Example sentence rules

- Examples must be short, simple, and appropriate to the word’s part of speech.
- Examples should be in pinyin with tone marks.
- Keep examples beginner-friendly (A1/A2 style).
- Avoid overly complex grammar unless explicitly requested.

Special cases:
- Prepositions/coverbs (e.g., 给/到/用) must have examples that clearly show coverb use:
  - `Wǒ gěi nǐ mǎi.` (for you)
  - `Wǒ dào xuéxiào qù.` (to ...)
  - `Wǒ yòng shǒujī kàn.` (using ...)
- If a word has multiple roles (e.g., 在 as verb vs prep), maintain separate entries only when both are desired, and make examples clearly distinguish them.

---

## 8) Hanzi rules

- Use simplified Chinese characters.
- Ensure the hanzi matches the intended meaning and pinyin.
- For measure words/classifiers, use the correct hanzi (e.g., 个/本/只/张/杯...).

---

## 9) Media fields (audio/image)

- `audio` and `image` fields are optional and may be blank.
- If adding an image, use HTML:
  - `<img src="filename.png">`
- Image/audio files must exist under the repo’s media location (if defined); do not invent filenames.
- Do not add external URLs.

---

## 10) Editing discipline

When asked to “add entries”:
- Append new lines (or place them near related entries for human readability) but do not change existing IDs.
- Ensure no duplicate IDs.
- Ensure every row has all fields.

When asked to “fix the TSV”:
- Validate: correct column count, correct IDs format, correct tags, correct pinyin tone marks, correct examples.

---

## 11) Output expectations

When generating TSV content:
- Output in a TSV code block.
- Include header only if requested; otherwise output just the new lines to paste.
- Never output markdown tables for TSV content.

When generating code/scripts:
- Keep them small and repo-focused.
- Do not propose destructive edits (bulk renumbering, mass deletion) unless explicitly requested.

---
