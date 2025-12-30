# Mandarin Anki Deck

This repository contains the **source of truth** for a custom Mandarin Anki deck.

- Notes are stored in a Git-friendly TSV file: `deck/notes.tsv`
- Card templates + CSS are versioned in Git under: `anki/note-type/`
- Media (optional) can be stored under `media/` and copied into Anki’s `collection.media`
- Preview your card templates in `preview/`

<img src="./assets/preview.png" width="600">

## Design Goals

- Stable, unique IDs (safe re-import without losing scheduling)
- Git-friendly TSV source (easy to edit in VS Code)
- Real Anki tags (via the TSV `tags` column → mapped to Anki “Tags”)
- Optional audio & images
- Card templates and CSS versioned in Git
- Works on macOS, Windows, and iOS (with platform notes below)

---

## Workflow

1. Edit `deck/notes.tsv` in VS Code
2. Import into Anki using the custom note type
3. Learn in Anki
4. Update TSV (or optionally edit fields in Anki)
5. Re-import with “Update existing notes” enabled

> This repository remains the **source of truth**.  
> If you edit notes in Anki, export/merge changes back into `deck/notes.tsv` to keep Git authoritative.

---

## Setup Guide (Anki)

You only need to do this **once per Anki profile**.

### Prerequisites

- Create (or choose) an Anki deck to import into (e.g. `Mandarin`).

---

## Create a Custom Note Type

1. Open Anki
2. `Tools → Manage Note Types`
3. Click **Add**
4. Choose **Add: Basic**
5. Name it: `Mandarin (TSV)` (or similar)

> Note: We are not using “Basic (and reversed card)”.  
> We keep a single note type and control front/back via templates.

---

## Configure Fields

With your note type selected:

1. Click **Fields**
2. Remove all existing fields
3. Add fields in this exact order:

   1. `id`
   2. `hanzi`
   3. `pinyin`
   4. `meaning`
   5. `example-hanzi`
   6. `example-pinyin`
   7. `example-meaning`
   8. `audio`
   9. `audio-example`
   10. `image`

✅ Important:

- Fields are **data containers**, not “front/back”.
- The “Front” and “Back” are defined in the **Cards** templates.
- Do **not** add a `tags` field: tags are handled separately by Anki during import.

Click **Save**.

---

## Configure Card Templates + Styling

1. Select your note type
2. Click **Cards…**
3. Rename the `Card 1` to `Mandarin -> English` (production)
4. Add a new card and name it `English -> Mandarin` (recognition)
5. Paste templates from this repo:

`Mandarin -> English` card:
- Front Template: `anki/note-type/front1.html`
- Back Template: `anki/note-type/back1.html`
- Styling: `anki/note-type/style.css`

`English -> Mandarin` card:
- Front Template: `anki/note-type/front2.html`
- Back Template: `anki/note-type/back2.html`
- Styling:  `anki/note-type/style.css`

Click **Save**.

---

## System TTS Fallback

The templates include a fallback when no audio file exists:

- If `audio` is present: play `[sound:...]`
- Else: use Anki’s built-in `{{tts ...}}`

### Platform notes

- **macOS:** usually works out-of-the-box (install Chinese voices if needed)
- **Windows:** you must install a Chinese TTS voice (see Troubleshooting)
- **iOS:** AnkiMobile does **not** support Anki’s `{{tts ...}}` tag.
  - iOS will only play audio if you provide `[sound:...]` files.

---

## Import the TSV Deck

1. In Anki: `File → Import`
2. Select `deck/notes.tsv`

In the import dialog:

- Type: **Notes**
- Separator: **Tab**
- “Allow HTML in fields”: ✅
- Note Type: `Mandarin (TSV)`
- Deck: your deck (e.g. `Mandarin`)
- Existing notes: ✅ **Update existing notes**
- Match scope: (default is fine)
- “Fields separated by”: Tab

### Field mapping

Map TSV columns like this:

| TSV Column      | Map to          |
|-----------------|-----------------|
| id              | id              |
| hanzi           | hanzi           |
| pinyin          | pinyin          |
| meaning         | meaning         |
| example-hanzi   | example-hanzi   |
| example-pinyin  | example-pinyin  |
| example-meaning | example-meaning |
| audio           | audio           |
| audio-example   | audio-example   |
| image           | image           |
| tags            | **Tags**        |

✅ Important:

- The `tags` column must map to **Tags**, not to a field.

Click **Import**.

---

## Re-importing Changes Safely

Safe changes:

- Fix typos
- Update meanings/examples
- Add tags
- Add audio/image references
- Update templates/CSS

Rules:

- The `id` field must never change
- Always re-import with **Update existing notes**
- Do not create duplicates by importing without matching IDs

Your review history and scheduling will remain intact.

---

## Media (Audio & Images)

### Repo layout (optional)

- `media/audio/` — mp3 files
- `media/images/` — image files

### Anki media folder

Media must be copied into Anki’s profile media directory:

- **macOS:** `~/Library/Application Support/Anki2/<Profile>/collection.media`
- **Windows:** `%APPDATA%\Anki2\<Profile>\collection.media`

### TSV references

- Audio:
  - `[sound:ATTS <ID>.mp3]`
- Image:
  - `<img src="ni.png">`

> Do not use external URLs.

### Sync media from repo → Anki (recommended)

Anki can only play/show media files that exist inside your profile’s `collection.media/`.

This repo stores media under:

- `media/audio/`
- `media/images/`

Use the provided sync script to copy (or symlink) repo media into Anki’s local `collection.media` folder.

#### Find your Anki `collection.media` path

- **macOS:**
  - `~/Library/Application Support/Anki2/<Profile>/collection.media`
- **Windows:**
  - `%APPDATA%\Anki2\<Profile>\collection.media`
  - Example: `C:\Users\<You>\AppData\Roaming\Anki2\User 1\collection.media`

> `<Profile>` is usually `User 1` unless you renamed it.

#### Run the sync script (copy mode, safest)

Copy new files from `media/audio/` and `media/images/` without overwriting existing media.

From the repo root:

**npm Scripts:**

```bash
npm run anki:sync-media:mac
npm run anki:sync-media:windows
```
**macOS / Linux (Terminal):**

```bash
python3 scripts/sync_media_to_anki.py --anki-media "$HOME/Library/Application Support/Anki2/User 1/collection.media"
```

**Windows (PowerShell):**

```powershell
python scripts\sync_media_to_anki.py --anki-media "$env:APPDATA\Anki2\User 1\collection.media"
```

---

## Optional: Generate Audio with AwesomeTTS

If you want iOS-compatible audio, generate mp3 files and write them into the `audio` field.

Recommended:

- Configure `AwesomeTTS` to name files using the `id` field, resulting in:
  - `collection.media/<id>.mp3` (`Tools → AwesomeTTS → MP3 → Filename Template → {{id}}.mp3`)
  - `notes.tsv`: `audio` field like `[sound:ATTS <id>.mp3]`

<img src="./assets/awesometts-settings.png" width="300">
<img src="./assets/awesometts-add-audio.png" width="300">

---

## Optional: Generate Images with Generative AI

This repo supports batch-generated, consistent card images using a sprite-sheet workflow.

### Overview

- List the notes you want images for in `image-data.tsv`
- Generate one sprite sheet with ChatGPT
- Automatically slice it into transparent 256×256 PNGs
- Images end up in media/images/, named by note ID

1) Prepare `image-data.tsv`

This should be done by categories, e.g. `people_family`, `animals`, etc.
Add one line per note (only the first column is required):

```tsv
NOUN-20251221-0001	人	rén	person
NOUN-20251221-0002	中国人	Zhōngguó rén	Chinese person
NOUN-20251221-0003	爸爸	bàba	father
```

Order matters: images are sliced in this order.

2) Generate the Sprite Prompt

`npm run sprite:prompt`

- This creates `sprite-prompt.txt`
- Copy the file contents
- Paste it into ChatGPT (without the JSON debug info)
- Generate one single PNG sprite sheet
- Save it with the category as name, e.g.`media/sprites/people_family.png`

1) Slice the Sprite into Images

Update the image name in the `sprite:slice` script if needed, e.g. `media/sprites/people_family.png`, then run:

`npm run sprite:slice`

This will:

- Detect the grid automatically
- Crop each tile
- Remove the white background
- Fix edge halos
- Save images as transparent PNGs: `media/images/<NOTE_ID>.png`
- A debug image with crop boxes is written to: `media/sprites/<NOTE_ID>.debug.png`

4) Apply Images to Notes

Automatically add `<img ...>` tags to the `image` field in `deck/notes.tsv` by running:

`npm run sprite:apply-images`

5) Sync Media to Anki

Run the media sync script from above to copy new images into Anki’s `collection.media/`.

`npm run anki:sync-media`

6) Notes

- The art style stays consistent across batches
- You can regenerate sprites anytime
- Existing images are overwritten safely
- Works across light/dark Anki themes

---

## Optional: Template Styling & Live Preview

This repository is not only the **source of truth for note data**, but also the **source of truth for card styling**.

You can fully design, test, and iterate on your Anki card templates **outside of Anki**, using a live preview in VS Code or any browser.

The preview system renders:

- `anki/note-type/front.html`
- `anki/note-type/back.html`
- `anki/note-type/style.css`

using **mock note data**, closely matching how Anki renders cards.

This allows you to:
- adjust layout, spacing, fonts, and colors
- test POS coloring (`noun`, `verb`, `adj`, etc.)
- verify conditional blocks (`{{#audio}}`, `{{#image}}`, etc.)
- preview example sentences and media placement

> Note: Anki-only features like `{{tts ...}}` and `[sound:...]` are intentionally ignored in preview.

Run the preview in the repository root with:

```bash
npm install
npm run preview
```

Then open `http://localhost:5173/preview/` in your browser.
 
---

## AnkiWeb Sync (Across Devices)

- Create a free AnkiWeb account
- Sync from Anki Desktop
- Log in on mobile devices and sync
- AnkiWeb syncs:
  - Review progress
  - Decks and notes
  - Media files

---

## Troubleshooting

### “no players found for TTSTag(...)” on Windows

Windows has no TTS voice available for Chinese.

Fix:

1. Windows Settings → Time & Language
2. Add Chinese (Simplified, China) language pack
3. Install Speech/Voice components
4. Restart Anki

### iOS has no TTS fallback

This is expected. Use generated mp3 files for iOS playback.

---

## Recommended Workflow Summary

`VS Code → deck/notes.tsv → Anki Import → Review → Update TSV → Re-import`

Templates/CSS live in Git and should be pasted into the note type as needed.
