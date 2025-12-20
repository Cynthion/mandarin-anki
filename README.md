# Mandarin Anki Deck

This repository contains the **source of truth** for a custom Mandarin Anki deck.

## Workflow

1. Edit `deck/notes.tsv` in VS Code
2. Import into Anki using the custom note type
3. Learn in Anki
4. Update TSV or Anki fields
5. Re-import with “Update existing notes”

## Design Goals

- Stable IDs
- Git-friendly TSV source
- Optional audio & images
- Real Anki tags
- Card templates versioned in git

## What the Cards show

- **Front:** hanzi → pinyin → audio (file or TTS fallback) → image (optional)
- **Back:** meaning → example (optional) → audio again → image (optional)

To auto-play audio on both sides:

- `Tools → Preferences → Playback → Automatically play audio`

## Setup Guide (Anki)

This guide explains how to create the Anki deck so it matches this repository.

You only need to do this **once** per Anki profile.

---

### Prerequisites

Create a new Anki Deck (e.g. `Default`).

---

### 1. Create a Custom Note Type

1. Open **Anki**
2. Go to **Tools → Manage Note Types**
3. Click **Add**
4. Choose **Add: Basic**
5. Name it (e.g. `Anki Mandarin Deck`)

---

### 2. Configure Fields

With the new note type selected:

1. Click **Fields**
2. Remove all existing fields
3. Add fields **in this exact order**:
   1. id
   2. hanzi
   3. pinyin
   4. meaning
   5. example
   6. audio
   7. image

⚠️ **Important**

- The `id` field must stay stable forever
- Do not add a `tags` field — tags are handled separately by Anki

Click **Save**.

---

### 3. Configure Card Templates

Select the note type → click **Cards**

#### Front Template

Copy the content of `anki/note-type/front.html` and paste it into the **Front Template** editor.

#### Back Template

Copy the content of `anki/note-type/back.html` and paste it into the **Back Template** editor.

#### Styling

Copy the content of `anki/note-type/style.css` and paste it into the **Styling** editor.

Click **Save**.

---

### 4. (Optional) Enable Text-to-Speech

The templates use Anki’s built-in TTS as a fallback when no audio file is provided.

This requires:

- Anki 2.1.50+
- A system TTS voice for Chinese (`zh_CN`)

If TTS does not play:

- Remove the `{{tts zh_CN:hanzi}}` lines from the templates
- Or install the **AwesomeTTS** add-on instead

---

### 5. Import the TSV deck

1. In Anki, click **Import File**
2. Select `deck/notes.tsv`

In the **Import dialog**:

- Field separator: `Tab`
- Allow HTML in fields: ✅
- Note Type: `Anki Mandarin Deck`
- Deck: `Default`
- Existing notes: `Update`
- Field Mapping:

| TSV Column | Anki Field |
|------------|------------|
| id         | id         |
| hanzi      | hanzi      |
| pinyin     | pinyin     |
| meaning    | meaning    |
| example    | example    |
| audio      | audio      |
| image      | image      |
| tags       | **Tags**   |

⚠️ Make sure:

- The `tags` column is mapped to **Tags**, not to a field

Leave the rest at defaults.

Click **Import**.

---

### 6. Re-importing changes safely

You can safely:

- Fix typos
- Add examples
- Add audio or images
- Change templates or CSS

As long as:

- The `id` field never changes
- You re-import with **Update existing notes** enabled

Your review history and scheduling will remain intact.

---

### 7. Audio & image files

If you use audio or images:

- Place files in:
  - `media/audio/`
  - `media/images/`
- Copy them into Anki’s `collection.media` directory
- Reference them in TSV as:
  - Audio: `[sound:filename.mp3]`
  - Image: `<img src="filename.png">`

---

### 8. Recommended workflow

`VS Code → notes.tsv → Anki Import → Review → Edit → Re-import`

This repository remains the **source of truth**.
