# Notes schema

## Field order (important)

1. id (unique, stable, never change)
2. hanzi
3. pinyin
4. meaning
5. example
6. audio (optional, [sound:xxx.mp3])
7. image (optional, HTML `<img>`)
8. tags (space-separated, mapped to Anki tags)

## Rules

- IDs must never change once imported
- Tags are real Anki tags, not a display field
- Audio/image files must exist in Anki collection.media
- HTML is allowed in `image`
