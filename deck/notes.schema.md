# Notes schema

## Field order (important)

1. id (unique, stable, never change)
2. hanzi
3. pinyin
4. meaning
5. example-hanzi
6. example-pinyin
7. example-meaning
8. audio (optional, [sound:<id>.mp3])
9. audio-example (optional, [sound:<id>-ex.mp3])
10. image (optional, HTML `<img>`)
11. tags (space-separated, mapped to Anki tags)

## Rules

- IDs must never change once imported
- Tags are real Anki tags, not a display field
- Audio/image files must exist in Anki collection.media
- HTML is allowed in `image`
