import argparse
import os
import shutil
import sys
from pathlib import Path

def sync_dir(src: Path, dst: Path, mode: str):
    if not src.exists():
        print(f"Source does not exist: {src}")
        return

    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name

        if target.exists() or target.is_symlink():
            continue  # do not overwrite existing media

        if mode == "symlink":
            try:
                os.symlink(item.resolve(), target)
                print(f"Symlinked: {item.name}")
            except OSError as e:
                print(f"Failed to symlink {item.name}: {e}")
        else:
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
            print(f"Copied: {item.name}")

def main():
    parser = argparse.ArgumentParser(description="Sync Anki media from repo to collection.media")
    parser.add_argument(
        "--anki-media",
        required=True,
        help="Path to Anki collection.media folder",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to sync files (default: copy)",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    audio_src = repo_root / "media" / "audio"
    image_src = repo_root / "media" / "images"

    anki_media = Path(args.anki_media)

    print(f"Repo root: {repo_root}")
    print(f"Anki media: {anki_media}")
    print(f"Mode: {args.mode}")
    print()

    sync_dir(audio_src, anki_media, args.mode)
    sync_dir(image_src, anki_media, args.mode)

    print("\nDone.")

if __name__ == "__main__":
    main()
