"""One-command OCR runner for OCR-KSHITIJ.

Usage:
    python run_ocr.py
    python run_ocr.py path/to/product.jpg

If an image path is supplied, the image is copied into input_images/.
Then ocr_engine.py processes it and writes TXT + JSON into output_text/.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input_images"

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 2:
        print("Usage: python run_ocr.py [image_path]")
        return 2

    if len(sys.argv) == 2:
        source = Path(sys.argv[1]).expanduser().resolve()
        if not source.is_file():
            print(f"ERROR: Image not found: {source}")
            return 1
        if source.suffix.lower() not in SUPPORTED:
            print(f"ERROR: Unsupported image type: {source.suffix}")
            return 1

        destination = INPUT_DIR / source.name
        shutil.copy2(source, destination)
        print(f"Image added: {destination.relative_to(ROOT)}")

    images = [p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not images:
        print("ERROR: No image found. Upload/copy an image to input_images/ or pass its path.")
        return 1

    print("Starting OCR...")
    result = subprocess.run([sys.executable, str(ROOT / "ocr_engine.py")], cwd=ROOT)
    if result.returncode != 0:
        return result.returncode

    print("\nOCR complete.")
    print("TXT + JSON files are in: output_text/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
