import subprocess
import sys
from pathlib import Path

# Run OCR first, then automatically run extraction.
# This connects:
# input_images -> output_text -> extracted

ROOT = Path(__file__).resolve().parent
OCR_SCRIPT = ROOT / "ocr_folder.py"
EXTRACTOR_SCRIPT = ROOT / "extractor.py"


def run_script(script):
    print(f"\n{'=' * 60}")
    print(f"Running: {script.name}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(
            f"\nERROR: {script.name} failed with exit code {result.returncode}."
        )


def main():
    required = [OCR_SCRIPT, EXTRACTOR_SCRIPT]
    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise SystemExit("Missing required file(s):\n" + "\n".join(missing))

    (ROOT / "input_images").mkdir(exist_ok=True)
    (ROOT / "output_text").mkdir(exist_ok=True)
    (ROOT / "extracted").mkdir(exist_ok=True)

    run_script(OCR_SCRIPT)
    run_script(EXTRACTOR_SCRIPT)

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print("OCR text:  output_text/")
    print("Extracted: extracted/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
