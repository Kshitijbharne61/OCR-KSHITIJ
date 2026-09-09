import subprocess
import sys
from pathlib import Path

# ============================================================
# OCR -> EXTRACTION PIPELINE
# ============================================================
# input_images -> ocr_folder.py -> output_text -> extractor.py -> extracted

ROOT = Path(__file__).resolve().parent
OCR_SCRIPT = ROOT / "ocr_folder.py"


def run_ocr():
    print(f"\n{'=' * 60}")
    print("STEP 1: RUNNING OCR")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, str(OCR_SCRIPT)],
        cwd=str(ROOT),
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(
            f"\nERROR: OCR failed with exit code {result.returncode}."
        )


def run_extraction():
    print(f"\n{'=' * 60}")
    print("STEP 2: EXTRACTING FIELDS FROM OCR TEXT")
    print(f"{'=' * 60}\n")

    # Import the existing extractor and redirect its folders.
    # This avoids copying files or maintaining a second extractor.
    import extractor

    extractor.INPUT_FOLDER = ROOT / "output_text"
    extractor.OUTPUT_FOLDER = ROOT / "extracted"
    extractor.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    extractor.main()


def main():
    input_images = ROOT / "input_images"
    output_text = ROOT / "output_text"
    extracted = ROOT / "extracted"

    input_images.mkdir(exist_ok=True)
    output_text.mkdir(exist_ok=True)
    extracted.mkdir(exist_ok=True)

    if not any(input_images.iterdir()):
        raise SystemExit(
            "No product images found in input_images/. "
            "Put your product images there and run pipeline.py again."
        )

    run_ocr()
    run_extraction()

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"OCR text : {output_text}")
    print(f"Extracted: {extracted}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
