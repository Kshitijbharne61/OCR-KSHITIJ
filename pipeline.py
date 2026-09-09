from pathlib import Path

# SIH26034 end-to-end pipeline
# input_images -> PaddleOCR -> output_text -> keyword/regex extraction -> compliance_reports

ROOT = Path(__file__).resolve().parent


def run_ocr():
    print(f"\n{'=' * 64}\nSTEP 1: PADDLEOCR\n{'=' * 64}")
    import ocr_engine
    ocr_engine.INPUT_DIR = ROOT / "input_images"
    ocr_engine.OUTPUT_DIR = ROOT / "output_text"
    ocr_engine.run()


def run_extraction():
    print(f"\n{'=' * 64}\nSTEP 2: FIELD EXTRACTION\n{'=' * 64}")
    import extractor
    extractor.KEYWORDS_FOLDER = str(ROOT / "keywords")
    extractor.INPUT_FOLDER = ROOT / "output_text"
    extractor.OUTPUT_FOLDER = ROOT / "extracted"
    extractor.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    extractor.main()


def run_compliance():
    print(f"\n{'=' * 64}\nSTEP 3: SIH26034 COMPLIANCE SCREENING\n{'=' * 64}")
    import compliance_engine
    compliance_engine.ROOT = ROOT
    compliance_engine.main()


def main():
    input_images = ROOT / "input_images"
    output_text = ROOT / "output_text"
    extracted = ROOT / "extracted"
    reports = ROOT / "compliance_reports"

    input_images.mkdir(exist_ok=True)
    output_text.mkdir(exist_ok=True)
    extracted.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    images = [p for p in input_images.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}]
    if not images:
        raise SystemExit("No product images found in input_images/. Add one or more product images and run: python pipeline.py")

    run_ocr()
    run_extraction()
    run_compliance()

    print(f"\n{'=' * 64}\nPIPELINE COMPLETED SUCCESSFULLY\n{'=' * 64}")
    print(f"OCR text       : {output_text}")
    print(f"Extracted JSON : {extracted}")
    print(f"Compliance     : {reports}")


if __name__ == "__main__":
    main()
