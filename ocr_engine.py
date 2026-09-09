"""Production-oriented PaddleOCR runner for packaged-product images.

Keeps OCR text, confidence, and bounding boxes so downstream compliance checks
can explain where a declaration came from.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import cv2

os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input_images"
OUTPUT_DIR = ROOT / "output_text"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("％", "%").replace("﹪", "%")
    text = re.sub(r"(?i)\bINR\s*(?=\d)", "₹", text)
    text = re.sub(r"(?i)\bRs\.?\s*(?=\d)", "₹", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    h, w = image.shape[:2]
    if max(h, w) < 1600:
        scale = 1600 / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Keep colour for OCR; use CLAHE only on luminance to improve small-print contrast.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def result_to_dict(result: Any) -> dict:
    data = result.json if hasattr(result, "json") else result
    if callable(data):
        data = data()
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    return data if isinstance(data, dict) else {}


def extract_regions(result: Any) -> list[dict]:
    regions: list[dict] = []
    data = result_to_dict(result)
    texts = data.get("rec_texts", []) or []
    scores = data.get("rec_scores", []) or []
    boxes = data.get("rec_boxes", []) or data.get("dt_polys", []) or []

    for i, text in enumerate(texts):
        text = normalize_text(str(text))
        if not text:
            continue
        score = float(scores[i]) if i < len(scores) else 0.0
        box = boxes[i] if i < len(boxes) else None
        bbox = None
        if box is not None:
            try:
                pts = box.tolist() if hasattr(box, "tolist") else box
                if len(pts) == 4 and isinstance(pts[0], (list, tuple)):
                    xs = [float(p[0]) for p in pts]
                    ys = [float(p[1]) for p in pts]
                    bbox = {
                        "x": int(min(xs)), "y": int(min(ys)),
                        "width": int(max(xs) - min(xs)),
                        "height": int(max(ys) - min(ys)),
                    }
                elif len(pts) >= 4 and isinstance(pts[0], (int, float)):
                    bbox = {"x": int(pts[0]), "y": int(pts[1]), "width": int(pts[2]), "height": int(pts[3])}
            except Exception:
                bbox = None
        regions.append({"text": text, "confidence": round(score * 100, 2), "bbox": bbox})
    return regions


def build_ocr() -> PaddleOCR:
    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def process_image(ocr: PaddleOCR, image_path: Path) -> dict:
    processed = preprocess_image(image_path)
    temp = OUTPUT_DIR / f"._{image_path.stem}_preprocessed.jpg"
    cv2.imwrite(str(temp), processed)
    try:
        result = ocr.predict(str(temp))
        regions = extract_regions(result)
    finally:
        temp.unlink(missing_ok=True)

    # Preserve reading order while removing exact duplicate OCR lines.
    seen = set()
    ordered = []
    for region in regions:
        key = region["text"].casefold()
        if key not in seen:
            seen.add(key)
            ordered.append(region)

    text = "\n".join(r["text"] for r in ordered)
    confidence = round(sum(r["confidence"] for r in ordered) / len(ordered), 2) if ordered else 0.0
    return {
        "image": image_path.name,
        "ocr_confidence": confidence,
        "regions": ordered,
        "text": text,
    }


def run() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit("No images found in input_images/. Add product images and run pipeline.py again.")

    ocr = build_ocr()
    # Clear only generated OCR files; never delete user images.
    for p in OUTPUT_DIR.glob("*.txt"):
        p.unlink()
    for p in OUTPUT_DIR.glob("*_regions.json"):
        p.unlink()

    for image in images:
        print(f"OCR: {image.name}")
        result = process_image(ocr, image)
        (OUTPUT_DIR / f"{image.stem}.txt").write_text(result["text"], encoding="utf-8")
        (OUTPUT_DIR / f"{image.stem}_regions.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  regions={len(result['regions'])} confidence={result['ocr_confidence']:.2f}%")


if __name__ == "__main__":
    run()
