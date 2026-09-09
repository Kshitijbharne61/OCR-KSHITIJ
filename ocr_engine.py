"""High-accuracy product-label OCR for SIH26034.

Input : input_images/
Output: output_text/<image>.txt
        output_text/<image>.json
        output_text/<image>_regions.json

Design goals:
- PaddleOCR is the primary engine.
- Multiple image preprocessing passes are used to improve small/low-contrast text.
- OCR regions, bounding boxes and confidence are preserved for downstream evidence.
- Results from passes are merged and exact/near duplicates are removed.
- OCR failures are reported clearly instead of silently producing empty files.
- No claim of 100% accuracy: low-confidence text is explicitly marked for review.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Disable the problematic oneDNN/MKLDNN path on CPU installations.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

import cv2
import numpy as np
from paddleocr import PaddleOCR

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input_images"
OUTPUT_DIR = ROOT / "output_text"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Configuration through environment variables keeps the code usable on both
# a local Windows laptop and a future SIH backend/container.
OCR_LANG = os.getenv("OCR_LANG", "en")
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.20"))
OCR_MAX_PASSES = max(1, int(os.getenv("OCR_MAX_PASSES", "5")))
OCR_SCALE_MIN = int(os.getenv("OCR_SCALE_MIN", "1800"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SIH26034-OCR")


@dataclass
class OCRRegion:
    text: str
    confidence: float
    bbox: list[int] | None
    pass_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox,
            "pass": self.pass_name,
        }


def normalize_text(text: str) -> str:
    """Conservative normalization: do not alter characters that may be legal label data."""
    text = str(text).replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("％", "%").replace("﹪", "%")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def canonical_text(text: str) -> str:
    """Canonical form used only for duplicate comparison."""
    value = normalize_text(text).casefold()
    value = re.sub(r"[^\w₹@.%+\-/ ]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def preprocess_variants(path: Path) -> list[tuple[str, np.ndarray]]:
    """Create several OCR-friendly variants without destroying the original image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    h, w = image.shape[:2]
    longest = max(h, w)
    scale = max(1.0, OCR_SCALE_MIN / max(1, longest))
    if scale > 1.0:
        base = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        base = image.copy()

    variants: list[tuple[str, np.ndarray]] = [("original", base)]

    # Luminance contrast enhancement while retaining colour information.
    lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    clahe_img = cv2.cvtColor(cv2.merge((enhanced_l, a, b)), cv2.COLOR_LAB2BGR)
    variants.append(("clahe", clahe_img))

    gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
    # Mild denoise + unsharp masking preserves character edges.
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpened = cv2.addWeighted(gray, 1.6, denoised, -0.6, 0)
    variants.append(("sharp_gray", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)))

    # Adaptive threshold helps very uneven packaging backgrounds.
    adaptive = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11,
    )
    variants.append(("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)))

    # Otsu is useful for clean high-contrast printed labels.
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))

    return variants[:OCR_MAX_PASSES]


def build_ocr() -> PaddleOCR:
    """Construct one reusable PaddleOCR instance."""
    kwargs = {
        "lang": OCR_LANG,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": False,
    }
    try:
        # Explicitly select PaddlePaddle for local inference when supported.
        return PaddleOCR(engine="paddle", **kwargs)
    except TypeError:
        # Compatibility with PaddleOCR releases where engine is not accepted.
        return PaddleOCR(**kwargs)


def result_to_dict(result: Any) -> dict[str, Any]:
    data = result.json if hasattr(result, "json") else result
    if callable(data):
        data = data()
    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    return data if isinstance(data, dict) else {}


def _box_to_xyxy(box: Any) -> list[int] | None:
    if box is None:
        return None
    try:
        pts = box.tolist() if hasattr(box, "tolist") else box
        # Polygon: [[x,y], [x,y], ...]
        if len(pts) >= 4 and isinstance(pts[0], (list, tuple)):
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        # Already xyxy.
        if len(pts) >= 4 and all(isinstance(v, (int, float)) for v in pts[:4]):
            return [int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])]
    except Exception:
        pass
    return None


def extract_regions(result: Any, pass_name: str) -> list[OCRRegion]:
    data = result_to_dict(result)
    texts = data.get("rec_texts", []) or []
    scores = data.get("rec_scores", []) or []
    boxes = data.get("rec_boxes", []) or data.get("dt_polys", []) or []

    regions: list[OCRRegion] = []
    for i, raw_text in enumerate(texts):
        text = normalize_text(raw_text)
        if not text:
            continue
        try:
            confidence = float(scores[i]) if i < len(scores) else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < OCR_MIN_CONFIDENCE:
            continue
        bbox = _box_to_xyxy(boxes[i]) if i < len(boxes) else None
        regions.append(OCRRegion(text, confidence, bbox, pass_name))
    return regions


def _iou(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / float(area_a + area_b - inter)


def merge_regions(regions: list[OCRRegion]) -> list[OCRRegion]:
    """Merge exact/near duplicates from different preprocessing passes."""
    # Highest-confidence observations are considered first.
    regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
    merged: list[OCRRegion] = []

    for candidate in regions:
        key = canonical_text(candidate.text)
        if not key:
            continue

        duplicate = False
        for existing in merged:
            existing_key = canonical_text(existing.text)
            if key == existing_key:
                duplicate = True
                break
            # Same location with a very similar string: keep the stronger OCR result.
            if candidate.bbox and existing.bbox and _iou(candidate.bbox, existing.bbox) >= 0.65:
                shorter, longer = sorted((key, existing_key), key=len)
                if shorter and shorter in longer and len(shorter) / max(1, len(longer)) >= 0.75:
                    duplicate = True
                    break
        if not duplicate:
            merged.append(candidate)

    # Reading order: top-to-bottom, then left-to-right.
    merged.sort(key=lambda r: ((r.bbox or [0, 0, 0, 0])[1], (r.bbox or [0, 0, 0, 0])[0]))
    return merged


def process_image(ocr: PaddleOCR, image_path: Path) -> dict[str, Any]:
    variants = preprocess_variants(image_path)
    all_regions: list[OCRRegion] = []
    pass_stats: list[dict[str, Any]] = []

    for pass_name, image in variants:
        logger.info("  OCR pass: %s", pass_name)
        try:
            result = ocr.predict(image)
            regions = extract_regions(result, pass_name)
            all_regions.extend(regions)
            pass_stats.append({
                "pass": pass_name,
                "regions": len(regions),
                "status": "ok",
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("  OCR pass '%s' failed: %s", pass_name, exc)
            pass_stats.append({
                "pass": pass_name,
                "regions": 0,
                "status": "failed",
                "error": str(exc),
            })

    regions = merge_regions(all_regions)
    text = "\n".join(r.text for r in regions)
    confidence = round(float(np.mean([r.confidence for r in regions])) * 100, 2) if regions else 0.0
    high_conf = sum(1 for r in regions if r.confidence >= 0.80)
    low_conf = sum(1 for r in regions if r.confidence < 0.60)

    return {
        "schema_version": "1.0",
        "image": image_path.name,
        "engine": "PaddleOCR",
        "language": OCR_LANG,
        "ocr_confidence_percent": confidence,
        "review_required": bool(low_conf or not regions),
        "statistics": {
            "preprocessing_passes": len(variants),
            "successful_passes": sum(1 for p in pass_stats if p["status"] == "ok"),
            "regions": len(regions),
            "high_confidence_regions": high_conf,
            "low_confidence_regions": low_conf,
        },
        "text": text,
        "regions": [r.as_dict() for r in regions],
        "passes": pass_stats,
    }


def run() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(
            "No images found in input_images/. Add one or more product images and run: python ocr_engine.py"
        )

    logger.info("Loading PaddleOCR (language=%s)...", OCR_LANG)
    ocr = build_ocr()

    # Only generated OCR files are cleared.
    for pattern in ("*.txt", "*.json"):
        for p in OUTPUT_DIR.glob(pattern):
            p.unlink()

    failures = 0
    for image in images:
        logger.info("OCR: %s", image.name)
        try:
            result = process_image(ocr, image)
            stem = image.stem
            (OUTPUT_DIR / f"{stem}.txt").write_text(result["text"], encoding="utf-8")
            (OUTPUT_DIR / f"{stem}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # Kept for compatibility with the existing evidence.py pipeline.
            (OUTPUT_DIR / f"{stem}_regions.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(
                "  DONE: regions=%d confidence=%.2f%% review=%s",
                result["statistics"]["regions"],
                result["ocr_confidence_percent"],
                result["review_required"],
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.exception("OCR failed for %s: %s", image.name, exc)

    if failures:
        raise SystemExit(f"OCR completed with {failures} failed image(s). Check the log above.")

    logger.info("OCR completed successfully for %d image(s).", len(images))


if __name__ == "__main__":
    run()
