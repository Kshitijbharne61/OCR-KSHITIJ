"""Verify that the local OCR environment is ready before running product images."""
from __future__ import annotations

import sys


def main() -> int:
    print("=== SIH26034 OCR ENVIRONMENT CHECK ===")
    print(f"Python: {sys.version.split()[0]}")

    if sys.version_info[:2] != (3, 12):
        print("WARNING: Python 3.12.x is the recommended Windows runtime for this project.")

    try:
        import paddle
        print(f"PaddlePaddle: {paddle.__version__}")
    except Exception as exc:
        print(f"ERROR: PaddlePaddle is not usable: {exc}")
        print("Install it with:")
        print("python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/")
        return 1

    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except Exception as exc:
        print(f"ERROR: OpenCV is not usable: {exc}")
        return 1

    try:
        from paddleocr import PaddleOCR
        print("PaddleOCR: import OK")
        # Construction verifies the actual local inference configuration.
        PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
        print("PaddleOCR: initialization OK")
    except Exception as exc:
        print(f"ERROR: PaddleOCR is not usable: {exc}")
        return 1

    print("\nREADY: put product images in input_images/ and run: python ocr_engine.py")
    print("Outputs: output_text/*.txt + output_text/*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
