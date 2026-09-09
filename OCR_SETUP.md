# SIH26034 OCR Setup

This repository's OCR role is to convert product-label images into:

- `output_text/<image>.txt` — clean OCR text
- `output_text/<image>.json` — structured OCR result with text, confidence and bounding boxes
- `output_text/<image>_regions.json` — compatibility file for the evidence pipeline

## 1. Windows environment

Use Python 3.12.x for the recommended PaddlePaddle CPU setup.

```powershell
py -3.12 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
```

## 2. Verify the runtime

```powershell
python verify_ocr.py
```

The check must end with `READY`.

## 3. Run OCR

Put product photos in:

```text
input_images/
```

Then run:

```powershell
python ocr_engine.py
```

For the complete SIH pipeline instead:

```powershell
python pipeline.py
```

## 4. Accuracy strategy

The OCR engine uses PaddleOCR as the primary recognizer and performs several preprocessing passes:

1. original/upscaled image
2. CLAHE contrast enhancement
3. sharpened grayscale
4. adaptive threshold
5. Otsu threshold

The outputs from the successful passes are merged, duplicate observations are removed, and the strongest confidence observation is retained. Bounding boxes are preserved so later modules can show evidence on the original product image.

The JSON includes a `review_required` flag whenever there are low-confidence regions or no detected text. This is intentional: OCR cannot honestly guarantee 100% accuracy for every photograph, blur level, font, glare, curved package, or language.

## 5. Indian-language labels

The default is `OCR_LANG=en`. If a supported PaddleOCR language model is installed, it can be selected without changing code:

```powershell
$env:OCR_LANG="hi"
python ocr_engine.py
```

For a multilingual production workflow, benchmark the required Indian languages on the team's real product-image dataset before enabling them by default.
