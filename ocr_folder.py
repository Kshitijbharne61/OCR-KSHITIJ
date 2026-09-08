import os
import re
import json
import cv2


# =========================================================
# IMPORTANT PACKAGING SYMBOLS
# =========================================================
IMPORTANT_SYMBOLS = {
    "₹": "Indian Rupee",
    "®": "Registered Trademark",
    "™": "Trademark",
    "©": "Copyright",
    "$": "Dollar",
    "€": "Euro",
    "£": "Pound",
    "¥": "Yen",
    "%": "Percentage",
    "°": "Degree",
    "±": "Plus/Minus",
    "×": "Multiplication",
    "÷": "Division",
    "µ": "Micro",
    "Ω": "Ohm"
}


def normalize_symbols(text):
    """
    Conservatively normalize OCR symbols.
    Do NOT blindly convert R -> ® or TM -> ™ because that
    can create false product-label information.
    """
    text = text.replace("\u00a0", " ")
    text = text.replace("％", "%")
    text = text.replace("﹪", "%")

    # Currency alternatives commonly produced by OCR
    text = re.sub(r"(?i)\bINR\s*(?=\d)", "₹", text)
    text = re.sub(r"(?i)\bRs\.?\s*(?=\d)", "₹", text)

    return text.strip()


def detect_symbols(text):
    """Return important symbols actually detected in OCR text."""
    return [
        {"symbol": symbol, "meaning": meaning}
        for symbol, meaning in IMPORTANT_SYMBOLS.items()
        if symbol in text
    ]

# =========================================================
# PADDLEOCR SETTINGS
# =========================================================

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)


# =========================================================
# KEYWORDS
# =========================================================

KEYWORDS = {

    "manufacturer": [
        "Manufactured by",
        "Manufactured & marketed by",
        "Manufactured and marketed by",
        "Manufactured for",
        "Manufacturer"
    ],

    "packer": [
        "Packed by",
        "Packed & marketed by",
        "Packed and marketed by",
        "Packer"
    ],

    "importer": [
        "Imported by",
        "Importer"
    ],

    "net_quantity": [
        "Net quantity",
        "Net qty",
        "Net weight",
        "Net wt",
        "Net volume",
        "Contents"
    ],

    "mrp": [
        "mrp",
        "Maximum Retail Price",
        "Max Retail Price"
    ],

    "manufacturing_date": [
        "manufactured on",
        "manufactured date",
        "date of manufacture",
        "mfg",
        "mfd"
    ],

    "packing_date": [
        "packed on",
        "packing date",
        "date of packing",
        "pkd",
        "pkg"
    ],

    "import_date": [
        "date of import",
        "imported on"
    ],

    "batch": [
        "batch no",
        "batch number",
        "batch","BN"
    ],

    "lot": [
        "lot no",
        "lot number",
        "lot"
    ],

    "consumer_care": [
        "consumer care",
        "customer care",
        "customer service",
        "consumer service",
        "helpline",
        "contact us"
    ],

    "expiry": [
        "expiry",
        "exp","EXP",
        "use by",
        "best before"
    ],

    "address": [
        "address",
        "manufactured at",
        "manufactured by",
        "packed at",
        "packed by",
        "registered office"
    ]
}


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Cannot read:", image_path)
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # Upscale small images
    if width < 1200:

        gray = cv2.resize(
            gray,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    return gray


# =========================================================
# OCR RESULT
# =========================================================

def extract_text(result):

    texts = []
    scores = []

    for res in result:

        data = res.json

        if isinstance(data, dict) and "res" in data:
            data = data["res"]

        rec_texts = data.get("rec_texts", [])
        rec_scores = data.get("rec_scores", [])

        texts.extend(rec_texts)
        scores.extend(rec_scores)

    return texts, scores


# =========================================================
# NORMALIZE OCR TEXT
# =========================================================

def normalize(text):

    text = text.strip()

    # Preserve important Unicode packaging symbols
    text = normalize_symbols(text)

    # Remove excessive spaces
    text = re.sub(r"\s+", " ", text)

    return text


# =========================================================
# FIND KEYWORD
# =========================================================

def find_keyword_line(lines, keywords):

    for i, line in enumerate(lines):

        lower = line.lower()

        for keyword in keywords:

            if keyword in lower:

                return i, line

    return None, None


# =========================================================
# EXTRACT VALUE AFTER COLON
# =========================================================

def value_after_colon(line):

    if ":" in line:

        value = line.split(":", 1)[1].strip()

        if value:
            return value

    return ""


# =========================================================
# MANUFACTURER / PACKER / IMPORTER
# =========================================================

def extract_company(lines, field):

    index, line = find_keyword_line(
        lines,
        KEYWORDS[field]
    )

    if index is None:
        return None

    value = value_after_colon(line)

    # If OCR line contains "Manufactured by ABC"
    if not value:

        for keyword in KEYWORDS[field]:

            position = line.lower().find(keyword.lower())

            if position != -1:

                value = line[
                    position + len(keyword):
                ].strip(" :-")

                if value:
                    break

    if not value:
        return None

    return {
        "value": value,
        "line": index
    }


# =========================================================
# NET QUANTITY
# =========================================================

def extract_quantity(lines):

    pattern = re.compile(
        r"\b\d+(?:\.\d+)?\s*"
        r"(kg|g|mg|l|ml|cl|m|cm|mm|pcs|pieces|units?)\b",
        re.IGNORECASE
    )

    # First look near Net Quantity keyword
    index, line = find_keyword_line(
        lines,
        KEYWORDS["net_quantity"]
    )

    if index is not None:

        search_lines = lines[
            max(0, index - 1):
            min(len(lines), index + 3)
        ]

        for text in search_lines:

            match = pattern.search(text)

            if match:

                return match.group(0)

    # Fallback
    for line in lines:

        match = pattern.search(line)

        if match:

            return match.group(0)

    return None


# =========================================================
# MRP
# =========================================================

def extract_mrp(lines):

    pattern = re.compile(
        r"(?P<currency>₹|rs\.?|inr|\$|€|£|¥)?\s*"
        r"(?P<amount>\d+(?:,\d{3})*(?:\.\d{1,2})?)",
        re.IGNORECASE
    )

    index, line = find_keyword_line(
        lines,
        KEYWORDS["mrp"]
    )

    if index is None:
        return None

    # Search MRP line and next 2 lines
    search_lines = lines[
        index:min(len(lines), index + 3)
    ]

    for text in search_lines:

        match = pattern.search(text)

        if match:

            value = match.group("amount").replace(",", "")
            currency = match.group("currency")

            if currency:
                currency = currency.strip().lower()
                if currency in ("rs", "rs.", "inr"):
                    currency = "₹"
            else:
                currency = "₹"

            return currency + value

    return None


# =========================================================
# DATE EXTRACTION
# =========================================================

def extract_date(lines, field):

    date_pattern = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\b\d{1,2}[/-]\d{2,4}\b"
        r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|"
        r"oct|nov|dec)[a-z]*[\s/-]+\d{2,4}\b",
        re.IGNORECASE
    )

    index, line = find_keyword_line(
        lines,
        KEYWORDS[field]
    )

    if index is None:
        return None

    search_lines = lines[
        index:min(len(lines), index + 3)
    ]

    for text in search_lines:

        match = date_pattern.search(text)

        if match:

            return match.group(0)

    return None


# =========================================================
# BATCH / LOT
# =========================================================

def extract_batch(lines):

    pattern = re.compile(
        r"(?:batch|lot)\s*"
        r"(?:no\.?|number)?"
        r"\s*[:\-]?\s*"
        r"([A-Z0-9][A-Z0-9./_-]{2,})",
        re.IGNORECASE
    )

    for line in lines:

        match = pattern.search(line)

        if match:

            return match.group(1)

    return None


# =========================================================
# PHONE
# =========================================================

def extract_phone(text):

    pattern = re.compile(
        r"(?:\+91[\s-]?)?[6-9]\d{9}"
    )

    match = pattern.search(text)

    if match:
        return match.group(0)

    return None


# =========================================================
# EMAIL
# =========================================================

def extract_email(text):

    pattern = re.compile(
        r"[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}"
    )

    match = pattern.search(text)

    if match:
        return match.group(0)

    return None


# =========================================================
# CONSUMER CARE
# =========================================================

def extract_consumer_care(lines):

    index, line = find_keyword_line(
        lines,
        KEYWORDS["consumer_care"]
    )

    if index is None:
        return {
            "phone": None,
            "email": None
        }

    search_lines = lines[
        index:min(len(lines), index + 4)
    ]

    combined = " ".join(search_lines)

    return {
        "phone": extract_phone(combined),
        "email": extract_email(combined)
    }


# =========================================================
# EXPIRY / BEST BEFORE
# =========================================================

def extract_expiry(lines):

    index, line = find_keyword_line(
        lines,
        KEYWORDS["expiry"]
    )

    if index is None:
        return None

    # Return complete declaration
    return line


# =========================================================
# ADDRESS
# =========================================================

def extract_address(lines, start_index):

    if start_index is None:
        return None

    address_lines = []

    # Take following lines
    for i in range(
        start_index,
        min(len(lines), start_index + 5)
    ):

        line = lines[i].strip()

        if not line:
            continue

        # Stop when another major field starts
        lower = line.lower()

        stop_words = [
            "mrp",
            "net quantity",
            "batch",
            "lot no",
            "consumer care",
            "best before",
            "expiry"
        ]

        if i != start_index:

            if any(word in lower for word in stop_words):
                break

        address_lines.append(line)

    if address_lines:

        return "\n".join(address_lines)

    return None


# =========================================================
# PRODUCT NAME
# =========================================================

def extract_product_name(lines):

    # Strong indicators only
    strong_keywords = [
        "product name",
        "brand name"
    ]

    index, line = find_keyword_line(
        lines,
        strong_keywords
    )

    if index is not None:

        value = value_after_colon(line)

        if value:
            return value

    # Don't blindly use any line containing "product".
    # Use first reasonable large text line as fallback.
    candidates = []

    for line in lines:

        line = line.strip()

        if len(line) < 3:
            continue

        lower = line.lower()

        # Ignore obvious declaration lines
        ignored = [
            "mrp",
            "net quantity",
            "manufactured",
            "packed by",
            "imported by",
            "batch",
            "best before",
            "consumer care",
            "ingredients",
            "directions"
        ]

        if any(x in lower for x in ignored):
            continue

        if len(line) <= 60:
            candidates.append(line)

    if candidates:
        return candidates[0]

    return None


# =========================================================
# COMPLETE FIELD EXTRACTION
# =========================================================

def extract_fields(all_text):

    lines = [
        normalize(line)
        for line in all_text.split("\n")
        if normalize(line)
    ]

    fields = {}

    # Product
    fields["product_name"] = extract_product_name(lines)

    # Companies
    manufacturer = extract_company(
        lines,
        "manufacturer"
    )

    packer = extract_company(
        lines,
        "packer"
    )

    importer = extract_company(
        lines,
        "importer"
    )

    fields["manufacturer"] = (
        manufacturer["value"]
        if manufacturer else None
    )

    fields["packer"] = (
        packer["value"]
        if packer else None
    )

    fields["importer"] = (
        importer["value"]
        if importer else None
    )

    # Quantity
    fields["net_quantity"] = extract_quantity(lines)

    # MRP
    fields["mrp"] = extract_mrp(lines)

    # Dates
    fields["manufacturing_date"] = extract_date(
        lines,
        "manufacturing_date"
    )

    fields["packing_date"] = extract_date(
        lines,
        "packing_date"
    )

    fields["import_date"] = extract_date(
        lines,
        "import_date"
    )

    # Batch
    fields["batch_number"] = extract_batch(lines)

    # Consumer care
    fields["consumer_care"] = extract_consumer_care(
        lines
    )

    # Expiry
    fields["expiry_best_before"] = extract_expiry(
        lines
    )

    # Address
    address_index, address_line = find_keyword_line(
        lines,
        KEYWORDS["address"]
    )

    fields["address"] = extract_address(
        lines,
        address_index
    )

    return fields


# =========================================================
# BASIC COMPLIANCE
# =========================================================

def compliance_check(fields):

    # Candidate fields only.
    # Actual applicability will depend on commodity/rules.

    required_candidates = [
        "product_name",
        "net_quantity",
        "mrp"
    ]

    checks = {}

    for field in required_candidates:

        if fields.get(field):

            checks[field] = "FOUND"

        else:

            checks[field] = "NOT DETECTED"

    return checks


# =========================================================
# MAIN
# =========================================================

def process_folder():

    folder = "product_photos"

    if not os.path.exists(folder):

        print("❌ product_photos folder not found!")

        return


    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    )

    image_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ])


    if not image_files:

        print("❌ No images found!")

        return


    print("\n========================================")
    print("       PRODUCT PACKAGE OCR")
    print("========================================")

    print(
        "Images found:",
        len(image_files)
    )


    all_text = []
    all_scores = []


    # =====================================================
    # PROCESS EACH PHOTO
    # =====================================================

    for image_file in image_files:

        print("\n----------------------------------------")
        print("Processing:", image_file)
        print("----------------------------------------")

        image_path = os.path.join(
            folder,
            image_file
        )

        processed = preprocess_image(
            image_path
        )

        if processed is None:
            continue


        temp_file = os.path.join(
            folder,
            "_temp.jpg"
        )

        cv2.imwrite(
            temp_file,
            processed
        )


        print("Running OCR...")


        try:

            result = ocr.predict(
                temp_file
            )

            texts, scores = extract_text(
                result
            )

        except Exception as e:

            print("❌ OCR ERROR:")
            print(e)

            continue


        if not texts:

            print("⚠ No text detected")

            continue


        print("\nDetected:")

        for text, score in zip(
            texts,
            scores
        ):

            text = normalize(text)

            print(
                f"{text} "
                f"[{score * 100:.2f}%]"
            )

            all_text.append(text)
            all_scores.append(score)


    # =====================================================
    # CLEAN DUPLICATES
    # =====================================================

    unique_text = []

    seen = set()


    for text in all_text:

        key = text.lower()

        if key not in seen:

            unique_text.append(text)

            seen.add(key)


    final_text = "\n".join(
        unique_text
    )


    if not final_text:

        print("\n❌ Nothing detected")

        return


    # =====================================================
    # IMPORTANT SYMBOL DETECTION
    # =====================================================

    found_symbols = detect_symbols(final_text)

    print("\n========================================")
    print("        IMPORTANT OCR SYMBOLS")
    print("========================================")

    if found_symbols:
        for item in found_symbols:
            print(f"✓ {item['symbol']} = {item['meaning']}")
    else:
        print("⚠ No configured special symbols detected")


    # =====================================================
    # OCR CONFIDENCE
    # =====================================================

    average_confidence = (
        sum(all_scores)
        / len(all_scores)
    ) * 100


    # =====================================================
    # EXTRACT FIELDS
    # =====================================================

    fields = extract_fields(
        final_text
    )


    checks = compliance_check(
        fields
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n\n========================================")
    print("       EXTRACTED PRODUCT DETAILS")
    print("========================================")


    for field, value in fields.items():

        print(f"\n{field.upper()}:")

        if isinstance(value, dict):

            for k, v in value.items():

                print(
                    f"  {k}: {v}"
                )

        else:

            print(
                f"  {value}"
            )


    print("\n========================================")
    print("        OCR CONFIDENCE")
    print("========================================")

    print(
        f"{average_confidence:.2f}%"
    )


    print("\n========================================")
    print("        BASIC FIELD CHECK")
    print("========================================")


    for field, status in checks.items():

        if status == "FOUND":

            print(
                f"✅ {field}: FOUND"
            )

        else:

            print(
                f"⚠ {field}: NOT DETECTED"
            )


    # =====================================================
    # SAVE RAW TEXT
    # =====================================================

    with open(
        "combined_text.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "RAW OCR TEXT\n"
        )

        file.write(
            "============\n\n"
        )

        file.write(
            final_text
        )

        file.write(
            "\n\nOCR CONFIDENCE: "
            f"{average_confidence:.2f}%"
        )


    # =====================================================
    # SAVE JSON
    # =====================================================

    output = {

        "product_details": fields,

        "ocr_confidence": round(
            average_confidence,
            2
        ),

        "basic_checks": checks,

        "detected_symbols": found_symbols,

        "raw_ocr_text": final_text
    }


    with open(
        "product_details.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )


    # =====================================================
    # SAVE HUMAN READABLE FILE
    # =====================================================

    with open(
        "product_details.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PRODUCT DETAILS\n"
        )

        file.write(
            "===============\n\n"
        )


        for field, value in fields.items():

            file.write(
                f"{field}: {value}\n"
            )


        file.write(
            "\n\nOCR CONFIDENCE: "
            f"{average_confidence:.2f}%\n"
        )


        file.write(
            "\n\nBASIC CHECK\n"
        )

        file.write(
            "===========\n\n"
        )


        for field, status in checks.items():

            file.write(
                f"{field}: {status}\n"
            )


    # =====================================================
    # DELETE TEMP
    # =====================================================

    temp_file = os.path.join(
        folder,
        "_temp.jpg"
    )

    if os.path.exists(temp_file):

        os.remove(temp_file)


    print("\n========================================")
    print("              DONE")
    print("========================================")

    print(
        "\n✅ combined_text.txt created"
    )

    print(
        "✅ product_details.txt created"
    )

    print(
        "✅ product_details.json created"
    )


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    process_folder()