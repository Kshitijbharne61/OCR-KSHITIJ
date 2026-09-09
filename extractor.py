import os
import re
import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

KEYWORDS_FOLDER = "keywords"
INPUT_FOLDER = "output_text"      # OCR TXT files
OUTPUT_FOLDER = "extracted"       # Final JSON files

os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize OCR text while keeping useful product information."""
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")
    text = re.sub(r"Rs\s*\.\s*", "Rs ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# ============================================================
# LOAD KEYWORD FILES
# ============================================================

def load_keywords():
    keyword_database = []

    for file in sorted(Path(KEYWORDS_FOLDER).glob("*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            keyword_database.append({
                "filename": file.name,
                "field": data.get("field", file.stem),
                "keywords": data.get("keywords", []),
                "patterns": data.get("patterns", [])
            })
        except Exception as e:
            print(f"ERROR reading {file}: {e}")

    return keyword_database


# ============================================================
# MATCHING
# ============================================================

def keyword_match(line, keyword):
    line_lower = line.lower()
    keyword_lower = keyword.lower()

    if keyword_lower in line_lower:
        return True

    clean_line = re.sub(r"[^a-z0-9₹@./%+\-]", " ", line_lower)
    clean_keyword = re.sub(r"[^a-z0-9₹@./%+\-]", " ", keyword_lower)
    return clean_keyword in clean_line


def find_keyword_matches(lines, field_data):
    matches = []

    for line_number, line in enumerate(lines):
        for keyword in field_data["keywords"]:
            if keyword_match(line, keyword):
                matches.append({
                    "line_number": line_number,
                    "line": line,
                    "keyword": keyword
                })

    return matches


# ============================================================
# REGEX EXTRACTION
# ============================================================

def regex_extract(text, patterns):
    results = []

    for pattern in patterns:
        try:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = " ".join(match)
                if match not in results:
                    results.append(match)
        except re.error:
            continue

    return results


def extract_mrp(text):
    patterns = [
        r"(?:MRP|M\.R\.P|MAXIMUM RETAIL PRICE|MAX RETAIL PRICE)\s*[:\-]?\s*(?:₹|Rs\.?|INR)?\s*[\d,]+(?:\.\d{1,2})?",
        r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d{1,2})?"
    ]
    results = regex_extract(text, patterns)
    return results[0] if results else ""


def extract_quantity(text):
    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|litre|liter|litres|liters)\b",
        r"\b\d+(?:\.\d+)?\s*(?:pcs|pieces|nos)\b"
    ]
    results = regex_extract(text, patterns)
    return results[0] if results else ""


def extract_phone(text):
    patterns = [
        r"\b1800[-\s]?\d{3}[-\s]?\d{3,4}\b",
        r"\b[6-9]\d{9}\b",
        r"\b\d{3,4}[-\s]\d{3,4}[-\s]\d{3,4}\b"
    ]
    results = regex_extract(text, patterns)
    return results[0] if results else ""


def extract_email(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    results = regex_extract(text, [pattern])
    return results[0] if results else ""


def extract_date(text):
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b"
    ]
    return regex_extract(text, patterns)


# ============================================================
# VALUE EXTRACTION
# ============================================================

def clean_value(value):
    value = value.strip()
    value = re.sub(r"^[\s:;\-–—]+", "", value)
    value = re.sub(r"[\s:;\-–—]+$", "", value)
    return value.strip()


def extract_after_keyword(line, keyword):
    pattern = re.escape(keyword)
    match = re.search(pattern + r"\s*[:\-]?\s*(.*)", line, flags=re.IGNORECASE)
    return clean_value(match.group(1)) if match else ""


# ============================================================
# FIELD EXTRACTION
# ============================================================

def extract_field(text, lines, field_data):
    field = field_data["field"]

    # Use custom regex patterns from each keyword JSON first.
    custom_results = regex_extract(text, field_data.get("patterns", []))
    if custom_results:
        if field in {"mrp", "net_quantity", "consumer_care_phone", "consumer_care_email"}:
            return custom_results[0]

    if field == "mrp":
        value = extract_mrp(text)
        if value:
            return value

    if field == "net_quantity":
        value = extract_quantity(text)
        if value:
            return value

    if field == "consumer_care_phone":
        value = extract_phone(text)
        if value:
            return value

    if field == "consumer_care_email":
        value = extract_email(text)
        if value:
            return value

    if field in {"manufacturing_packing_date", "manufacturing_date", "packing_date", "import_date", "expiry", "expiry_date"}:
        matches = extract_date(text)
        for match in matches:
            for line in lines:
                if match in line:
                    lower = line.lower()
                    if any(word in lower for word in ["mfd", "mfg", "manufactur", "pkd", "pack", "import", "exp", "best before", "use by"]):
                        return match
        if matches:
            return matches[0]

    matches = find_keyword_matches(lines, field_data)
    candidates = []

    for match in matches:
        line_number = match["line_number"]
        keyword = match["keyword"]
        line = match["line"]

        value = extract_after_keyword(line, keyword)
        if value and value.lower() != keyword.lower():
            candidates.append({
                "value": value,
                "score": 10 + min(len(value), 30) / 100,
                "line": line,
                "keyword": keyword
            })

        # Product labels are often on the line immediately before the value
        # or followed by a separate value line.
        for offset in range(1, 3):
            next_line_number = line_number + offset
            if next_line_number >= len(lines):
                continue

            next_line = lines[next_line_number].strip()
            if not next_line:
                continue

            if re.search(r"^(mrp|batch|lot|expiry|mfd|mfg|net quantity|net qty|best before)", next_line, re.IGNORECASE):
                continue

            candidates.append({
                "value": clean_value(next_line),
                "score": 8 - offset,
                "line": next_line,
                "keyword": keyword
            })

    if candidates:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[0]["value"]

    return ""


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(value, field):
    if not value:
        return 0

    if field == "consumer_care_email" and re.search(r"@.*\.", value):
        return 98

    if field == "consumer_care_phone":
        digits = re.sub(r"\D", "", value)
        return 98 if len(digits) >= 10 else 60

    if field == "mrp" and re.search(r"(₹|Rs|INR|MRP)", value, re.IGNORECASE):
        return 95

    if field == "net_quantity" and re.search(r"\d+\s*(mg|g|kg|ml|l|litre|liter|pcs|pieces|nos)", value, re.IGNORECASE):
        return 95

    return 75


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(txt_file, keyword_database):
    print("\n======================================")
    print(f"Processing: {txt_file}")
    print("======================================")

    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    text = normalize_text(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    result = {}

    for field_data in keyword_database:
        field = field_data["field"]
        print(f"Detecting: {field}")

        value = extract_field(text, lines, field_data)
        result[field] = {
            "value": value,
            "confidence": calculate_confidence(value, field)
        }

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading keyword database...")
    keyword_database = load_keywords()
    print(f"Loaded {len(keyword_database)} keyword files.")

    txt_files = sorted(Path(INPUT_FOLDER).glob("*.txt"))

    if not txt_files:
        print(f"\nNo TXT files found in: {INPUT_FOLDER}")
        return

    for txt_file in txt_files:
        result = process_file(str(txt_file), keyword_database)
        output_file = Path(OUTPUT_FOLDER) / f"{txt_file.stem}_extracted.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"Saved: {output_file}")

    print("\n======================================")
    print("EXTRACTION COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()
