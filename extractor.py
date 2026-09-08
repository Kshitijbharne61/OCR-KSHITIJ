import os
import re
import json
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

KEYWORDS_FOLDER = "keywords"
INPUT_FOLDER = "input_text"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize OCR text while keeping useful information.
    """

    # Normalize different dash characters
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    # Normalize rupee variations
    text = text.replace("Rs.", "Rs")
    text = text.replace("Rs .", "Rs")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# LOAD KEYWORD FILES
# ============================================================

def load_keywords():
    """
    Load all JSON files from keywords folder.
    """

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
# FIND KEYWORD
# ============================================================

def keyword_match(line, keyword):

    line_lower = line.lower()
    keyword_lower = keyword.lower()

    # Exact phrase
    if keyword_lower in line_lower:
        return True

    # Handle punctuation differences
    clean_line = re.sub(r"[^a-z0-9₹@./%-]", " ", line_lower)
    clean_keyword = re.sub(r"[^a-z0-9₹@./%-]", " ", keyword_lower)

    return clean_keyword in clean_line


# ============================================================
# FIND BEST KEYWORD MATCH
# ============================================================

def find_keyword_matches(lines, field_data):

    matches = []

    keywords = field_data["keywords"]

    for line_number, line in enumerate(lines):

        for keyword in keywords:

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

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE
            )

            for match in matches:

                if isinstance(match, tuple):
                    match = " ".join(match)

                if match not in results:
                    results.append(match)

        except re.error:
            continue

    return results


# ============================================================
# GENERIC VALUE EXTRACTION
# ============================================================

def clean_value(value):

    value = value.strip()

    value = re.sub(
        r"^[\s:;\-–—]+",
        "",
        value
    )

    value = re.sub(
        r"[\s:;\-–—]+$",
        "",
        value
    )

    return value.strip()


def extract_after_keyword(line, keyword):

    pattern = re.escape(keyword)

    match = re.search(
        pattern + r"\s*[:\-]?\s*(.*)",
        line,
        flags=re.IGNORECASE
    )

    if match:

        value = match.group(1)

        return clean_value(value)

    return ""


# ============================================================
# FIELD-SPECIFIC EXTRACTION
# ============================================================

def extract_mrp(text):

    patterns = [

        r"(?:MRP|M\.R\.P)\s*[:\-]?\s*(?:₹|Rs\.?)?\s*[\d,]+(?:\.\d{1,2})?",

        r"(?:Maximum Retail Price|Max Retail Price)"
        r"\s*[:\-]?\s*(?:₹|Rs\.?)?\s*[\d,]+(?:\.\d{1,2})?"

    ]

    results = regex_extract(text, patterns)

    if results:
        return results[0]

    return ""


def extract_quantity(text):

    patterns = [

        r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|litre|liter)\b",

        r"\b\d+(?:\.\d+)?\s*(?:pcs|pieces|nos)\b"

    ]

    results = regex_extract(text, patterns)

    return results[0] if results else ""


def extract_phone(text):

    patterns = [

        r"\b1800[-\s]?\d{3}[-\s]?\d{3,4}\b",

        r"\b\d{10}\b",

        r"\b\d{3,4}[-\s]\d{3,4}[-\s]\d{3,4}\b"

    ]

    results = regex_extract(text, patterns)

    return results[0] if results else ""


def extract_email(text):

    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    results = regex_extract(text, [pattern])

    return results[0] if results else ""


def extract_date(text):

    patterns = [

        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",

        r"\b\d{1,2}[/-]\d{2,4}\b",

        r"\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b"

    ]

    results = regex_extract(text, patterns)

    return results


# ============================================================
# CONTEXT EXTRACTION
# ============================================================

def get_context(lines, line_number, window=2):

    start = max(0, line_number - window)

    end = min(
        len(lines),
        line_number + window + 1
    )

    return lines[start:end]


# ============================================================
# EXTRACT FIELD
# ============================================================

def extract_field(text, lines, field_data):

    field = field_data["field"]

    # --------------------------------------------------------
    # SPECIAL REGEX FIELDS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DATE FIELDS
    # --------------------------------------------------------

    if field == "manufacturing_packing_date":

        matches = extract_date(text)

        # Prefer dates near manufacturing/packing keywords

        for match in matches:

            for line in lines:

                if match in line:

                    lower = line.lower()

                    if (
                        "mfd" in lower
                        or "mfg" in lower
                        or "manufactur" in lower
                        or "pkd" in lower
                        or "pack" in lower
                    ):
                        return match

        if matches:
            return matches[0]

    # --------------------------------------------------------
    # KEYWORD + CONTEXT
    # --------------------------------------------------------

    matches = find_keyword_matches(
        lines,
        field_data
    )

    candidates = []

    for match in matches:

        line_number = match["line_number"]

        keyword = match["keyword"]

        line = match["line"]

        # ----------------------------------------------
        # First try value on same line
        # ----------------------------------------------

        value = extract_after_keyword(
            line,
            keyword
        )

        if value:

            candidates.append({
                "value": value,
                "score": 10,
                "line": line,
                "keyword": keyword
            })

        # ----------------------------------------------
        # Look at next 1-2 lines
        # ----------------------------------------------

        for offset in range(1, 3):

            next_line_number = line_number + offset

            if next_line_number >= len(lines):
                continue

            next_line = lines[next_line_number].strip()

            if not next_line:
                continue

            # Ignore another obvious label
            if re.search(
                r"^(mrp|batch|expiry|mfd|mfg|net quantity)",
                next_line,
                re.IGNORECASE
            ):
                continue

            score = 8 - offset

            candidates.append({
                "value": clean_value(next_line),
                "score": score,
                "line": next_line,
                "keyword": keyword
            })

    # --------------------------------------------------------
    # Return highest scoring result
    # --------------------------------------------------------

    if candidates:

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return candidates[0]["value"]

    return ""


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence(value, field):

    if not value:
        return 0

    score = 50

    # Strong regex-based fields
    if field == "consumer_care_email":
        if re.search(
            r"@.*\.",
            value
        ):
            score = 98

    elif field == "consumer_care_phone":

        digits = re.sub(
            r"\D",
            "",
            value
        )

        if len(digits) >= 10:
            score = 98

    elif field == "mrp":

        if re.search(
            r"(₹|Rs|MRP)",
            value,
            re.IGNORECASE
        ):
            score = 95

    elif field == "net_quantity":

        if re.search(
            r"\d+\s*(mg|g|kg|ml|l|litre|liter|pcs)",
            value,
            re.IGNORECASE
        ):
            score = 95

    else:

        score = 75

    return min(score, 100)


# ============================================================
# PROCESS ONE TEXT FILE
# ============================================================

def process_file(txt_file, keyword_database):

    print("\n======================================")
    print(f"Processing: {txt_file}")
    print("======================================")

    with open(
        txt_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        text = f.read()

    text = normalize_text(text)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    result = {}

    for field_data in keyword_database:

        field = field_data["field"]

        print(f"Detecting: {field}")

        value = extract_field(
            text,
            lines,
            field_data
        )

        confidence = calculate_confidence(
            value,
            field
        )

        result[field] = {
            "value": value,
            "confidence": confidence
        }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading keyword database...")

    keyword_database = load_keywords()

    print(
        f"Loaded {len(keyword_database)} keyword files."
    )

    txt_files = list(
        Path(INPUT_FOLDER).glob("*.txt")
    )

    if not txt_files:

        print(
            f"\nNo TXT files found in: {INPUT_FOLDER}"
        )

        return

    for txt_file in txt_files:

        result = process_file(
            str(txt_file),
            keyword_database
        )

        output_file = (
            Path(OUTPUT_FOLDER)
            / f"{txt_file.stem}_extracted.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False
            )

        print(
            f"\nSaved: {output_file}"
        )

    print("\n======================================")
    print("EXTRACTION COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()