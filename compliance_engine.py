"""Explainable first-pass compliance engine for SIH26034.

This module deliberately separates extraction from legal decisions. It performs
screening checks only; commodity-specific applicability and final legal decisions
should be confirmed against the current Department of Consumer Affairs ruleset.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RULESET_VERSION = "PCR-2011-plus-amendments-through-2026-05-29"


class Status:
    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_VERIFICATION = "NEEDS_VERIFICATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class Finding:
    rule_id: str
    rule_reference: str
    requirement: str
    status: str
    field: str
    detected_value: str | None
    confidence: float
    notes: str


def value_of(data: dict, field: str) -> tuple[str | None, float]:
    item = data.get(field, {})
    if isinstance(item, dict):
        value = item.get("value")
        return (value or None, float(item.get("confidence", 0) or 0))
    return (str(item) if item else None, 0.0)


def present(data: dict, *fields: str) -> tuple[str | None, float, str | None]:
    for field in fields:
        value, confidence = value_of(data, field)
        if value:
            return value, confidence, field
    return None, 0.0, None


def check_presence(rule_id: str, reference: str, requirement: str, data: dict, fields: tuple[str, ...], *, conditional=False) -> Finding:
    value, confidence, field = present(data, *fields)
    if conditional and value is None:
        return Finding(rule_id, reference, requirement, Status.NOT_APPLICABLE, fields[0], None, 0.0, "Conditional declaration; applicability must be confirmed for this commodity.")
    if value is None:
        return Finding(rule_id, reference, requirement, Status.FAIL, fields[0], None, 0.0, "No matching declaration was extracted from OCR text.")
    if confidence and confidence < 60:
        return Finding(rule_id, reference, requirement, Status.NEEDS_VERIFICATION, field or fields[0], value, confidence, "A value was detected, but OCR confidence is low; inspect the package image.")
    return Finding(rule_id, reference, requirement, Status.PASS, field or fields[0], value, confidence, "Declaration detected by the extraction pipeline.")


def run_checks(extracted: dict, *, imported: bool = False, perishable: bool = False, medical_device: bool = False) -> list[Finding]:
    findings = [
        check_presence("R6-NAME", "Rule 6(1)", "Name/common or generic name of the commodity should be declared.", extracted, ("product_name", "common_generic_name")),
        check_presence("R6-QTY", "Rule 6(1)", "Net quantity/number of the commodity should be declared.", extracted, ("net_quantity",)),
        check_presence("R6-MRP", "Rule 6(1)", "Maximum Retail Price (MRP) should be declared.", extracted, ("mrp",)),
        check_presence("R6-ENTITY", "Rule 6(1)", "Manufacturer/packer/importer declaration should be present as applicable.", extracted, ("manufacturer", "packer", "importer")),
        check_presence("R6-DATE", "Rule 6(1)", "Month/year of manufacture or packing/import should be declared as applicable.", extracted, ("manufacturing_packing_date", "manufacturing_date", "packing_date", "import_date")),
        check_presence("R6-CARE", "Rule 6(1)", "Consumer-care contact information should be declared where applicable.", extracted, ("consumer_care_phone", "consumer_care_email", "consumer_care"), conditional=True),
    ]

    if imported:
        findings.append(check_presence(
            "R6-COO", "Rule 6", "Country of origin should be declared for imported products where applicable.",
            extracted, ("country_of_origin",)
        ))

    if perishable:
        findings.append(check_presence(
            "R6-BESTBEFORE", "Rule 6(1)", "Best-before/use-by information should be declared for applicable commodities.",
            extracted, ("expiry_best_before", "best_before", "expiry", "expiry_date")
        ))

    # The 2025 amendment created a specific medical-device exception for declaration placement
    # and numeral/letter height. We therefore never claim a normal font rule is satisfied for a
    # medical device without a Medical Devices Rules check.
    if medical_device:
        findings.append(Finding(
            "R7-MEDICAL-DEVICE", "Rule 7", "Medical-device declaration size/placement follows the applicable Medical Devices Rules.",
            Status.NEEDS_VERIFICATION, "font_size", None, 0.0,
            "Medical device detected/flagged: use the Medical Devices Rules, 2017 requirements rather than a generic PCR font-size verdict."
        ))

    return findings


def summarize(findings: list[Finding]) -> dict:
    applicable = [f for f in findings if f.status != Status.NOT_APPLICABLE]
    fail = sum(f.status == Status.FAIL for f in applicable)
    review = sum(f.status == Status.NEEDS_VERIFICATION for f in applicable)
    passed = sum(f.status == Status.PASS for f in applicable)
    overall = Status.FAIL if fail else Status.NEEDS_VERIFICATION if review else Status.PASS
    score = round(100 * passed / len(applicable), 1) if applicable else None
    return {
        "overall_status": overall,
        "compliance_score": score,
        "pass_count": passed,
        "fail_count": fail,
        "needs_verification_count": review,
        "ruleset_version": RULESET_VERSION,
    }


def load_extracted(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_file(extracted_path: Path, *, imported=False, perishable=False, medical_device=False) -> dict:
    extracted = load_extracted(extracted_path)
    findings = run_checks(extracted, imported=imported, perishable=perishable, medical_device=medical_device)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ruleset_version": RULESET_VERSION,
        "summary": summarize(findings),
        "findings": [asdict(f) for f in findings],
    }


def main() -> None:
    extracted_dir = ROOT / "extracted"
    compliance_dir = ROOT / "compliance_reports"
    compliance_dir.mkdir(exist_ok=True)
    files = sorted(extracted_dir.glob("*_extracted.json"))
    if not files:
        print("No extracted JSON files found. Run pipeline.py first.")
        return

    for path in files:
        report = run_file(path)
        out = compliance_dir / f"{path.stem.replace('_extracted', '')}_compliance.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        s = report["summary"]
        print(f"Compliance: {path.name} -> {s['overall_status']} ({s['compliance_score']}%)")


if __name__ == "__main__":
    main()
