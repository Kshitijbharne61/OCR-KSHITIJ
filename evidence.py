"""Attach OCR-region evidence to compliance findings for an explainable SIH demo."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FIELD_ANCHORS = {
    "product_name": ["product name", "common name", "generic name"],
    "net_quantity": ["net quantity", "net qty", "net weight", "net wt", "net volume"],
    "mrp": ["mrp", "m.r.p", "maximum retail price", "max retail price"],
    "manufacturer": ["manufactured by", "manufacturer", "manufactured for"],
    "packer": ["packed by", "packer", "packed at"],
    "importer": ["imported by", "importer"],
    "manufacturing_packing_date": ["mfd", "mfg", "manufactured", "packed on", "pkd", "pkg"],
    "consumer_care_phone": ["consumer care", "customer care", "helpline", "toll free"],
    "consumer_care_email": ["email", "e-mail"],
    "country_of_origin": ["country of origin", "made in", "product of"],
    "expiry_best_before": ["best before", "use by", "expiry", "exp"],
}


def load_regions(stem: str) -> list[dict]:
    path = ROOT / "output_text" / f"{stem}_regions.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("regions", [])


def nearest_region(field: str, regions: list[dict]) -> dict | None:
    anchors = FIELD_ANCHORS.get(field, [])
    for region in regions:
        text = str(region.get("text", "")).casefold()
        if any(a.casefold() in text for a in anchors):
            return region
    return None


def enrich_report(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stem = report_path.stem.replace("_compliance", "")
    regions = load_regions(stem)
    for finding in report.get("findings", []):
        region = nearest_region(finding.get("field", ""), regions)
        finding["evidence_region"] = region
        if finding.get("status") == "FAIL" and region:
            finding["notes"] += " Evidence region is available in the OCR region file for UI highlighting."
    report["evidence"] = {
        "region_count": len(regions),
        "source": f"output_text/{stem}_regions.json",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    directory = ROOT / "compliance_reports"
    for path in sorted(directory.glob("*_compliance.json")):
        enrich_report(path)
        print(f"Evidence linked: {path.name}")


if __name__ == "__main__":
    main()
