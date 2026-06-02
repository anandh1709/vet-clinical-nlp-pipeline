# quality.py — Stage 4: Data quality scoring across six dimensions

import json
import re


def score_completeness(records, is_processed=False):
    """Measure proportion of expected fields that are present."""
    total = 0
    filled = 0
    for r in records:
        total += 3
        if r["raw_text"] and len(r["raw_text"]) > 0:
            filled += 1
        if len(r["ground_truth"]["icd_labels"]) > 0:
            filled += 1
        if len(r["ground_truth"]["disease_spans"]) > 0:
            filled += 1
        if is_processed:
            total += 3
            if r.get("extracted") and len(r["extracted"]) > 0:
                filled += 1
            if r.get("standardised") and len(r["standardised"]) > 0:
                filled += 1
            if r.get("deidentified") and len(r["deidentified"]) > 0:
                filled += 1
    return filled / total if total > 0 else 0


def score_accuracy(records, is_processed=False):
    """Measure proportion of values matching canonical vocabularies."""
    if not is_processed:
        canonical_terms = {
            "No Abnormalities Detected", "Bright Alert Responsive",
            "Heart Rate", "Capillary Refill Time", "Respiratory Rate",
            "Within Normal Limits", "Clinical Examination",
            "Physical Examination", "Body Condition Score",
        }
        total = 0
        matched = 0
        for r in records:
            words = re.findall(r'\b[A-Z]{2,5}\b', r["raw_text"])
            total += len(words)
            for w in words:
                if w in canonical_terms:
                    matched += 1
        return matched / total if total > 0 else 0
    else:
        total = 0
        matched = 0
        for r in records:
            std = r.get("standardised", {})
            for ptype, items in std.items():
                if ptype == "diagnoses":
                    for item in items:
                        total += 1
                        if item.get("method") != "unmatched":
                            matched += 1
                else:
                    for item in items:
                        total += 1
                        if item.get("method") in ("exact_match", "parsed"):
                            matched += 1
        return matched / total if total > 0 else 0


def score_consistency(records, is_processed=False):
    """Check cross-field logical validity (weight, temperature, age ranges)."""
    if not is_processed:
        return 0.5
    checks = 0
    passed = 0
    for r in records:
        std = r.get("standardised", {})
        for item in std.get("weight", []):
            if "value" in item:
                checks += 1
                if 0.1 <= item["value"] <= 100:
                    passed += 1
        for item in std.get("temperature", []):
            if "value" in item:
                checks += 1
                if 35 <= item["value"] <= 42:
                    passed += 1
        for item in std.get("age", []):
            if "value" in item:
                checks += 1
                if 0 < item["value"] <= 30:
                    passed += 1
    return passed / checks if checks > 0 else 0.5


def score_validity(records, is_processed=False):
    """Check format correctness (non-empty text, numeric values have units)."""
    total = 0
    valid = 0
    for r in records:
        total += 1
        if len(r["raw_text"]) > 5:
            valid += 1
        if is_processed:
            std = r.get("standardised", {})
            for item in std.get("weight", []):
                total += 1
                if "value" in item and "unit" in item:
                    valid += 1
            for item in std.get("age", []):
                total += 1
                if "value" in item and "unit" in item:
                    valid += 1
            for item in std.get("temperature", []):
                total += 1
                if "value" in item:
                    valid += 1
    return valid / total if total > 0 else 0


def score_uniqueness(records):
    """Detect duplicate records based on raw text."""
    texts = [r["raw_text"] for r in records]
    unique = len(set(texts))
    return unique / len(texts) if texts else 0


def score_standardisation(records, is_processed=False):
    """Measure proportion of terms successfully mapped to canonical forms."""
    if not is_processed:
        return 0.0
    total = 0
    standardised = 0
    for r in records:
        std = r.get("standardised", {})
        for ptype, items in std.items():
            for item in items:
                total += 1
                if item.get("method") in ("exact_match", "fuzzy_match", "semantic_match", "zero_shot", "parsed"):
                    standardised += 1
    return standardised / total if total > 0 else 0


if __name__ == "__main__":
    # Load raw and processed records
    with open("data/raw/peteval_records.json") as f:
        raw_records = json.load(f)
    with open("data/outputs/peteval_deidentified.json") as f:
        processed_records = json.load(f)

    # Compute all six dimensions
    dimensions = {
        "Completeness": (
            score_completeness(raw_records, is_processed=False),
            score_completeness(processed_records, is_processed=True)
        ),
        "Accuracy": (
            score_accuracy(raw_records, is_processed=False),
            score_accuracy(processed_records, is_processed=True)
        ),
        "Consistency": (
            score_consistency(raw_records, is_processed=False),
            score_consistency(processed_records, is_processed=True)
        ),
        "Validity": (
            score_validity(raw_records, is_processed=False),
            score_validity(processed_records, is_processed=True)
        ),
        "Uniqueness": (
            score_uniqueness(raw_records),
            score_uniqueness(processed_records)
        ),
        "Standardisation": (
            score_standardisation(raw_records, is_processed=False),
            score_standardisation(processed_records, is_processed=True)
        ),
    }

    # Print quality report
    print("DATA QUALITY REPORT")
    print(f"\n{'Dimension':<20s} {'Before':>8s} {'After':>8s} {'Delta':>8s}")
    print("-" * 46)
    for dim, (before, after) in dimensions.items():
        delta = after - before
        print(f"{dim:<20s} {before:>8.3f} {after:>8.3f} {delta:>+8.3f}")

    before_avg = sum(b for b, a in dimensions.values()) / len(dimensions)
    after_avg = sum(a for b, a in dimensions.values()) / len(dimensions)
    print("-" * 46)
    print(f"{'OVERALL':<20s} {before_avg:>8.3f} {after_avg:>8.3f} {after_avg - before_avg:>+8.3f}")

    # Save quality scores into records
    for r in processed_records:
        r["quality_scores"] = {
            dim: {"before": before, "after": after}
            for dim, (before, after) in dimensions.items()
        }

    with open("data/outputs/peteval_final.json", "w") as f:
        json.dump(processed_records, f, indent=2)

    print(f"\nSaved {len(processed_records)} records with quality scores")
