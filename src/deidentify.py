# deidentify.py — Stage 3: De-identification with regex, NER, and k-anonymity

import json
import re
import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

# Terms to skip during NER-based redaction
VET_STOPLIST = {
    "BAR", "NAD", "QAR", "ADR", "TPR", "DRB", "BARH",
    "BC", "GI", "HPC", "OE", "CE", "RE", "LE",
    "FS", "MN", "MC", "FI", "MI", "FE", "ME",
    "HR", "Hr", "CRT", "RR", "BP", "BID", "SID", "TID",
    "PO", "IV", "SC", "IM", "PRN", "EOD",
    "mm", "kg", "mg", "ml",
    "Nobivac", "Metacam", "Rimadyl", "Synulox",
    "Post", "pre",
    "RH", "LH", "RF", "LF", "CCL", "CrCL", "ACL", "PE",
    "KC", "DUDE", "MM", "Adv", "FNA", "Abdo", "abdo",
    "D+", "NSAIDs", "NSAID", "BCS", "POC", "TM", "GA",
    "Owner", "owner", "WNL", "BIOP", "DHP", "LHS", "OA",
    "Advise", "advise", "IOP", "Tael", "RV", "STT", "wt",
    "L", "l4", "L4", "CE NAD", "BCS 5/9", "BCS 4/9",
    "BCS 6/9", "BCS 7/9", "BCS 3/9",
    "FeLV", "URT", "CRT<2", "CRT<2s", "CRT <2s",
    "Thoracic", "thoracic ausc", "EAG", "INI", "SI",
    "Booster", "neoplasia", "occ", "Cont", "Reex",
    "HGE", "LN", "ROM", "U/S", "KCS", "Horner",
    "clin exam", "Haematology", "Pre-Surgical Panel",
    "Booster Vaccination - Authorised", "L4 + KC",
    "Tues", "Mon", "Wed", "Thurs", "Fri", "Sat", "Sun",
    "Next", "HIND", "Hind", "FORE", "Fore",
    "Twice Daily", "Once Daily", "Three Times Daily",
    "anaesthesia", "Anaesthesia", "anesthesia", "BSAVA",
}


def regex_redact(text):
    """Step 1: Find and replace structured PII patterns (phone, email, postcode, etc)."""
    placeholder_map = {}
    counter = {}

    def get_placeholder(pii_type, value):
        """Assign a consistent indexed placeholder to each unique PII value."""
        if value in placeholder_map:
            return placeholder_map[value]
        count = counter.get(pii_type, 0) + 1
        counter[pii_type] = count
        placeholder = f"[{pii_type}_{count}]"
        placeholder_map[value] = placeholder
        return placeholder

    patterns = {
        "PHONE": r'\b(?:07\d{3}\s?\d{6}|0\d{3,4}\s?\d{3}\s?\d{3,4})\b',
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        "POSTCODE": r'\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b',
        "MICROCHIP": r'\b\d{15}\b',
        "DATE": r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b|\b\d{1,2}[\-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\-]\d{2,4}\b',
        "ID": r'\b\d{5,7}[A-Z]?\b(?=\s*MRCVS)',
    }

    redacted = text
    found_entities = []

    for pii_type, pattern in patterns.items():
        for match in re.finditer(pattern, redacted):
            value = match.group()
            placeholder = get_placeholder(pii_type, value)
            found_entities.append({
                "text": value, "type": pii_type,
                "start": match.start(), "end": match.end(),
                "placeholder": placeholder,
            })

    # Replace from end to start so character positions don't shift
    for entity in sorted(found_entities, key=lambda x: x["start"], reverse=True):
        redacted = redacted[:entity["start"]] + entity["placeholder"] + redacted[entity["end"]:]

    return redacted, found_entities


def ner_redact(text):
    """Step 2: Find and replace names, orgs, locations using spaCy NER."""
    doc = nlp(text)
    placeholder_map = {}
    counter = {}
    found_entities = []

    def get_placeholder(pii_type, value):
        """Assign a consistent indexed placeholder to each unique PII value."""
        if value in placeholder_map:
            return placeholder_map[value]
        count = counter.get(pii_type, 0) + 1
        counter[pii_type] = count
        placeholder = f"[{pii_type}_{count}]"
        placeholder_map[value] = placeholder
        return placeholder

    for ent in doc.ents:
        text_clean = ent.text.strip()

        # Smart filters to reduce false positives
        if text_clean in VET_STOPLIST:
            continue
        if text_clean.isupper() and len(text_clean) <= 5:
            continue
        if len(text_clean) <= 1:
            continue
        if any(c.isdigit() for c in text_clean):
            continue
        if text_clean.islower() and ent.label_ in ("PERSON", "ORG"):
            continue

        # Map spaCy labels to PII types
        if ent.label_ == "PERSON":
            pii_type = "PERSON"
        elif ent.label_ == "ORG":
            pii_type = "ORG"
        elif ent.label_ in ("GPE", "LOC"):
            pii_type = "LOCATION"
        else:
            continue

        found_entities.append({
            "text": ent.text, "type": pii_type,
            "start": ent.start_char, "end": ent.end_char,
            "placeholder": get_placeholder(pii_type, ent.text),
        })

    # Replace from end to start
    redacted = text
    for entity in sorted(found_entities, key=lambda x: x["start"], reverse=True):
        redacted = redacted[:entity["start"]] + entity["placeholder"] + redacted[entity["end"]:]

    return redacted, found_entities


def deidentify_text(text):
    """Run regex redaction first, then NER redaction on the result."""
    text_after_regex, regex_entities = regex_redact(text)
    redacted, ner_entities = ner_redact(text_after_regex)
    return redacted, regex_entities + ner_entities


def assess_k_anonymity(records, k=5):
    """Step 3: Check if quasi-identifier combinations can re-identify individuals."""
    groups = defaultdict(list)

    for i, record in enumerate(records):
        patterns = record.get("extracted", {}).get("vet_patterns", {})
        age = patterns.get("age", ["unknown"])[0] if patterns.get("age") else "unknown"
        weight = patterns.get("weight", ["unknown"])[0] if patterns.get("weight") else "unknown"
        sex = patterns.get("sex_neuter", ["unknown"])[0] if patterns.get("sex_neuter") else "unknown"
        key = (age, weight, sex)
        groups[key].append(i)

    risky = {key: members for key, members in groups.items() if len(members) < k}
    safe = {key: members for key, members in groups.items() if len(members) >= k}
    return risky, safe


def generalise_age(age_str):
    """Bucket age into ranges to reduce re-identification risk."""
    num = re.search(r'\d+', age_str)
    if not num:
        return "unknown"
    age = int(num.group())
    if "month" in age_str.lower() or "wk" in age_str.lower() or "week" in age_str.lower():
        return "juvenile"
    if age <= 2: return "0-2 years"
    if age <= 5: return "3-5 years"
    if age <= 10: return "6-10 years"
    return "11+ years"


def generalise_weight(weight_str):
    """Bucket weight into ranges to reduce re-identification risk."""
    num = re.search(r'\d+\.?\d*', weight_str)
    if not num:
        return "unknown"
    weight = float(num.group())
    if weight < 5: return "0-5kg"
    if weight < 10: return "5-10kg"
    if weight < 20: return "10-20kg"
    if weight < 40: return "20-40kg"
    return "40+kg"


def mitigate_risk(records, risky_groups):
    """Generalise age and weight in risky groups to achieve k-anonymity."""
    mitigated = 0
    for key, members in risky_groups.items():
        for idx in members:
            patterns = records[idx].get("extracted", {}).get("vet_patterns", {})
            if patterns.get("age"):
                patterns["age"] = [generalise_age(patterns["age"][0])]
            if patterns.get("weight"):
                patterns["weight"] = [generalise_weight(patterns["weight"][0])]
            mitigated += 1
    return mitigated


if __name__ == "__main__":
    # Load standardised records
    with open("data/outputs/peteval_standardised.json") as f:
        records = json.load(f)

    # Evaluate de-identification against ground truth
    true_positive = 0
    false_negative = 0
    false_positive = 0

    for record in records:
        spans = record["ground_truth"]["anon_spans"]
        if not spans:
            continue
        _, found = deidentify_text(record["raw_text"])
        gt_texts = {s["entity"].lower() for s in spans}
        found_texts = {e["text"].lower() for e in found}

        for gt in gt_texts:
            if any(gt in f or f in gt for f in found_texts):
                true_positive += 1
            else:
                false_negative += 1
        for f in found_texts:
            if not any(f in gt or gt in f for gt in gt_texts):
                false_positive += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"De-identification — Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # K-anonymity assessment
    risky, safe = assess_k_anonymity(records, k=5)
    print(f"Before mitigation — Risky groups: {len(risky)}, Records at risk: {sum(len(m) for m in risky.values())}")

    mitigated = mitigate_risk(records, risky)
    risky_after, safe_after = assess_k_anonymity(records, k=5)
    print(f"After mitigation  — Risky groups: {len(risky_after)}, Records at risk: {sum(len(m) for m in risky_after.values())}")

    # Run de-identification on all records
    for record in records:
        redacted, entities = deidentify_text(record["raw_text"])
        record["deidentified"] = {
            "redacted_text": redacted,
            "entities_found": entities,
        }

    # Save
    with open("data/outputs/peteval_deidentified.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} de-identified records")
