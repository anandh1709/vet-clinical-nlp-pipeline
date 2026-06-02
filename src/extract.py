# extract.py — Stage 1: Entity Extraction using spaCy NER and regex matchers

import json
import re
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Terms spaCy misclassifies as PERSON/ORG — skip these during NER
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


def extract_ner_entities(text):
    """Run spaCy NER and return PERSON, ORG, GPE, LOC, DATE entities."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE"):
            if ent.text.strip() not in VET_STOPLIST:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
    return entities


def extract_vet_patterns(text):
    """Extract vet-specific patterns using regex: vitals, weight, age, etc."""
    patterns = {
        "vital_sign": r'\b(?:HR|Hr|hr|RR|rr|CRT|Crt|crt|BP)\s*[=:]?\s*\d+\.?\d*',
        "temperature": r'\bT\s*[=:]?\s*\d+\.\d+',
        "weight": r'\b\d+\.?\d*\s*(?:kg|Kg|KG|lbs|g)\b',
        "age": r'\b\d+\s*(?:yo|y\.?o\.?|yr|yrs|years?|mo|months?|wk|wks|weeks?)\b',
        "sex_neuter": r'\b(?:FS|FN|MN|MC|MI|FI|FE|ME|MIFE)\b',
        "clinical_status": r'\b(?:BAR|NAD|QAR|ADR|TPR|DRB|BARH)\b',
        "dosage": r'\b\d+\.?\d*\s*(?:mg/kg|mg|ml|IU|mcg)(?:\s*(?:PO|IV|SC|IM|SID|BID|TID|QID|EOD|PRN|q\d+h))?\b',
    }
    extracted = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            extracted[name] = matches
    return extracted


def extract_all(record):
    """Merge NER and regex extractions into one result."""
    text = record["raw_text"]
    return {
        "ner_entities": extract_ner_entities(text),
        "vet_patterns": extract_vet_patterns(text),
    }


if __name__ == "__main__":
    # Load raw records
    with open("data/raw/peteval_records.json") as f:
        records = json.load(f)

    # Run extraction on all records
    for record in records:
        record["extracted"] = extract_all(record)

    # Save
    with open("data/outputs/peteval_extracted.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Extracted entities from {len(records)} records")
