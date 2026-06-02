# standardise.py — Stage 2: Four-tier standardisation cascade

import json
import re
import warnings
import logging
from collections import Counter

import numpy as np
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Lookup tables: known abbreviation → canonical term
lookup_tables = {
    "clinical_status": {
        "NAD": "No Abnormalities Detected",
        "BAR": "Bright Alert Responsive",
        "QAR": "Quiet Alert Responsive",
        "ADR": "Ain't Doing Right",
        "WNL": "Within Normal Limits",
        "DRB": "Dull Responsive Bright",
    },
    "vitals": {
        "HR": "Heart Rate",
        "RR": "Respiratory Rate",
        "CRT": "Capillary Refill Time",
        "MM": "Mucous Membranes",
        "BCS": "Body Condition Score",
        "TPR": "Temperature Pulse Respiration",
    },
    "exam": {
        "CE": "Clinical Examination",
        "PE": "Physical Examination",
        "HPC": "Health Plan Check",
        "ROM": "Range of Motion",
        "FNA": "Fine Needle Aspirate",
        "GA": "General Anaesthesia",
        "BIOP": "Biopsy",
    },
    "vaccination": {
        "KC": "Kennel Cough",
        "DHP": "Distemper Hepatitis Parvovirus",
    },
    "limb": {
        "RH": "Right Hind",
        "LH": "Left Hind",
        "RF": "Right Fore",
        "LF": "Left Fore",
        "LHS": "Left Hand Side",
    },
    "medication": {
        "BID": "Twice Daily",
        "SID": "Once Daily",
        "TID": "Three Times Daily",
        "QID": "Four Times Daily",
        "EOD": "Every Other Day",
        "PRN": "As Needed",
        "PO": "Per Os (Oral)",
        "IV": "Intravenous",
        "SC": "Subcutaneous",
        "IM": "Intramuscular",
        "NSAID": "Non-Steroidal Anti-Inflammatory Drug",
    },
    "diagnostic": {
        "FB": "Foreign Body",
        "GI": "Gastrointestinal",
        "OA": "Osteoarthritis",
        "POC": "Post-Operative Check",
        "RV": "Revisit",
        "AG": "Anal Glands",
        "STI": "Soft Tissue Injury",
        "GI upset": "Acute gastroenteritis",
        "GI problems": "Acute gastroenteritis",
        "GI probs": "Acute gastroenteritis",
    },
    "professional": {
        "MRCVS": "Member of the Royal College of Veterinary Surgeons",
    },
}

# Flatten nested lookup tables into a single dict for quick access
flat_lookup = {}
for category, terms in lookup_tables.items():
    for abbr, full in terms.items():
        flat_lookup[abbr] = {"full_term": full, "category": category}


def parse_value(raw, pattern_type):
    """Split a raw value string into numeric value + unit."""
    num = re.search(r'\d+\.?\d*', raw)
    if not num:
        return {"raw": raw, "method": "unmatched"}
    value = float(num.group())
    if pattern_type == "age":
        unit = re.search(r'(years?|months?|wks?|weeks?|yo|yr)', raw, re.IGNORECASE)
        return {"value": value, "unit": unit.group() if unit else "unknown", "method": "parsed"}
    if pattern_type == "temperature":
        return {"value": value, "unit": "°C", "method": "parsed"}
    if pattern_type == "weight":
        unit = re.search(r'(kg|lbs|g)', raw, re.IGNORECASE)
        return {"value": value, "unit": unit.group().lower() if unit else "unknown", "method": "parsed"}
    if pattern_type == "dosage":
        unit = re.search(r'(mg/kg|mg|ml|IU|mcg)', raw, re.IGNORECASE)
        return {"value": value, "unit": unit.group() if unit else "unknown", "method": "parsed"}
    return {"raw": raw, "method": "unmatched"}


def standardise_exact(record):
    """Tier 1: Map abbreviations via lookup table, parse numeric values."""
    results = {}
    patterns = record["extracted"]["vet_patterns"]
    for pattern_type, matches in patterns.items():
        results[pattern_type] = []
        for match in matches:
            abbr = re.match(r'[A-Za-z]+', match)
            abbr = abbr.group() if abbr else match
            if abbr.upper() in flat_lookup:
                results[pattern_type].append({
                    "raw": match,
                    "standard": flat_lookup[abbr.upper()]["full_term"],
                    "category": flat_lookup[abbr.upper()]["category"],
                    "method": "exact_match",
                })
            elif abbr in flat_lookup:
                results[pattern_type].append({
                    "raw": match,
                    "standard": flat_lookup[abbr]["full_term"],
                    "category": flat_lookup[abbr]["category"],
                    "method": "exact_match",
                })
            else:
                parsed = parse_value(match, pattern_type)
                parsed["raw"] = match
                results[pattern_type].append(parsed)
    return results


# Canonical breed list for fuzzy matching
canonical_breeds = [
    "Labrador Retriever", "German Shepherd", "Golden Retriever",
    "French Bulldog", "Bulldog", "Poodle", "Beagle", "Rottweiler",
    "Yorkshire Terrier", "Boxer", "Cavalier King Charles Spaniel",
    "Staffordshire Bull Terrier", "Cocker Spaniel", "Border Collie",
    "Jack Russell Terrier", "Shih Tzu", "Dachshund", "Chihuahua",
    "Greyhound", "Whippet", "Springer Spaniel", "Westie",
    "Pomeranian", "Maltese", "Husky", "Dalmatian",
]

# Canonical diagnosis list for semantic and fuzzy matching
canonical_diagnoses = [
    "Acute gastroenteritis", "Otitis externa", "Otitis media",
    "Skin allergy", "Atopic dermatitis", "Flea allergy dermatitis",
    "Urinary tract infection", "Kennel cough", "Parvovirus",
    "Cruciate ligament rupture", "Osteoarthritis", "Hip dysplasia",
    "Dental disease", "Periodontal disease", "Gastric foreign body",
    "Pancreatitis", "Diabetes mellitus", "Cushings disease",
    "Hypothyroidism", "Hyperthyroidism", "Renal failure",
    "Heart murmur", "Cardiac disease", "Liver disease",
    "Inflammatory bowel disease", "Colitis", "Conjunctivitis",
    "Corneal ulcer", "Ear infection", "Anal gland impaction",
    "Pyoderma", "Hot spot", "Wound infection", "Abscess",
    "Vomiting", "Diarrhoea", "Weight loss", "Obesity",
    "Seizures", "Epilepsy", "Anxiety", "Aggression",
    "Mammary tumour", "Lymphoma", "Mast cell tumour",
    "Gastritis", "Degenerative disc disease", "Patella luxation",
    "Oesophageal disease", "Asthma", "Sebaceous cyst",
    "Muscle strain", "Injection site reaction", "Myelosuppression",
]


def fuzzy_match(term, candidates, threshold=75):
    """Tier 2: Match typos/variants using weighted ratio similarity."""
    if len(term) < 4:
        return None
    result = process.extractOne(term, candidates, scorer=fuzz.WRatio, score_cutoff=threshold)
    if result:
        return {"matched": result[0], "score": result[1], "method": "fuzzy_match"}
    return None


def fuzzy_match_diagnosis(term, threshold=85):
    """Tier 2 (strict): Match diagnosis terms using full string ratio."""
    if len(term) < 4:
        return None
    result = process.extractOne(term, canonical_diagnoses, scorer=fuzz.ratio, score_cutoff=threshold)
    if result:
        return {"matched": result[0], "score": result[1], "method": "fuzzy_match"}
    return None


# Load S-PubMedBERT for semantic similarity
model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
diagnosis_embeddings = model.encode(canonical_diagnoses)


def semantic_match(term, threshold=0.86):
    """Tier 3: Match by meaning using PubMedBERT cosine similarity."""
    term_embedding = model.encode([term])
    similarities = cosine_similarity(term_embedding, diagnosis_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    if best_score >= threshold:
        return {"matched": canonical_diagnoses[best_idx], "score": float(best_score), "method": "semantic_match"}
    return None


# Load BART for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

candidate_labels = [
    "Acute gastroenteritis", "Otitis externa", "Skin allergy",
    "Dental disease", "Osteoarthritis", "Cruciate ligament rupture",
    "Urinary tract infection", "Vomiting", "Diarrhoea",
    "Wound infection", "Abscess", "Hot spot", "Obesity",
    "Pancreatitis", "Kennel cough", "Flea allergy dermatitis",
    "Conjunctivitis", "Ear infection", "Anal gland impaction",
    "Seizures",
]


def zero_shot_diagnose(clinical_text, threshold=0.2):
    """Tier 4: Predict diagnosis from full clinical note when all else fails."""
    result = classifier(clinical_text[:512], candidate_labels)
    if result["scores"][0] >= threshold:
        return {"matched": result["labels"][0], "score": result["scores"][0], "method": "zero_shot"}
    return None


def standardise_record(record, use_zero_shot=False):
    """Run all four tiers in cascade on a single record."""
    results = standardise_exact(record)

    disease_results = []
    for span in record["ground_truth"]["disease_spans"]:
        entity = span["entity"]

        # Tier 1: exact lookup
        if entity.upper() in flat_lookup:
            disease_results.append({"raw": entity, "standard": flat_lookup[entity.upper()]["full_term"], "method": "exact_match"})
            continue

        # Tier 2: fuzzy match
        fuzzy = fuzzy_match_diagnosis(entity)
        if fuzzy:
            disease_results.append({"raw": entity, "standard": fuzzy["matched"], "score": fuzzy["score"], "method": "fuzzy_match"})
            continue

        # Tier 3: semantic match
        semantic = semantic_match(entity)
        if semantic:
            disease_results.append({"raw": entity, "standard": semantic["matched"], "score": semantic["score"], "method": "semantic_match"})
            continue

        # No match found
        disease_results.append({"raw": entity, "standard": entity, "method": "unmatched"})

    # Tier 4: zero-shot fallback when no disease was found
    if not disease_results and use_zero_shot:
        zs = zero_shot_diagnose(record["raw_text"])
        if zs:
            disease_results.append({"raw": None, "standard": zs["matched"], "score": zs["score"], "method": "zero_shot"})

    results["diagnoses"] = disease_results
    return results


if __name__ == "__main__":
    # Load extracted records
    with open("data/outputs/peteval_extracted.json") as f:
        records = json.load(f)

    # Standardise all records (zero-shot disabled for speed)
    for record in records:
        record["standardised"] = standardise_record(record, use_zero_shot=False)

    # Print method distribution
    methods = Counter()
    for r in records:
        for d in r["standardised"]["diagnoses"]:
            methods[d["method"]] += 1
    print(f"Standardised {len(records)} records")
    print(f"Diagnosis methods: {methods}")

    # Save
    with open("data/outputs/peteval_standardised.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records")
