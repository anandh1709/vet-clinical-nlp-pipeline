# Veterinary Clinical NLP Pipeline

An end-to-end NLP pipeline for standardising and de-identifying veterinary clinical records. Built on the [PetEVAL](https://huggingface.co/datasets/SAVSNET/PetEVAL) benchmark dataset, 4,999 real-world veterinary electronic health records from UK first-opinion practices collected through the Small Animal Veterinary Surveillance Network (SAVSNET).

## Motivation

Veterinary clinical records are messy. Different clinics use different abbreviations ("BAR" vs "Bright Alert Responsive"), inconsistent formatting ("Hr=100" vs "HR 104"), and embed personally identifiable information in free-text notes. This makes the data difficult to analyse at scale, share across institutions, or use for research without risking patient/owner re-identification.

This project addresses three problems simultaneously: extracting structured information from unstructured clinical narratives, standardising terminology to a canonical vocabulary, and removing identifiers while assessing re-identification risk from quasi-identifiers. A data quality framework measures improvement across six dimensions before and after processing.

## Pipeline Architecture

Raw clinical notes -> **Entity Extraction** -> **Standardisation** -> **De-identification** -> **Data Quality Scoring** -> Clean, standardised, de-identified records with quality metrics.

### Stage 1: Entity Extraction (`extract.py`)

Two parallel extraction engines merged into a unified output:

**spaCy NER** detects general named entities (person names, organisations, locations, dates) in free-text clinical narratives using the `en_core_web_sm` model. A veterinary-specific stoplist filters out false positives where clinical abbreviations like "BAR", "GI", "BC" are misclassified as organisations or persons.

**Rule-based regex matchers** extract veterinary-specific patterns that no general NER model would recognise: vital signs (Hr=100, CRT=1, T=39.1), weight (32kg, 5.2kg), age (4yo, 6 months, 8wk), sex/neuter status (FS, MN, MC), clinical status abbreviations (BAR, NAD, WNL), dosage patterns (15mg/kg BID), and temperature readings.

Smart filters on the NER output reduce false positives: all-uppercase tokens ≤5 characters are skipped (likely abbreviations), tokens containing digits are skipped (not names), and lowercase-only tokens labelled as PERSON/ORG are skipped (names are capitalised).

### Stage 2: Standardisation (`standardise.py`)

A four-tier cascade where each tier catches what the previous one missed:

**Tier 1 — Exact Lookup Tables.** Hand-curated dictionary mappings for 40+ known veterinary abbreviations across seven categories: clinical status, vitals, exam types, vaccination, limb references, medication routes/frequencies, and diagnostic terms. A value parser handles numeric fields (weight, age, temperature, dosage) by splitting them into value + unit. Deterministic, instant, handles the bulk of known variants.

**Tier 2 — Fuzzy Matching (rapidfuzz).** Catches typos and spelling variants using `fuzz.WRatio` for breed names (threshold 75) and `fuzz.ratio` for diagnosis terms (threshold 85, stricter to avoid false matches). Examples: "Labardor" → "Labrador Retriever", "Staffodshire Bull Terrier" → "Staffordshire Bull Terrier", "abcess" → "Abscess".

**Tier 3 — Semantic Matching (S-PubMedBERT).** For terms where the surface form is completely different but the meaning is equivalent. Uses `pritamdeka/S-PubMedBert-MS-MARCO` (PubMedBERT fine-tuned for sentence similarity) to embed diagnosis terms and find nearest neighbours by cosine similarity (threshold 0.86) against a canonical vocabulary of 55 diagnoses. Examples: "otitis" → "Otitis media" (0.98), "allergic skin disease" → "Skin allergy" (0.83), "arthritis" → "Osteoarthritis" (0.94).

**Tier 4 — Zero-Shot Classification (BART).** When the diagnosis field is missing entirely or unmatched by all three tiers, `facebook/bart-large-mnli` reads the full clinical note and predicts the most likely diagnosis from a candidate list of 20 common conditions. Only invoked as a last resort due to inference cost (~1-2 seconds per record). Optional and disabled by default for batch processing.

**Standardisation Results:**
| Method | Count | Percentage |
|--------|-------|------------|
| Semantic match | 1,038 | 77% |
| Fuzzy match | 130 | 10% |
| Exact lookup | 35 | 3% |
| Unmatched | 153 | 11% |

89% standardisation rate across Tiers 1–3 without zero-shot.

### Stage 3: De-identification (`deidentify.py`)

Three sub-stages:

**Regex-based redaction** targets structured identifier patterns: UK phone numbers (07xxx xxxxxx), email addresses, UK postcodes (A9 9AA format), microchip numbers (15-digit ISO), dates in multiple formats (DD/MM/YYYY, DD-Mon-YY), and MRCVS registration numbers. Each match is replaced with an indexed placeholder ([PHONE_1], [DATE_1]) that preserves referential structure without leaking the actual value.

**NER-based redaction** uses spaCy to detect person names, organisation names, and locations in free text, replacing them with [PERSON_1], [ORG_1], [LOCATION_1]. Smart filters (described in Stage 1) suppress false positives from clinical vocabulary.

**Quasi-identifier risk assessment** checks whether combinations of remaining non-redacted fields (age, weight, sex) could re-identify individuals through a k-anonymity analysis (k=5). Records are grouped by quasi-identifier combinations, and groups with fewer than k members are flagged as risky. Automatic mitigation generalises values: age is bucketed (e.g., "4 years" → "3–5 years"), weight is banded (e.g., "32kg" → "20–40kg"). This reduced risky groups from 419 (509 records) to 22 (29 records) — a 94% reduction in re-identification risk.

**De-identification Performance (evaluated against PetEVAL expert annotations):**
| Metric | Score |
|--------|-------|
| Precision | 0.490 |
| Recall | 0.541 |
| F1 Score | 0.514 |

The primary limitation is pet name detection, names like "Novaa", "Lumman", "Lunah" are unusual tokens that general-purpose NER models don't recognise. The PetEVAL paper itself identifies this as one of the hardest challenges in veterinary de-identification. A fine-tuned domain-specific model (e.g., PetBERT) would improve recall.

### Stage 4: Data Quality Scoring (`dataquality.py`)

Six dimensions scored 0–1, computed on both the raw input and the pipeline output:

**Completeness** — proportion of expected fields that are present and non-empty. **Accuracy** : proportion of field values that match canonical vocabularies. **Consistency** : cross-field logical checks (weight 0.1–100kg, temperature 35–42°C, age 0–30 years). **Validity** : format validation (numeric values have units, text fields are non-empty). **Uniqueness** : duplicate detection using raw text as a composite key. **Standardisation rate** : proportion of extracted terms successfully mapped to canonical forms.

**Quality Improvement:**
| Dimension | Before | After | Delta |
|-----------|--------|-------|-------|
| Completeness | 0.677 | 0.839 | +0.161 |
| Accuracy | 0.000 | 0.966 | +0.966 |
| Consistency | 0.500 | 0.982 | +0.482 |
| Validity | 1.000 | 1.000 | +0.000 |
| Uniqueness | 1.000 | 1.000 | +0.000 |
| Standardisation | 0.000 | 0.966 | +0.966 |
| **Overall** | **0.530** | **0.959** | **+0.429** |

## Project Structure

vet-clinical-nlp-pipeline/
│
├── data/
│   ├── raw/
│   │   └── peteval_records.json
│   │
│   └── outputs/
│       ├── peteval_extracted.json
│       ├── peteval_standardised.json
│       ├── peteval_deidentified.json
│       └── peteval_final.json
│
├── src/
│   ├── load_data.py
│   ├── extract.py
│   ├── standardise.py
│   ├── deidentify.py
│   └── quality.py
│
├── requirements.txt
└── README.md

## Setup

### Prerequisites

Python 3.10+ and a HuggingFace account with access to the PetEVAL dataset.

### Installation

```bash
git clone https://github.com/yourusername/vet-nlp-pipeline.git
cd vet-nlp-pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### HuggingFace Authentication

The PetEVAL dataset is gated. You need to agree to the terms at [huggingface.co/datasets/SAVSNET/PetEVAL](https://huggingface.co/datasets/SAVSNET/PetEVAL) and authenticate:

```bash
hf auth login
```

### Running the Pipeline

Run each stage in order:

```bash
python src/load_data.py
python src/extract.py
python src/standardise.py
python src/deidentify.py
python src/quality.py
```

## Dataset

This project uses **PetEVAL** (Farrell et al., 2025), the first publicly available benchmark for veterinary EHR NLP. The test split contains 4,999 clinical records from 253 UK veterinary practices with three expert annotation layers: anonymisation NER spans (8,244 labels), ICD-11 syndromic chapter classifications (20,408 labels), and disease NER spans (429 labels). Licensed under Apache 2.0.

## Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| NER | spaCy (en_core_web_sm) | Name, organisation, location detection |
| Pattern matching | Python regex | Vital signs, dosages, weights, PII patterns |
| Fuzzy matching | rapidfuzz | Typo correction for breeds and diagnoses |
| Semantic matching | S-PubMedBert-MS-MARCO | Meaning-level diagnosis mapping |
| Zero-shot classification | facebook/bart-large-mnli | Diagnosis prediction from clinical notes |
| Data handling | pandas, numpy | Record processing and analysis |

## References

1. Farrell, S., Radford, A., Al Moubayed, N., & Noble, P.J.M. (2025). PetEVAL: A veterinary free text electronic health records benchmark. *BioNLP Workshop, ACL 2025*.

2. Brundage et al. (2026). Synthetic Data for Veterinary EHR De-identification: Benefits, Limits, and Safety Trade-offs Under Fixed Compute. *arXiv:2601.09756*.

3. Farrell, S., Radford, A.D., & Noble, P.J.M. (2024). Text mining for disease surveillance in veterinary clinical data: Parts 1 & 2. *Frontiers in Veterinary Science*, 11.

4. Boguslav, M.R. et al. (2025). Fine-tuning foundational models to code diagnoses from veterinary health records. *arXiv:2410.15186*.

5. Zhang et al. (2018). DeepTag: inferring diagnoses from veterinary clinical notes. *npj Digital Medicine*, 1(1).

## License

The PetEVAL dataset is licensed under Apache 2.0 by SAVSNET.
