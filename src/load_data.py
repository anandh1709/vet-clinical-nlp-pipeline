# load_data.py — Stage 0: Load and prepare PetEVAL dataset

import json
import ast
from datasets import load_dataset

def safe_parse(val):
    """Safely convert stringified Python literals to actual objects."""
    if not val or val == "[]":
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        try:
            return ast.literal_eval(val + "]")
        except:
            return []

def load_and_prepare():
    """Download PetEVAL, parse annotations, structure records, save as JSON."""
    # Download test split from HuggingFace (4,999 records)
    dataset = load_dataset("SAVSNET/PetEVAL", split="test")
    df = dataset.to_pandas()

    # Parse stringified annotation columns into Python objects
    df["icd_label"] = df["icd_label"].apply(safe_parse)
    df["annonymisation"] = df["annonymisation"].apply(safe_parse)
    df["disease"] = df["disease"].apply(safe_parse)

    # Build structured record dicts for the pipeline
    records = []
    for _, row in df.iterrows():
        record = {
            "record_id": int(row["id"]),
            "raw_text": row["sentence"],
            "ground_truth": {
                "icd_labels": row["icd_label"],
                "anon_spans": row["annonymisation"],
                "disease_spans": row["disease"],
            },
            "extracted": {},
            "standardised": {},
            "deidentified": {},
            "quality_scores": {},
        }
        records.append(record)

    # Save locally so other stages don't need HuggingFace
    with open("data/raw/peteval_records.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"Loaded and saved {len(records)} records")
    return records

if __name__ == "__main__":
    load_and_prepare()
