import os
import re
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset_bilingual_records")

def extract_entities(text: str) -> set:
    """Extract currency figures, percentages, and dates."""
    numbers = set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text))
    dates = set(re.findall(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b', text))
    return numbers.union(dates)

def verify_records():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    print(f"[*] Running hallucination / factual verification on {len(files)} records...\n")

    for f in files:
        with open(os.path.join(INPUT_DIR, f), "r", encoding="utf-8") as infile:
            data = json.load(infile)

        orig_entities = extract_entities(data.get("order_clause", ""))
        simp_entities = extract_entities(data.get("simplified_plain_tamil", ""))

        # Check if model introduced any number NOT present in original text
        hallucinated_numbers = simp_entities - orig_entities

        status = "PASSED" if not hallucinated_numbers else "FLAGGED"
        print(f"[{status}] {f}")
        if hallucinated_numbers:
            print(f"   ⚠️ Unverified numbers found in summary: {hallucinated_numbers}")

if __name__ == "__main__":
    verify_records()