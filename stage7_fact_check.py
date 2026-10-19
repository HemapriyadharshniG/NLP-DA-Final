import os
import re
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset_bilingual_records")

POLITICAL_NAMES = [
    "எடப்பாடி", "பழனிசாமி", "ஸ்டாலின்", "மு.க.ஸ்டாலின்", "மு.க. ஸ்டாலின்",
    "ஜெயலலிதா", "கருணாநிதி", "பன்னீர்செல்வம்", "ஓ.பன்னீர்செல்வம்", "மோடி", "அமித்ஷா"
]
POLITICAL_PARTIES = ["திமுக", "அதிமுக", "காங்கிரஸ்", "பாஜக", "பாமக", "கம்யூனிஸ்ட்", "நாம் தமிழர்"]
TITLES = ["முதலமைச்சர்", "பிரதமர்"]
NEWS_MEDIA = [
    "நிதர்சனம்", "தந்தி டிவி", "நக்கீரன்", "தினமலர்", "தினத்தந்தி",
    "மாலை மலர்", "புதிய தலைமுறை", "பாலிமர்", "நியூஸ் 18", "பிபிசி"
]

def extract_entities(text: str) -> set:
    """Extract currency figures, percentages, and dates."""
    numbers = set(re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text))
    dates = set(re.findall(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b', text))
    return numbers.union(dates)

def check_hallucinated_names(source: str, summary: str) -> list[str]:
    """Check if model introduced any politician, CM title, party, or news tag NOT in source."""
    hallucinated = []
    for name in POLITICAL_NAMES + POLITICAL_PARTIES + TITLES + NEWS_MEDIA:
        if name in summary and name not in source:
            hallucinated.append(name)
    return hallucinated

def verify_records():
    files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(".json")]
    print(f"[*] Running hallucination & factual verification on {len(files)} records...\n")

    passed_count = 0
    flagged_count = 0

    for f in files:
        with open(os.path.join(INPUT_DIR, f), "r", encoding="utf-8") as infile:
            data = json.load(infile)

        source_text = data.get("order_clause", "")
        summary_text = data.get("simplified_plain_tamil", "")

        orig_entities = extract_entities(source_text)
        simp_entities = extract_entities(summary_text)

        # Check numerical hallucinations
        hallucinated_numbers = simp_entities - orig_entities
        # Check political / named entity hallucinations (e.g. Chief Minister name)
        hallucinated_names = check_hallucinated_names(source_text, summary_text)

        is_flagged = bool(hallucinated_numbers or hallucinated_names)
        status = "FLAGGED" if is_flagged else "PASSED"

        if is_flagged:
            flagged_count += 1
            print(f"[{status}] {f}")
            if hallucinated_names:
                print(f"   ⚠️ Hallucinated entities/names: {hallucinated_names}")
            if hallucinated_numbers:
                print(f"   ⚠️ Unverified numbers/dates: {hallucinated_numbers}")
        else:
            passed_count += 1
            print(f"[{status}] {f}")

    print(f"\n[✓] Verification Finished: {passed_count} Passed, {flagged_count} Flagged.")

if __name__ == "__main__":
    verify_records()