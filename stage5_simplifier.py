import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset_processed_records")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_simplified_records")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "ai4bharat/MultiIndicSentenceSummarizationSS"

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

ADMIN_LEXICON = {
    "பரிசீலித்து": "ஆராய்ந்து",
    "ஆணை பிறப்பிக்கப்படுகிறது": "உத்தரவிடப்படுகிறது",
    "ஆணை வெளியிடப்படுகிறது": "அறிவிக்கப்படுகிறது",
    "பற்று வைக்கப்படும்": "செலவிடப்படும்",
    "நிதி ஒதுக்கீடு": "பண ஒதுக்கீடு",
    "கருத்துரு": "பரிந்துரை / கோரிக்கை",
    "வைப்பு நிதி": "சேமிப்புத் தொகை",
    "பயனாளிகள்": "உதவி பெறுவோர்",
    "நடைமுறைப்படுத்த": "செயல்படுத்த",
    "ஒப்புதல் அளிக்கப்படுகிறது": "அனுமதி தரப்படுகிறது",
    "நிர்வாக அனுமதி": "அதிகாரப்பூர்வ அனுமதி",
    "விடுவித்து": "வழங்கி",
    "ஒப்பளிப்பு": "அனுமதி",
    "சுற்றறிக்கை": "அரசு தகவல் அறிக்கை",
    "வழிகாட்டு நெறிமுறைகள்": "செயல்முறை விதிகள்",
    "பணிமேற்பார்வையாளர்": "வேலை மேற்பார்வையாளர்",
    "அறிவுறுத்தப்படுகிறது": "உத்தரவிடப்படுகிறது",
    "மேற்கொள்ளப்பட வேண்டும்": "செய்யப்பட வேண்டும்"
}

def clean_news_artifacts(text: str) -> str:
    text = re.sub(r'</s>', '', text)
    text = re.sub(r'<2ta>', '', text)
    text = re.sub(r'\.\.!+', '.', text)
    text = re.sub(
        r'\s*[:\-–]\s*(?:நிதர்சனம்|தந்தி டிவி|செய்திகள்|நக்கீரன்|தினமலர்|தினத்தந்தி|மாலை மலர்|புதிய தலைமுறை|பாலிமர்).*$',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(r'\(?\b(?:திமுக|அதிமுக|பாஜக|காங்கிரஸ்)\b\)?', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def detect_hallucinations(source_chunk: str, generated_summary: str) -> list[str]:
    hallucinations = []
    for name in POLITICAL_NAMES + POLITICAL_PARTIES + TITLES + NEWS_MEDIA:
        if name in generated_summary and name not in source_chunk:
            hallucinations.append(name)
    return hallucinations

def apply_lexicon_simplification(text: str) -> str:
    res = text
    for formal, plain in ADMIN_LEXICON.items():
        if formal in res:
            res = res.replace(formal, plain)
    return res

print(f"[*] Loading IndicBART summarization checkpoint: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"[✓] Model loaded on compute device: {device}")

def simplify_tamil_text(text: str, max_tokens: int = 140) -> str:
    """Pass complex Tamil administrative text through IndicBART with anti-hallucination guardrails."""
    if not text or len(text.strip()) < 15:
        return text

    input_text = text.strip()[:600]
    formatted_input = f"{input_text} <2ta>"
    
    inputs = tokenizer(
        formatted_input,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    decoder_start_id = tokenizer._convert_token_to_id_with_added_voc("<2ta>")

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            length_penalty=0.8,
            max_length=max_tokens,
            min_length=10,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True,
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            decoder_start_token_id=decoder_start_id
        )

    simplified_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    simplified_text = clean_news_artifacts(simplified_text)

    # Check for hallucinated Chief Minister names or political entities
    hallucinations = detect_hallucinations(text, simplified_text)
    if hallucinations:
        print(f"   ⚠️ Hallucination detected ({hallucinations}), using grounded lexicon simplification.")
        return apply_lexicon_simplification(text[:250].strip())

    return simplified_text if len(simplified_text) > 10 else apply_lexicon_simplification(text[:250].strip())

def run_stage5():
    files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(".json")]
    print(f"[*] Processing {len(files)} records for Stage 5 simplification...")

    for f in files:
        file_path = os.path.join(INPUT_DIR, f)
        with open(file_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        raw_clause = data.get("order_clause", "")
        has_tamil = any('\u0B80' <= c <= '\u0BFF' for c in raw_clause)
        
        if has_tamil and len(raw_clause.strip()) > 30:
            print(f"[*] Simplifying Tamil clause for: {f}...")
            simplified_output = simplify_tamil_text(raw_clause)
        else:
            simplified_output = " ".join(raw_clause.split("\n")[:4]).strip()

        data["simplified_plain_tamil"] = simplified_output

        out_path = os.path.join(OUTPUT_DIR, f)
        with open(out_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)

        print(f" [✓] Completed Stage 5: {f}")

    print(f"\n[✓] Stage 5 Complete! Simplified records saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_stage5()