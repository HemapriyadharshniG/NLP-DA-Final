import os
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset_simplified_records")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_bilingual_records")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print(f"[*] Loading Translation Engine ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="tam_Taml")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()
print(f"[✓] Translator loaded on: {device}")

def translate_ta_to_en(tamil_text: str) -> str:
    """Translates simplified Tamil into plain English."""
    if not tamil_text or len(tamil_text.strip()) < 5:
        return ""

    inputs = tokenizer(tamil_text.strip(), return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # English target code in NLLB
    tgt_lang_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_token_id,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0].strip()

def run_translation():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    print(f"[*] Translating {len(files)} records to English...")

    for f in files:
        path = os.path.join(INPUT_DIR, f)
        with open(path, "r", encoding="utf-8") as infile:
            record = json.load(infile)

        plain_tamil = record.get("simplified_plain_tamil", "")
        english_summary = translate_ta_to_en(plain_tamil)
        record["english_summary"] = english_summary

        out_path = os.path.join(OUTPUT_DIR, f)
        with open(out_path, "w", encoding="utf-8") as outfile:
            json.dump(record, outfile, ensure_ascii=False, indent=2)

        print(f" [✓] Translated: {f}")

    print(f"\n[✓] Stage 6 Complete! Bilingual records saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_translation()