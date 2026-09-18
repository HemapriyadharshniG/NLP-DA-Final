import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset_processed_records")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_simplified_records")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "ai4bharat/MultiIndicSentenceSummarizationSS"

print(f"[*] Loading IndicBART summarization checkpoint: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False, use_fast=False, keep_accents=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

# Check for Apple Silicon GPU acceleration (MPS) or fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"[✓] Model loaded on compute device: {device}")

def simplify_tamil_text(text: str, max_tokens: int = 120) -> str:
    """Pass complex Tamil administrative text through IndicBART for plain summarization."""
    if not text or len(text.strip()) < 15:
        return text

    # Truncate overly long clauses to avoid context overflow
    input_text = text.strip()[:1024]
    
    # Format required by IndicBART: "Sentence <2ta>"
    formatted_input = f"{input_text} <2ta>"
    
    inputs = tokenizer(
        formatted_input,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Special language and boundary tokens
    bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
    eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
    pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
    decoder_start_id = tokenizer._convert_token_to_id_with_added_voc("<2ta>")

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            length_penalty=0.9,
            max_length=max_tokens,
            min_length=15,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            decoder_start_token_id=decoder_start_id
        )

    simplified_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Strip residual target language tags if generated
    simplified_text = simplified_text.replace("<2ta>", "").strip()
    return simplified_text

def run_stage5():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    print(f"[*] Processing {len(files)} records for Stage 5 simplification...")

    for f in files:
        file_path = os.path.join(INPUT_DIR, f)
        with open(file_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        raw_clause = data.get("order_clause", "")
        
        # Check if the extracted text contains Tamil characters
        has_tamil = any('\u0B80' <= c <= '\u0BFF' for c in raw_clause)
        
        if has_tamil and len(raw_clause.strip()) > 30:
            print(f"[*] Simplifying Tamil clause for: {f}...")
            simplified_output = simplify_tamil_text(raw_clause)
        else:
            # Fallback for English GO documents: take first core sentences
            simplified_output = " ".join(raw_clause.split("\n")[:4]).strip()

        data["simplified_plain_tamil"] = simplified_output

        out_path = os.path.join(OUTPUT_DIR, f)
        with open(out_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)

        print(f" [✓] Completed Stage 5: {f}")

    print(f"\n[✓] Stage 5 Complete! Simplified records saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_stage5()