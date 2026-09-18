import os
import re
import json
import unicodedata
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PDF_DIR = os.path.join(BASE_DIR, "dataset_raw_gos")
CLEAN_TEXT_DIR = os.path.join(BASE_DIR, "dataset_cleaned_text")
PROCESSED_JSON_DIR = os.path.join(BASE_DIR, "dataset_processed_records")

os.makedirs(CLEAN_TEXT_DIR, exist_ok=True)
os.makedirs(PROCESSED_JSON_DIR, exist_ok=True)

# -------------------------------------------------------------
# Stage 4 Lexicon: Tamil Bureaucratic -> Plain Citizen Register
# -------------------------------------------------------------
TAMIL_ADMIN_LEXICON = {
    "பரிசீலித்து": "ஆராய்ந்து",
    "ஆணை பிறப்பிக்கப்படுகிறது": "உத்தரவிடப்படுகிறது",
    "ஆணை வெளியிடப்படுகிறது": "அறிவிக்கப்படுகிறது",
    "பற்று வைக்கப்படும்": "செலவிடப்படும்",
    "நிதி ஒதுக்கீடு": "பண உதவி",
    "கருத்துரு": "பரிந்துரை",
    "வைப்பு நிதி": "சேமிப்பு தொகை",
    "பயனாளிகள்": "உதவி பெறுவோர்",
    "நடைமுறைப்படுத்த": "செயல்படுத்த",
    "ஒப்புதல் அளிக்கப்படுகிறது": "அனுமதி தரப்படுகிறது",
    "நிர்வாக அனுமதி": "அதிகாரப்பூர்வ அனுமதி",
    "விடுவித்து": "வழங்கி"
}

# -------------------------------------------------------------
# Stage 1 & 2: Ingestion, OCR Fallback, Normalisation
# -------------------------------------------------------------
def is_tamil(text: str) -> bool:
    return any('\u0B80' <= c <= '\u0BFF' for c in text)

def clean_and_normalize(text: str) -> str:
    """Normalize Unicode to NFC form and strip scanner/layout noise."""
    text = unicodedata.normalize('NFC', text)
    # Remove recurring header/footer noise patterns & standalone pagination
    text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*[-—–]+\s*$', '', text, flags=re.MULTILINE)
    # Replace repeated periods or underscore leader lines
    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r'_{2,}', ' ', text)
    # Normalize excessive white spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract embedded digital text, fallback to OCR if empty or scanned."""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_text = page.get_text("text")
        
        # Fallback to OCR when digital text layer is inadequate
        if len(page_text.strip()) < 40:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                # Tesseract OCR with Tamil + English models
                page_text = pytesseract.image_to_string(img, lang="tam+eng")
            except Exception:
                page_text = pytesseract.image_to_string(img)
                
        full_text += page_text + "\n"
        
    doc.close()
    return clean_and_normalize(full_text)

# -------------------------------------------------------------
# Stage 3: Structural Parsing (Preamble, Clauses, Metadata)
# -------------------------------------------------------------
def parse_go_structure(cleaned_text: str) -> dict:
    parsed = {
        "go_number": "Not explicitly specified",
        "date": "Not explicitly specified",
        "references": [],
        "preamble": "",
        "order_clause": ""
    }

    # Extract GO Number (Tamil & English signatures)
    go_pattern = r'(?:அரசாணை\s*(?:\([^\)]+\))?\s*எண்\.?|G\.O\.?\s*(?:\([^\)]+\))?\s*(?:Ms|Rt)?\.?\s*No\.?)\s*[:.]?\s*([^\n,]+)'
    go_match = re.search(go_pattern, cleaned_text, re.IGNORECASE)
    if go_match:
        parsed["go_number"] = go_match.group(0).strip()

    # Extract Date
    date_pattern = r'(?:நாள்|Dated?)\s*[:.]?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})'
    date_match = re.search(date_pattern, cleaned_text, re.IGNORECASE)
    if date_match:
        parsed["date"] = date_match.group(1).strip()

    # Extract References (படிக்கப்பட்டது / Read)
    read_pattern = r'(?:படிக்கப்பட்டது|Read\s*[:-]):(.*?)(?=(?:ஆணை|ORDER|\n\s*\d+\.\s*பார்வை))'
    read_match = re.search(read_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    if read_match:
        raw_refs = read_match.group(1).split("\n")
        parsed["references"] = [r.strip() for r in raw_refs if len(r.strip()) > 3]

    # Split Preamble vs. Operative Clause (ஆணை / ORDER)
    order_split = re.split(r'\n(?:\d+\.?\s*)?(?:ஆணை|ORDER)\s*[:\-]', cleaned_text, flags=re.IGNORECASE)
    if len(order_split) > 1:
        parsed["preamble"] = order_split[0].strip()
        parsed["order_clause"] = order_split[1].strip()
    else:
        parsed["order_clause"] = cleaned_text.strip()

    return parsed

# -------------------------------------------------------------
# Stage 4: Bureaucratic Jargon Identification & Glossing
# -------------------------------------------------------------
def tag_legal_jargon(order_clause: str) -> tuple[str, list]:
    detected = []
    annotated = order_clause

    for formal, plain in TAMIL_ADMIN_LEXICON.items():
        if formal in annotated:
            detected.append({"bureaucratic_term": formal, "plain_language": plain})
            # Annotate inline using Markdown delimiter for accessibility
            annotated = re.sub(re.escape(formal), f"**{formal}** (*{plain}*)", annotated)

    return annotated, detected

# -------------------------------------------------------------
# Main Execution Coordinator
# -------------------------------------------------------------
def run_pipeline():
    pdf_files = []
    for root, _, files in os.walk(RAW_PDF_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    if not pdf_files:
        print(f"[!] No PDFs located inside: {RAW_PDF_DIR}")
        return

    print(f"[*] Starting Pipeline (Stages 1-4) over {len(pdf_files)} documents...")

    for idx, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"\n--- [{idx}/{len(pdf_files)}] Processing: {filename} ---")
        
        # Stage 1 & 2
        cleaned_text = extract_text_from_pdf(pdf_path)
        txt_out_path = os.path.join(CLEAN_TEXT_DIR, f"{os.path.splitext(filename)[0]}.txt")
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f" [✓] Stage 1 & 2: Extracted & Cleaned ({len(cleaned_text)} chars)")

        # Stage 3
        parsed_doc = parse_go_structure(cleaned_text)
        print(f" [✓] Stage 3: Parsed GO identifier: {parsed_doc['go_number']}")

        # Stage 4
        annotated_order, detected_terms = tag_legal_jargon(parsed_doc["order_clause"])
        parsed_doc["annotated_order_clause"] = annotated_order
        parsed_doc["detected_jargon"] = detected_terms
        parsed_doc["source_file"] = filename
        print(f" [✓] Stage 4: Tagged {len(detected_terms)} bureaucratic terms")

        # Save Final Structured Artifact
        json_out_path = os.path.join(PROCESSED_JSON_DIR, f"{os.path.splitext(filename)[0]}_processed.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(parsed_doc, f, ensure_ascii=False, indent=2)
        print(f" [✓] Saved pipeline output: {os.path.basename(json_out_path)}")

    print(f"\n[✓] Processing finished! Outputs stored in:\n    - Texts: {CLEAN_TEXT_DIR}\n    - Structured Records: {PROCESSED_JSON_DIR}")

if __name__ == "__main__":
    run_pipeline()