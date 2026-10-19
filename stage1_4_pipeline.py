import os
import re
import json
import unicodedata
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

try:
    from tamil.txt2unicode import auto2unicode
except ImportError:
    auto2unicode = lambda x: x

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PDF_DIR = os.path.join(BASE_DIR, "dataset_raw_gos")
CLEAN_TEXT_DIR = os.path.join(BASE_DIR, "dataset_cleaned_text")
PROCESSED_JSON_DIR = os.path.join(BASE_DIR, "dataset_processed_records")

os.makedirs(CLEAN_TEXT_DIR, exist_ok=True)
os.makedirs(PROCESSED_JSON_DIR, exist_ok=True)

COMMON_TAMIL_WORDS = {
    "அரசு", "அரசாணை", "துறை", "தமிழ்நாடு", "திட்டம்", "பணி", "பணிகள்", "ஊரக", "வளர்ச்சி",
    "ஊராட்சி", "மாவட்ட", "ஆணை", "எண்", "நாள்", "ரூபாய்", "ரூ", "செயல்பாடுகள்",
    "மற்றும்", "அல்லது", "சார்பாக", "வழங்க", "தெரிவிக்கப்படுகிறது", "பொருள்", "பார்வை",
    "சுற்றறிக்கை", "இயக்குநர்", "ஆணையர்", "தலைவர்", "சென்னை", "விவரம்", "அனுமதி",
    "தொகை", "நிதி", "ஒதுக்கீடு", "அலுவலக", "அலுவலர்", "விதிகள்", "உத்தரவு"
}

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
    "விடுவித்து": "வழங்கி",
    "ஒப்பளிப்பு": "அனுமதி",
    "சுற்றறிக்கை": "அரசு தகவல் அறிக்கை",
    "வழிகாட்டு நெறிமுறைகள்": "செயல்முறை விதிகள்",
    "பணிமேற்பார்வையாளர்": "வேலை மேற்பார்வையாளர்",
    "அறிவுறுத்தப்படுகிறது": "உத்தரவிடப்படுகிறது",
    "மேற்கொள்ளப்பட வேண்டும்": "செய்யப்பட வேண்டும்"
}

# -------------------------------------------------------------
# Stage 1 & 2: Ingestion, OCR Fallback, Normalisation
# -------------------------------------------------------------
def is_meaningful_tamil(text: str) -> bool:
    if not text.strip():
        return False
    words = re.findall(r'[\u0B80-\u0BFF]+', text)
    if len(words) < 2:
        return False
    match_count = sum(1 for w in words if w in COMMON_TAMIL_WORDS)
    tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    ratio = tamil_chars / max(len(text.strip()), 1)
    return (ratio > 0.15 and match_count >= 1) or match_count >= 2

def clean_and_normalize(text: str) -> str:
    """Normalize Unicode to NFC form and strip scanner/layout noise."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
    text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*[-—–]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r'_{2,}', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract embedded digital text, fallback to OCR if empty, scanned, or corrupt font."""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_text = page.get_text("text")
        
        # Check if authentic Unicode Tamil is present
        if is_meaningful_tamil(page_text):
            full_text += page_text + "\n"
            continue

        # Try converting legacy font encoding
        try:
            converted = auto2unicode(page_text)
            if is_meaningful_tamil(converted):
                full_text += converted + "\n"
                continue
        except Exception:
            pass

        # High-res OCR fallback with Tamil + English
        pix = page.get_pixmap(dpi=250)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
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
        "subject": "",
        "references": [],
        "preamble": "",
        "order_clause": ""
    }

    # Extract GO / Circular Number (Tamil & English signatures)
    go_pattern = (
        r'(?:(?:அரசாணை|சுற்றறிக்கை|கடிதம்|செயல்முறைகள்)\s*(?:\([^\)]+\))?\s*(?:நிலை|பல்வகை)?\s*எண்\.?|'
        r'ந\.?\s*க\.?\s*எண்\.?|Rc\.?\s*No\.?|D\.?O\.?\s*No\.?|'
        r'G\.O\.?\s*(?:\([^\)]+\))?\s*(?:Ms|Rt)?\.?\s*No\.?)\s*[:.]?\s*([^\n,]+)'
    )
    go_match = re.search(go_pattern, cleaned_text, re.IGNORECASE)
    if go_match:
        raw_id = go_match.group(0).strip()
        raw_id = re.split(r'[\s,]+(?:இயக்குநர்|Director|ஆணையர்|Commissioner)', raw_id)[0].strip()
        parsed["go_number"] = raw_id

    # Extract Date
    date_pattern = r'(?:நாள்|தேதி|Dated?)\s*[:.]?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})'
    date_match = re.search(date_pattern, cleaned_text, re.IGNORECASE)
    if date_match:
        parsed["date"] = date_match.group(1).strip()

    # Extract Subject (பொருள்)
    subject_match = re.search(
        r'(?:பொருள்|Subject|Sub)\s*[:\-–]\s*(.{10,400}?)(?:\n\s*(?:பார்வை|Ref|Reference)|தொடர்பாக\.?|குறித்து\.?|\n\s*\n)',
        cleaned_text, re.DOTALL | re.IGNORECASE
    )
    if subject_match:
        sub_raw = subject_match.group(1).replace("\n", " ").strip().rstrip(" -–.")
        parsed["subject"] = sub_raw + " - தொடர்பாக."

    # Extract References (படிக்கப்பட்டது / பார்வை)
    read_pattern = r'(?:பார்வை|படிக்கப்பட்டது|Read|Reference|Ref)\s*[:\-–]\s*(.*?)(?=(?:\n\s*(?:சுற்றறிக்கை|ஆணை|ORDER|\n\s*--+|\n\n[^\d\s])))'
    read_match = re.search(read_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    if read_match:
        raw_refs = read_match.group(1).split("\n")
        parsed["references"] = [r.strip() for r in raw_refs if len(r.strip()) > 3]

    # Split Preamble vs. Operative Clause (ஆணை / ORDER / Circular Body)
    order_split = re.split(r'\n(?:\d+\.?\s*)?(?:ஆணை|ORDER)\s*[:\-]', cleaned_text, flags=re.IGNORECASE)
    if len(order_split) > 1:
        parsed["preamble"] = order_split[0].strip()
        parsed["order_clause"] = order_split[1].strip()
    else:
        body = cleaned_text
        if parsed["subject"]:
            sub_key = parsed["subject"][:25]
            if sub_key in cleaned_text:
                body = cleaned_text.split(sub_key, 1)[-1].strip()
        parsed["preamble"] = cleaned_text[:len(cleaned_text)-len(body)].strip()
        parsed["order_clause"] = body.strip()

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
        base_name = os.path.splitext(filename)[0]
        print(f"\n[{idx}/{len(pdf_files)}] Ingesting & normalizing: {filename}")

        cleaned_text = extract_text_from_pdf(pdf_path)
        clean_txt_path = os.path.join(CLEAN_TEXT_DIR, f"{base_name}.txt")
        with open(clean_txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        parsed_data = parse_go_structure(cleaned_text)
        annotated_clause, detected_jargon = tag_legal_jargon(parsed_data["order_clause"])

        record = {
            "go_number": parsed_data["go_number"],
            "date": parsed_data["date"],
            "subject": parsed_data["subject"],
            "references": parsed_data["references"],
            "preamble": parsed_data["preamble"],
            "order_clause": parsed_data["order_clause"],
            "annotated_order_clause": annotated_clause,
            "detected_jargon": detected_jargon,
            "source_file": filename
        }

        json_out_path = os.path.join(PROCESSED_JSON_DIR, f"{base_name}_processed.json")
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"  [✓] Identified ID: {parsed_data['go_number']}")
        print(f"  [✓] Date: {parsed_data['date']}")
        print(f"  [✓] Jargon instances tagged: {len(detected_jargon)}")

    print(f"\n[✓] Finished Stages 1-4. Processed records saved in: {PROCESSED_JSON_DIR}")

if __name__ == "__main__":
    run_pipeline()