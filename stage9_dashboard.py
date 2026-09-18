import os
import io
import re
import json
import unicodedata
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Legacy font auto-converter
try:
    from tamil.txt2unicode import auto2unicode
except ImportError:
    auto2unicode = lambda x: x

st.set_page_config(
    page_title="Tamil Nadu GO Detailed Summarizer",
    page_icon="🏛️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset_bilingual_records")

# -------------------------------------------------------------
# Stage 4 Administrative Lexicon
# -------------------------------------------------------------
ADMIN_LEXICON = {
    "பரிசீலித்து": "ஆராய்ந்து (After careful examination)",
    "ஆணை பிறப்பிக்கப்படுகிறது": "உத்தரவிடப்படுகிறது (Order is issued)",
    "ஆணை வெளியிடப்படுகிறது": "அறிவிக்கப்படுகிறது (Published/Notified)",
    "பற்று வைக்கப்படும்": "செலவிடப்படும் (Debited/Expended)",
    "நிதி ஒதுக்கீடு": "பண ஒதுக்கீடு (Budget allocation)",
    "கருத்துரு": "பரிந்துரை / கோரிக்கை (Proposal)",
    "வைப்பு நிதி": "சேமிப்புத் தொகை (Deposit fund)",
    "பயனாளிகள்": "உதவி பெறுவோர் (Beneficiaries)",
    "நடைமுறைப்படுத்த": "செயல்படுத்த (Implement)",
    "ஒப்புதல் அளிக்கப்படுகிறது": "அனுமதி தரப்படுகிறது (Sanctioned/Approved)",
    "நிர்வாக அனுமதி": "அதிகாரப்பூர்வ அனுமதி (Administrative sanction)",
    "விடுவித்து": "வழங்கி (Released/Disbursed)"
}

# -------------------------------------------------------------
# Cached Inference Engine
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI Models (IndicBART & NLLB)...")
def load_models():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # IndicBART (Tamil Simplification & Summarization)
    bart_name = "ai4bharat/MultiIndicSentenceSummarizationSS"
    bart_tok = AutoTokenizer.from_pretrained(bart_name, use_fast=False)
    bart_mod = AutoModelForSeq2SeqLM.from_pretrained(bart_name).to(device)
    bart_mod.eval()

    # NLLB (Tamil -> English Translation)
    nllb_name = "facebook/nllb-200-distilled-600M"
    nllb_tok = AutoTokenizer.from_pretrained(nllb_name, src_lang="tam_Taml")
    nllb_mod = AutoModelForSeq2SeqLM.from_pretrained(nllb_name).to(device)
    nllb_mod.eval()

    return {
        "device": device,
        "bart_tok": bart_tok,
        "bart_mod": bart_mod,
        "nllb_tok": nllb_tok,
        "nllb_mod": nllb_mod
    }

# -------------------------------------------------------------
# Document Processing & Legacy Font Detection
# -------------------------------------------------------------
def clean_tamil_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'/ ?/ ?True Copy ?/ ?/', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*[-—–]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def is_authentic_tamil(text: str) -> bool:
    """Check if the text contains authentic Tamil Unicode glyphs."""
    if not text.strip():
        return False
    tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    return (tamil_chars / max(len(text.strip()), 1)) > 0.15

def extract_pdf_content(pdf_bytes: bytes) -> str:
    """Extracts text layer, handles legacy fonts, or falls back to OCR."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text()

        # Step 1: Already authentic Unicode Tamil?
        if is_authentic_tamil(raw_text):
            pages_text.append(raw_text)
            continue

        # Step 2: Try converting legacy font encoding (Bamini/TAB/TAM)
        try:
            converted = auto2unicode(raw_text)
            if is_authentic_tamil(converted):
                pages_text.append(converted)
                continue
        except Exception:
            pass

        # Step 3: Scanned PDF or corrupt font layer -> fallback to 300 DPI OCR
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            ocr_text = pytesseract.image_to_string(img, lang="tam")
            pages_text.append(ocr_text)
        except Exception:
            pages_text.append(pytesseract.image_to_string(img))

    doc.close()
    return clean_tamil_text("\n".join(pages_text))

def parse_metadata_and_sections(text: str) -> dict:
    go_match = re.search(r'(?:அரசாணை\s*(?:\([^\)]+\))?\s*எண்\.?|G\.O\.?\s*(?:\([^\)]+\))?\s*(?:Ms|Rt)?\.?\s*No\.?)\s*[:.]?\s*([^\n,]+)', text, re.IGNORECASE)
    date_match = re.search(r'(?:நாள்|Dated?)\s*[:.]?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', text, re.IGNORECASE)
    dept_match = re.search(r'([^\n]+(?:Department|துறை)[^\n]*)', text, re.IGNORECASE)

    # Split Preamble vs. Operative Clause
    order_split = re.split(r'\n(?:\d+\.?\s*)?(?:ஆணை|ORDER)\s*[:\-]', text, flags=re.IGNORECASE)
    preamble = order_split[0].strip() if len(order_split) > 1 else ""
    order_clause = order_split[1].strip() if len(order_split) > 1 else text.strip()

    # Extract References (படிக்கப்பட்டது / Read)
    ref_match = re.search(r'(?:படிக்கப்பட்டது|Read\s*[:-]):(.*?)(?=(?:ஆணை|ORDER|\n\s*\d+\.\s*பார்வை))', text, re.DOTALL | re.IGNORECASE)
    refs = [r.strip() for r in ref_match.group(1).split("\n") if len(r.strip()) > 5] if ref_match else []

    # Financial / Numeric details
    amounts = re.findall(r'(?:ரூ\.?|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?(?:\s*(?:கோடி|இலட்சம்|crore|lakh))?', text, re.IGNORECASE)

    return {
        "go_number": go_match.group(0).strip() if go_match else "G.O. Identifier Not Specified",
        "date": date_match.group(1).strip() if date_match else "Date Not Explicit",
        "department": dept_match.group(1).strip() if dept_match else "Government of Tamil Nadu",
        "references": refs,
        "amounts": list(set(amounts)),
        "preamble": preamble,
        "order_clause": order_clause
    }

# -------------------------------------------------------------
# Multi-Chunk Summarization & Degeneration-Safe Translation
# -------------------------------------------------------------
def summarize_chunk(chunk_text: str, models: dict) -> str:
    device = models["device"]
    tok = models["bart_tok"]
    mod = models["bart_mod"]

    fmt_input = f"{chunk_text} <2ta>"
    inputs = tok(fmt_input, return_tensors="pt", truncation=True, max_length=512, add_special_tokens=False).to(device)

    pad_id = tok._convert_token_to_id_with_added_voc("<pad>")
    bos_id = tok._convert_token_to_id_with_added_voc("<s>")
    eos_id = tok._convert_token_to_id_with_added_voc("</s>")
    decoder_start_id = tok._convert_token_to_id_with_added_voc("<2ta>")

    with torch.no_grad():
        summary_ids = mod.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=200,
            min_length=25,
            length_penalty=1.0,
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            decoder_start_token_id=decoder_start_id
        )
    return tok.decode(summary_ids[0], skip_special_tokens=True).replace("<2ta>", "").strip()

def generate_detailed_tamil_summary(order_text: str, models: dict) -> list[str]:
    raw_paragraphs = [p.strip() for p in order_text.split("\n\n") if len(p.strip()) > 35]
    if not raw_paragraphs:
        raw_paragraphs = [order_text[i:i+600] for i in range(0, min(len(order_text), 2400), 600)]

    summarized_points = []
    for chunk in raw_paragraphs[:4]:
        if is_authentic_tamil(chunk):
            pt = summarize_chunk(chunk, models)
            if len(pt) > 15:
                summarized_points.append(pt)
        else:
            summarized_points.append(chunk[:250])

    return summarized_points

def translate_chunks_to_english(tamil_points: list[str], models: dict) -> list[str]:
    device = models["device"]
    tok = models["nllb_tok"]
    mod = models["nllb_mod"]
    tgt_id = tok.convert_tokens_to_ids("eng_Latn")

    en_points = []
    for point in tamil_points:
        inputs = tok(point, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            ids = mod.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_length=180,
                num_beams=4,
                no_repeat_ngram_size=3,       # Stops repetition loops
                repetition_penalty=1.2,       # Penalizes degenerate repetitive phrases
                early_stopping=True
            )
        translated = tok.batch_decode(ids, skip_special_tokens=True)[0].strip()
        en_points.append(translated)
    return en_points

# -------------------------------------------------------------
# User Interface (Streamlit)
# -------------------------------------------------------------
st.title("🏛️ தமிழ்நாடு அரசாணை விரிவான சுருக்கம்")
st.markdown("### Tamil Nadu Government Order Detailed Summarizer")
st.caption("Legacy Font Converter • Operative Clause Extractor • IndicBART Plain Tamil • Degeneration-Safe NLLB")

tab1, tab2 = st.tabs(["📄 Upload & Summarize Document", "🗄️ Browse Database"])

with tab1:
    uploaded_file = st.file_uploader("Upload Tamil Nadu Government Order (PDF)", type=["pdf"])

    if uploaded_file and st.button("Generate Detailed Summary", type="primary"):
        models = load_models()

        with st.status("Analyzing document and generating summaries...", expanded=True) as status:
            st.write("1️⃣ Extracting text, detecting encoding, and resolving OCR...")
            clean_text = extract_pdf_content(uploaded_file.getvalue())

            st.write("2️⃣ Parsing administrative metadata and operative orders...")
            parsed = parse_metadata_and_sections(clean_text)

            st.write("3️⃣ Scanning administrative legal jargon...")
            detected_jargon = {k: v for k, v in ADMIN_LEXICON.items() if k in parsed["order_clause"]}

            st.write("4️⃣ Generating multi-clause plain Tamil summary...")
            tamil_summary_points = generate_detailed_tamil_summary(parsed["order_clause"], models)

            st.write("5️⃣ Generating English translation...")
            en_summary_points = translate_chunks_to_english(tamil_summary_points, models)

            status.update(label="Summarization complete!", state="complete", expanded=False)

        # Overview Metadata Card
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("அரசாணை எண் (GO Number)", parsed["go_number"])
        m2.metric("நாள் (Date)", parsed["date"])
        m3.metric("துறை (Department)", parsed["department"][:45])

        if parsed["amounts"]:
            st.info(f"💰 **நிதி ஒதுக்கீடு விவரங்கள் (Identified Allocations):** {', '.join(parsed['amounts'])}")

        st.divider()

        # Side-by-Side Detailed Summary View
        col_ta, col_en = st.columns(2)

        with col_ta:
            st.markdown("### 🗣️ விரிவான எளிய தமிழ் சுருக்கம் (Detailed Tamil Summary)")
            for idx, pt in enumerate(tamil_summary_points, 1):
                st.markdown(f"**பிரிவு {idx}:** {pt}")

            if detected_jargon:
                with st.expander(f"📌 நிர்வாக சட்ட சொற்கள் விளக்கம் (Glossary - {len(detected_jargon)} terms)"):
                    for formal, explanation in detected_jargon.items():
                        st.write(f"- **{formal}** ➜ *{explanation}*")

        with col_en:
            st.markdown("### 🇬🇧 Detailed English Summary")
            for idx, pt in enumerate(en_summary_points, 1):
                st.markdown(f"**Section {idx}:** {pt}")

            if parsed["references"]:
                with st.expander(f"📑 Official References ({len(parsed['references'])} items)"):
                    for r in parsed["references"]:
                        st.write(f"- {r}")

        with st.expander("📄 View Extracted Text Layer"):
            st.text_area("Cleaned Content", clean_text, height=300)

with tab2:
    st.subheader("Archived Processed Orders")
    if os.path.exists(PROCESSED_DIR):
        files = [f for f in sorted(os.listdir(PROCESSED_DIR)) if f.endswith(".json")]
        if files:
            picked = st.selectbox("Select a processed record to inspect:", files)
            with open(os.path.join(PROCESSED_DIR, picked), "r", encoding="utf-8") as f:
                data = json.load(f)

            st.markdown(f"#### 🏛️ {data.get('go_number', 'Unknown GO')} ({data.get('date', '')})")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### எளிய தமிழ் (Plain Tamil)")
                st.info(data.get("simplified_plain_tamil", "N/A"))
            with c2:
                st.markdown("##### English Summary")
                st.success(data.get("english_summary", "N/A"))
        else:
            st.warning("No archived records found.")
    else:
        st.warning("Processed records folder not found.")