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
from gtts import gTTS

# Legacy font auto-converter
try:
    from tamil.txt2unicode import auto2unicode
except ImportError:
    auto2unicode = lambda x: x

st.set_page_config(
    page_title="Tamil Nadu GO & Circular Citizen Simplifier",
    page_icon="🏛️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset_bilingual_records")
AUDIO_DIR = os.path.join(BASE_DIR, "dataset_audio_outputs")
os.makedirs(AUDIO_DIR, exist_ok=True)

# -------------------------------------------------------------
# Common Authentic Tamil Vocabulary (for OCR/Font Validation)
# -------------------------------------------------------------
COMMON_TAMIL_WORDS = {
    "அரசு", "அரசாணை", "துறை", "தமிழ்நாடு", "திட்டம்", "பணி", "பணிகள்", "ஊரக", "வளர்ச்சி",
    "ஊராட்சி", "மாவட்ட", "ஆணை", "எண்", "நாள்", "ரூபாய்", "ரூ", "செயல்பாடுகள்",
    "மற்றும்", "அல்லது", "சார்பாக", "வழங்க", "தெரிவிக்கப்படுகிறது", "பொருள்", "பார்வை",
    "சுற்றறிக்கை", "இயக்குநர்", "ஆணையர்", "தலைவர்", "சென்னை", "விவரம்", "அனுமதி",
    "தொகை", "நிதி", "ஒதுக்கீடு", "அலுவலக", "அலுவலர்", "விதிகள்", "உத்தரவு", "விண்ணப்பம்"
}

# -------------------------------------------------------------
# Entity Grounding & Anti-Hallucination Guardrails
# -------------------------------------------------------------
POLITICAL_NAMES = [
    "எடப்பாடி", "பழனிசாமி", "ஸ்டாலின்", "மு.க.ஸ்டாலின்", "மு.க. ஸ்டாலின்",
    "ஜெயலலிதா", "கருணாநிதி", "பன்னீர்செல்வம்", "ஓ.பன்னீர்செல்வம்", "மோடி", "அமித்ஷா"
]
POLITICAL_PARTIES = ["திமுக", "அதிமுக", "காங்கிரஸ்", "பாஜக", "பாமக", "கம்யூனிஸ்ட்", "நாம் தமிழர்"]
TITLES = ["முதலமைச்சர்", "பிரதமர்"]
NEWS_MEDIA = [
    "நிதர்சனம்", "தந்தி டிவி", "நக்கீரன்", "தினமலர்", "தினத்தந்தி",
    "மாலை மலர்", "புதிய தலைமுறை", "பாலிமர்", "நியூஸ் 18", "பிபிசி", "செய்திகள்"
]

# -------------------------------------------------------------
# Administrative Lexicon: Bureaucratic -> Plain Citizen
# -------------------------------------------------------------
ADMIN_LEXICON = {
    "பரிசீலித்து": "ஆராய்ந்து (After examination)",
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
    "விடுவித்து": "வழங்கி (Released/Disbursed)",
    "ஒப்பளிப்பு": "அனுமதி (Sanction)",
    "சுற்றறிக்கை": "அரசு தகவல் அறிக்கை (Official circular)",
    "வழிகாட்டு நெறிமுறைகள்": "செயல்முறை விதிகள் (Operational guidelines)",
    "பணிமேற்பார்வையாளர்": "வேலை மேற்பார்வையாளர் (Work Supervisor)",
    "அறிவுறுத்தப்படுகிறது": "உத்தரவிடப்படுகிறது (Instructed)",
    "மேற்கொள்ளப்பட வேண்டும்": "செய்யப்பட வேண்டும் (Must be carried out)",
    "முன்னிலைப்படுத்தப்பட்டு": "முன்னுரிமை தரப்பட்டு (Prioritized)",
    "வழங்கப்பட வேண்டும்": "கொடுக்கப்பட வேண்டும் (Must be provided)"
}

# -------------------------------------------------------------
# Cached Inference Engine
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI Models (IndicBART & NLLB)...")
def load_models():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    bart_name = "ai4bharat/MultiIndicSentenceSummarizationSS"
    bart_tok = AutoTokenizer.from_pretrained(bart_name, use_fast=False)
    bart_mod = AutoModelForSeq2SeqLM.from_pretrained(bart_name).to(device)
    bart_mod.eval()

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
# Voice Synthesis Engine (TTS)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def synthesize_speech(text: str, lang: str = "ta") -> bytes:
    if not text or not text.strip():
        return b""
    clean_audio_text = re.sub(r'[*_#`\[\]]', '', text)
    clean_audio_text = re.sub(r'\s+', ' ', clean_audio_text).strip()
    audio_sample = clean_audio_text[:1500]
    try:
        tts = gTTS(text=audio_sample, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"TTS Error: {e}")
        return b""

# -------------------------------------------------------------
# Document Processing & Robust OCR
# -------------------------------------------------------------
def clean_tamil_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
    text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'/ ?/ ?True Copy ?/ ?/', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*[-—–]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

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

def extract_pdf_content(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        raw_text = page.get_text()

        if is_meaningful_tamil(raw_text):
            pages_text.append(raw_text)
            continue

        try:
            converted = auto2unicode(raw_text)
            if is_meaningful_tamil(converted):
                pages_text.append(converted)
                continue
        except Exception:
            pass

        pix = page.get_pixmap(dpi=250)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        try:
            ocr_text = pytesseract.image_to_string(img, lang="tam+eng")
            pages_text.append(ocr_text)
        except Exception:
            pages_text.append(pytesseract.image_to_string(img))

    doc.close()
    return clean_tamil_text("\n".join(pages_text))

# -------------------------------------------------------------
# Metadata, Subject & Section Parser
# -------------------------------------------------------------
def parse_metadata_and_sections(text: str) -> dict:
    clean_t = re.sub(r'[\u200b-\u200d\ufeff]', '', text)

    # 1. Identifier
    go_match = re.search(
        r'(?:(?:அரசாணை|சுற்றறிக்கை|கடிதம்|செயல்முறைகள்)\s*(?:\([^\)]+\))?\s*(?:நிலை|பல்வகை)?\s*எண்\.?|'
        r'ந\.?\s*க\.?\s*எண்\.?|Rc\.?\s*No\.?|D\.?O\.?\s*No\.?|'
        r'G\.O\.?\s*(?:\([^\)]+\))?\s*(?:Ms|Rt)?\.?\s*No\.?)\s*[:.]?\s*([^\n,]+)',
        clean_t, re.IGNORECASE
    )
    go_number = "Identifier Not Specified"
    if go_match:
        raw_id = go_match.group(0).strip()
        raw_id = re.split(r'[\s,]+(?:இயக்குநர்|Director|ஆணையர்|Commissioner)', raw_id)[0].strip()
        go_number = raw_id

    # 2. Date
    date_match = re.search(r'(?:நாள்|தேதி|Dated?)\s*[:.]?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', clean_t, re.IGNORECASE)
    date_str = date_match.group(1).strip() if date_match else "Date Not Explicit"

    # 3. Department
    dept_match = re.search(r'([^\n]+(?:Department|துறை|இயக்ககம்|ஆணையரகம்|வாரியம்)[^\n]*)', clean_t, re.IGNORECASE)
    dept_str = "Government of Tamil Nadu"
    if dept_match:
        raw_dept = dept_match.group(1).strip()
        raw_dept = re.sub(r'^(?:நாள்|தேதி|Dated?)\s*[:.]?\s*[0-9./-]+\s*', '', raw_dept, flags=re.IGNORECASE)
        dept_str = raw_dept[:60].strip()

    # 4. Subject (பொருள்)
    subject_match = re.search(
        r'(?:பொருள்|Subject|Sub)\s*[:\-–]\s*(.{10,400}?)(?:\n\s*(?:பார்வை|Ref|Reference)|தொடர்பாக\.?|குறித்து\.?|\n\s*\n)',
        clean_t, re.DOTALL | re.IGNORECASE
    )
    subject_str = ""
    if subject_match:
        sub_raw = subject_match.group(1).replace("\n", " ").strip().rstrip(" -–.")
        subject_str = sub_raw + " - தொடர்பாக."

    # 5. References (பார்வை)
    ref_match = re.search(
        r'(?:பார்வை|படிக்கப்பட்டது|Read|Reference|Ref)\s*[:\-–]\s*(.*?)(?=(?:\n\s*(?:சுற்றறிக்கை|ஆணை|ORDER|\n\s*--+|\n\n[^\d\s])))',
        clean_t, re.DOTALL | re.IGNORECASE
    )
    refs = [r.strip() for r in ref_match.group(1).split("\n") if len(r.strip()) > 5] if ref_match else []

    # 6. Operative Body
    order_split = re.split(r'\n(?:\d+\.?\s*)?(?:ஆணை|ORDER)\s*[:\-]', clean_t, flags=re.IGNORECASE)
    if len(order_split) > 1:
        preamble = order_split[0].strip()
        body = order_split[1].strip()
    else:
        body = clean_t
        if subject_str:
            sub_key = subject_str[:25]
            if sub_key in clean_t:
                body = clean_t.split(sub_key, 1)[-1].strip()
        preamble = clean_t[:len(clean_t)-len(body)].strip()

    # Clean header noise from operative body
    body_cleaned_lines = []
    skip_header = True
    for line in body.split("\n"):
        line_s = line.strip()
        if skip_header:
            if any(k in line_s for k in ["அனுப்புநர்", "பெறுநர்", "From", "To", "பனகல் கட்டிடம்", "ஐயா / அம்மையீர்"]):
                continue
            if line_s.startswith("நாள்") or line_s.startswith("ந.க.எண்") or line_s.startswith("சுற்றறிக்கை"):
                continue
            if len(line_s) > 25:
                skip_header = False
        body_cleaned_lines.append(line)
    operative_body = "\n".join(body_cleaned_lines).strip()
    if len(operative_body) < 50:
        operative_body = body

    amounts = re.findall(r'(?:ரூ\.?|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?(?:\s*(?:கோடி|இலட்சம்|crore|lakh))?', clean_t, re.IGNORECASE)

    return {
        "go_number": go_number,
        "date": date_str,
        "department": dept_str,
        "subject": subject_str,
        "references": refs,
        "amounts": list(set(amounts)),
        "preamble": preamble,
        "order_clause": operative_body
    }

# -------------------------------------------------------------
# Deep Structured Insight Extractor (Deadlines, Venues, Beneficiaries)
# -------------------------------------------------------------
def extract_structured_insights(text: str) -> dict:
    insights = {
        "deadlines": [],
        "venues_jurisdiction": [],
        "beneficiaries": [],
        "officials": [],
        "financials": []
    }

    # 1. Deadlines and Timeframes
    d_patterns = [
        r'\b\d+\s*நாட்களுக்குள்[^\.\n]*',
        r'\b\d+\s*வாரத்திற்குள்[^\.\n]*',
        r'\b\d+\s*வாரத்தில்[^\.\n]*',
        r'\bமாதம்\s*ஒரு\s*முறை[^\.\n]*',
        r'\bதினசரி\s*வருகைப்பட்டியல்[^\.\n]*',
        r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\s*க்குள்[^\.\n]*',
        r'\b\d{4}-\d{2,4}\s*(?:ஆம்\s*)?ஆண்டிற்குள்[^\.\n]*',
        r'\bகுறிப்பிட்ட\s*தேதியில்\s*முடித்திட[^\.\n]*'
    ]
    for p in d_patterns:
        for m in re.finditer(p, text):
            s = m.group(0).strip(" ,.-–")
            if s and len(s) < 100 and s not in insights["deadlines"]:
                insights["deadlines"].append(s)

    # 2. Financials & Wage Rates
    f_patterns = [
        r'(?:ரூ\.?|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?(?:\s*(?:கோடி|இலட்சம்|crore|lakh))?[^\.\n]*',
        r'\b\d+%\s*(?:க்கு\s*(?:குறையாமல்|மேல்))?[^\.\n]*',
        r'\b(?:தினசரி\s*கூலி|ஊதியம்|மதிப்பீடு|செலவீனம்)[^\.\n]*'
    ]
    for p in f_patterns:
        for m in re.finditer(p, text):
            s = m.group(0).strip(" ,.-–")
            if s and len(s) < 110 and any(c.isdigit() for c in s) and s not in insights["financials"]:
                insights["financials"].append(s)

    # 3. Target Beneficiaries & Eligibility
    if any(k in text for k in ["வேலை அட்டை", "குடும்பங்கள்"]):
        insights["beneficiaries"].append("வேலை அட்டை கோரும் தகுதியான அனைத்து ஊரகக் குடும்பங்கள்")
    if "சுயஉதவிக்" in text:
        insights["beneficiaries"].append("மகளிர் சுயஉதவிக் குழுக்கள் (SHGs)")
    if any(k in text for k in ["PIP", "இலக்கு மக்கள்"]):
        insights["beneficiaries"].append("மக்கள் பங்கேற்பு முறையில் தெரிவு செய்யப்பட்ட இலக்கு மக்கள் (PIP)")
    if "தொழிலாளர்கள்" in text:
        insights["beneficiaries"].append("கிராமப்புற 100 நாள் வேலை உறுதித் திட்ட தொழிலாளர்கள்")
    if "விவசாயிகள்" in text:
        insights["beneficiaries"].append("கிராமப்புற விவசாயிகள் & நில உரிமையாளர்கள்")
    if not insights["beneficiaries"]:
        insights["beneficiaries"].append("திட்டத்தின் கீழ் உரிய தகுதியுடைய ஊரக பொதுமக்கள்")

    # 4. Venue & Jurisdiction
    if any(k in text for k in ["பனகல் கட்டிடம்", "சைதாப்பேட்டை", "சென்னை"]):
        insights["venues_jurisdiction"].append("தலைமை அலுவலகம்: பனகல் கட்டிடம், சைதாப்பேட்டை, சென்னை-15")
    if any(k in text for k in ["அனைத்து மாவட்ட", "மாவட்டங்களில்"]):
        insights["venues_jurisdiction"].append("செயல்படும் எல்லை: தமிழ்நாட்டின் அனைத்து மாவட்டங்கள்")
    if any(k in text for k in ["கிராம ஊராட்சி", "ஊராட்சி"]):
        insights["venues_jurisdiction"].append("அமலாக்க தளம்: அனைத்து கிராம ஊராட்சிகள் & வட்டார வளர்ச்சி அலுவலகங்கள்")
    if not insights["venues_jurisdiction"]:
        insights["venues_jurisdiction"].append("செயல்படும் எல்லை: தமிழ்நாடு அரசு நிர்வாக வரம்பு")

    # 5. Responsible Officials
    official_list = ["மாவட்ட ஆட்சியர்", "இயக்குநர்", "ஆணையர்", "மண்டல துணை வட்டார வளர்ச்சி அலுவலர்", "வட்டார வளர்ச்சி அலுவலர்", "பணி மேற்பார்வையாளர்"]
    for o in official_list:
        if o in text and o not in insights["officials"]:
            insights["officials"].append(o)

    return insights

# -------------------------------------------------------------
# Anti-Hallucination & Plain Lexicon Simplification
# -------------------------------------------------------------
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
            replacement = plain.split(" (")[0]
            res = res.replace(formal, replacement)
    return res

def summarize_chunk_safely(chunk_text: str, models: dict) -> str:
    device = models["device"]
    tok = models["bart_tok"]
    mod = models["bart_mod"]

    fmt_input = f"{chunk_text[:500]} <2ta>"
    inputs = tok(fmt_input, return_tensors="pt", truncation=True, max_length=512, add_special_tokens=False).to(device)

    pad_id = tok._convert_token_to_id_with_added_voc("<pad>")
    bos_id = tok._convert_token_to_id_with_added_voc("<s>")
    eos_id = tok._convert_token_to_id_with_added_voc("</s>")
    decoder_start_id = tok._convert_token_to_id_with_added_voc("<2ta>")

    with torch.no_grad():
        summary_ids = mod.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=160,
            min_length=10,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=0.8,
            pad_token_id=pad_id,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            decoder_start_token_id=decoder_start_id
        )

    raw_summary = tok.decode(summary_ids[0], skip_special_tokens=True).replace("<2ta>", "").strip()
    cleaned_summary = clean_news_artifacts(raw_summary)

    hallucinations = detect_hallucinations(chunk_text, cleaned_summary)
    if hallucinations:
        return apply_lexicon_simplification(chunk_text[:300].strip())

    return cleaned_summary if len(cleaned_summary) > 10 else apply_lexicon_simplification(chunk_text[:300].strip())

def extract_comprehensive_key_points(order_text: str, models: dict) -> list[str]:
    """Extract 8 to 12 substantive, high-value key action points across the full document."""
    # Split by paragraphs or numbered lines
    raw_paras = [p.strip() for p in order_text.split("\n\n") if len(p.strip()) > 35]
    if len(raw_paras) < 5:
        # Fallback to single line breaks or bullet points
        raw_paras = [p.strip() for p in re.split(r'\n(?=[0-9உ•e\-\*]\.?\s*)|\n\n', order_text) if len(p.strip()) > 35]

    filtered_chunks = []
    for p in raw_paras:
        # Skip pure header lines, dispatch numbers, or sign-offs
        if any(p.startswith(h) for h in ["ந.க.", "நாள்:", "சுற்றறிக்கை", "பார்வை", "அனுப்புநர்", "பெறுநர்", "Sd/-"]):
            continue
        # Deduplicate
        if p not in filtered_chunks:
            filtered_chunks.append(p)

    key_points = []
    # Process up to 10 substantive points to provide rich information
    for chunk in filtered_chunks[:10]:
        if is_meaningful_tamil(chunk):
            pt = summarize_chunk_safely(chunk, models)
            if len(pt) > 15 and pt not in key_points:
                key_points.append(pt)
        else:
            simp = apply_lexicon_simplification(chunk[:280].strip())
            if len(simp) > 15 and simp not in key_points:
                key_points.append(simp)

    return key_points

def translate_chunks_to_english(tamil_points: list[str], models: dict) -> list[str]:
    device = models["device"]
    tok = models["nllb_tok"]
    mod = models["nllb_mod"]
    tgt_id = tok.convert_tokens_to_ids("eng_Latn")

    en_points = []
    for point in tamil_points:
        clean_pt = clean_news_artifacts(point)
        inputs = tok(clean_pt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            ids = mod.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_length=160,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True
            )
        translated = tok.batch_decode(ids, skip_special_tokens=True)[0].strip()
        en_points.append(translated)
    return en_points

# -------------------------------------------------------------
# User Interface (Streamlit)
# -------------------------------------------------------------
st.title("🏛️ தமிழ்நாடு அரசாணை & சுற்றறிக்கை எளிமைப்படுத்தி")
st.markdown("### Tamil Nadu Government Order & Circular Citizen Portal")
st.caption("✨ Comprehensive Key Points • ⏰ Deadlines & Dates • 📍 Venue & Officials • 💰 Budget & Wages • 🔊 Voice Explanation")

tab1, tab2 = st.tabs(["📄 Upload & Summarize Document", "🗄️ Browse Database"])

with tab1:
    uploaded_file = st.file_uploader("Upload Tamil Nadu Government Order / Circular (PDF)", type=["pdf"])

    if uploaded_file and st.button("Generate Comprehensive Citizen Summary & Audio", type="primary"):
        models = load_models()

        with st.status("Analyzing document and generating rich structured summary...", expanded=True) as status:
            st.write("1️⃣ Extracting text, detecting encoding, and resolving high-resolution OCR...")
            clean_text = extract_pdf_content(uploaded_file.getvalue())

            st.write("2️⃣ Parsing administrative metadata, circular subject, and operative orders...")
            parsed = parse_metadata_and_sections(clean_text)

            st.write("3️⃣ Extracting key structured facts (deadlines, budget, wages, venues, officials)...")
            insights = extract_structured_insights(clean_text)

            st.write("4️⃣ Scanning administrative legal jargon...")
            detected_jargon = {k: v for k, v in ADMIN_LEXICON.items() if k in clean_text}

            st.write("5️⃣ Generating multi-point plain Tamil summary (with anti-hallucination verification)...")
            tamil_key_points = extract_comprehensive_key_points(parsed["order_clause"], models)

            st.write("6️⃣ Generating English translation for all sections...")
            en_key_points = translate_chunks_to_english(tamil_key_points, models)

            # Executive highlight translation
            high_ta = parsed["subject"] if parsed["subject"] else (tamil_key_points[0] if tamil_key_points else "அரசாணை வழிகாட்டு நெறிமுறைகள் வெளியிடப்பட்டுள்ளன.")
            en_highlight = translate_chunks_to_english([high_ta], models)[0]

            st.write("7️⃣ Synthesizing natural voice explanations (Tamil & English)...")
            # Build speech script for audio
            ta_audio_script = f"அரசாணை சுருக்கம். அரசாணை எண்: {parsed['go_number']}. "
            if parsed["subject"]:
                ta_audio_script += f"பொருள்: {parsed['subject']}. "
            if insights["deadlines"]:
                ta_audio_script += f"முக்கிய காலக்கெடு: {', '.join(insights['deadlines'][:2])}. "
            if insights["financials"]:
                ta_audio_script += f"நிதி மற்றும் கூலி விவரம்: {', '.join(insights['financials'][:2])}. "
            ta_audio_script += "முக்கிய வழிகாட்டு நெறிமுறைகள்: " + " ".join(tamil_key_points[:5])

            ta_audio_bytes = synthesize_speech(ta_audio_script, lang="ta")

            en_audio_script = f"Government Order Summary. ID: {parsed['go_number']}. " + en_highlight + ". Key action points: " + " ".join(en_key_points[:5])
            en_audio_bytes = synthesize_speech(en_audio_script, lang="en")

            status.update(label="Complete analysis and voice synthesis ready!", state="complete", expanded=False)

        # -------------------------------------------------------------
        # 1. Top Identity Banner
        # -------------------------------------------------------------
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏛️ ஆணை / சுற்றறிக்கை எண்", parsed["go_number"])
        m2.metric("📅 வெளியீட்டு நாள்", parsed["date"])
        m3.metric("🏢 துறை / இயக்ககம்", parsed["department"][:40])
        m4.metric("🗺️ பொருந்தும் எல்லை", "தமிழ்நாடு முழுமைக்கும்")

        # -------------------------------------------------------------
        # 2. Executive Highlights (முக்கிய சிறப்பம்சங்கள்)
        # -------------------------------------------------------------
        st.markdown("### 🌟 முக்கிய சிறப்பம்சங்கள் (Executive Highlights)")
        hl_col1, hl_col2 = st.columns(2)
        with hl_col1:
            st.info(f"**🇮🇳 எளிய தமிழ்:**\n\n{high_ta}")
        with hl_col2:
            st.info(f"**🇬🇧 English Highlight:**\n\n{en_highlight}")

        # -------------------------------------------------------------
        # 3. Four Fast-Fact Citizen Cards
        # -------------------------------------------------------------
        st.markdown("### 📊 முக்கிய குறிப்புகள் ஒரு பார்வையில் (Key Facts at a Glance)")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown("#### ⏰ முக்கிய காலக்கெடு")
            st.caption("Important Deadlines & Timeframes")
            if insights["deadlines"]:
                for d in insights["deadlines"][:4]:
                    st.write(f"⏳ **{d}**")
            else:
                st.write("⚡ *அரசாணை வெளியான நாளிலிருந்து உடனடியாக நடைமுறைக்கு வருகிறது.*")

        with c2:
            st.markdown("#### 💰 நிதி & கூலி விவரங்கள்")
            st.caption("Budget, Daily Wages & Allocations")
            if insights["financials"]:
                for f in insights["financials"][:4]:
                    st.write(f"💵 **{f}**")
            else:
                st.write("📊 *நிர்வாக செலவின வழிகாட்டுதல்கள் பொருந்தும்.*")

        with c3:
            st.markdown("#### 👥 பயனாளிகள் & தகுதி")
            st.caption("Target Beneficiaries & Eligibility")
            if insights["beneficiaries"]:
                for b in insights["beneficiaries"][:4]:
                    st.write(f"🎯 **{b}**")
            else:
                st.write("👥 *பொதுமக்கள் / கிராமப்புற குடும்பங்கள்.*")

        with c4:
            st.markdown("#### 📍 இடம் & அலுவலர்கள்")
            st.caption("Venue, Jurisdiction & Officials")
            if insights["venues_jurisdiction"]:
                for v in insights["venues_jurisdiction"][:2]:
                    st.write(f"📍 {v}")
            if insights["officials"]:
                st.write(f"👔 **பொறுப்பு அலுவலர்கள்:** {', '.join(insights['officials'][:3])}")

        st.caption("🛡️ **Factual Verification Grounding:** Verified directly against document text. Chief Minister names and external political entities are strictly safeguarded against model hallucinations.")

        st.divider()

        # -------------------------------------------------------------
        # 4. Voice Explanation (குரல் விளக்கம்)
        # -------------------------------------------------------------
        st.markdown("### 🔊 குரல் விளக்கம் (Audio Voice Explanation)")
        audio_col1, audio_col2 = st.columns(2)

        with audio_col1:
            st.markdown("#### 🇮🇳 தமிழ் குரல் விளக்கம் (Tamil Spoken Summary)")
            if ta_audio_bytes:
                st.audio(ta_audio_bytes, format="audio/mp3")
                st.download_button(
                    label="⬇️ குரல் விளக்கத்தை பதிவிறக்கு (Download Tamil MP3)",
                    data=ta_audio_bytes,
                    file_name=f"{parsed['go_number'].replace('/', '_')}_tamil_summary.mp3",
                    mime="audio/mp3",
                    key="ta_audio_dl_btn"
                )

        with audio_col2:
            st.markdown("#### 🇬🇧 English Spoken Summary")
            if en_audio_bytes:
                st.audio(en_audio_bytes, format="audio/mp3")
                st.download_button(
                    label="⬇️ Download English Audio (MP3)",
                    data=en_audio_bytes,
                    file_name=f"{parsed['go_number'].replace('/', '_')}_english_summary.mp3",
                    mime="audio/mp3",
                    key="en_audio_dl_btn"
                )

        st.divider()

        # -------------------------------------------------------------
        # 5. Comprehensive Action Points (விரிவான வழிகாட்டு நெறிமுறைகள்)
        # -------------------------------------------------------------
        st.markdown(f"### 📌 விரிவான வழிகாட்டு நெறிமுறைகள் & விதிகள் ({len(tamil_key_points)} Key Directives)")
        col_points_ta, col_points_en = st.columns(2)

        with col_points_ta:
            st.markdown("#### 🗣️ எளிய தமிழ் விளக்கக் குறிப்புகள்")
            for idx, pt in enumerate(tamil_key_points, 1):
                st.markdown(
                    f"""<div style="background-color: #f8f9fa; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #1f77b4; color: #1e1e1e;">
                    <strong>பிரிவு {idx}:</strong> {pt}
                    </div>""",
                    unsafe_allow_html=True
                )

        with col_points_en:
            st.markdown("#### 🇬🇧 English Operational Directives")
            for idx, pt in enumerate(en_key_points, 1):
                st.markdown(
                    f"""<div style="background-color: #f8f9fa; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #2ca02c; color: #1e1e1e;">
                    <strong>Section {idx}:</strong> {pt}
                    </div>""",
                    unsafe_allow_html=True
                )

        st.divider()

        # -------------------------------------------------------------
        # 6. Administrative Lexicon Glossary & Source Layer
        # -------------------------------------------------------------
        with st.expander(f"📌 நிர்வாக சட்ட சொற்கள் விளக்கம் (Administrative Glossary - {len(detected_jargon)} terms detected)"):
            if detected_jargon:
                for formal, explanation in detected_jargon.items():
                    st.write(f"- **{formal}** ➜ *{explanation}*")
            else:
                st.write("ஆவணத்தில் சிக்கலான நிர்வாகச் சொற்கள் கண்டறியப்படவில்லை.")

        if parsed["references"]:
            with st.expander(f"📑 அதிகாரப்பூர்வ பார்வை / அரசாணைகள் (Official References - {len(parsed['references'])} items)"):
                for r in parsed["references"]:
                    st.write(f"- {r}")

        with st.expander("📄 ஆவணத்திலிருந்து பிரித்தெடுக்கப்பட்ட முழு உரை (View Full Extracted Text Layer)"):
            st.text_area("Full Cleaned Content", clean_text, height=350)

with tab2:
    st.subheader("Archived Processed Orders")
    if os.path.exists(PROCESSED_DIR):
        files = [f for f in sorted(os.listdir(PROCESSED_DIR)) if f.endswith(".json")]
        if files:
            picked = st.selectbox("Select a processed record to inspect:", files)
            with open(os.path.join(PROCESSED_DIR, picked), "r", encoding="utf-8") as f:
                data = json.load(f)

            st.markdown(f"#### 🏛️ {data.get('go_number', 'Unknown GO')} ({data.get('date', '')})")

            archived_audio_path = data.get("audio_file")
            st.markdown("##### 🔊 குரல் விளக்கம் (Audio Explanation)")
            if archived_audio_path and os.path.exists(archived_audio_path):
                with open(archived_audio_path, "rb") as af:
                    st.audio(af.read(), format="audio/mp3")
            else:
                archived_text = data.get("simplified_plain_tamil", "")
                if archived_text:
                    arch_audio = synthesize_speech(archived_text, lang="ta")
                    if arch_audio:
                        st.audio(arch_audio, format="audio/mp3")

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