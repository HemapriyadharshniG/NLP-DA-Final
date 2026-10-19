# Tamil Document Simplifier (Tamil Nadu Government Orders & Circulars)

An end-to-end NLP pipeline and interactive dashboard designed to simplify, translate, and synthesize bureaucratic Tamil Government Orders (GOs) and Circulars into plain citizen-friendly language.

## 🌟 Overview

Government Orders and Circulars often use dense administrative jargon and legacy font encodings that are difficult for citizens to understand. This system transforms formal Tamil bureaucratic documents into clear, accessible plain Tamil, provides bilingual English translations, enforces anti-hallucination factual guardrails, and generates spoken audio explanations.

## 🚀 Key Features

- **Document Ingestion & Scraper**: Automated scraping of public Government Orders across departments (MGNREGS, Central Welfare Schemes, Panchayat Raj, etc.) from Tamil Nadu government portals.
- **Dual-Engine Text Extraction & OCR**: Dual-engine extraction using PyMuPDF (`fitz`) and Tesseract OCR (`lang="tam+eng"`) with Tamil Unicode NFC normalization, zero-width space stripping, and legacy font conversion.
- **Circular & GO Metadata Parser**: Robust extraction of G.O. numbers, Circular identifiers (`ந.க.எண்`, `சுற்றறிக்கை எண்`), dates, departments, and Subject lines (`பொருள்`).
- **Administrative Lexicon Mapping**: Automated substitution of complex bureaucratic Tamil phrases with citizen-friendly equivalents.
- **Anti-Hallucination & Factual Grounding**: Safeguards against news-model hallucinations, preventing fabricated Chief Minister names, political parties, or ungrounded claims.
- **AI Simplification & Summarization**: Powered by AI4Bharat's `MultiIndicSentenceSummarizationSS` (IndicBART) fine-tuned for Indic languages.
- **Neural Machine Translation**: Tamil to English translation using Meta's `nllb-200-distilled-600M` model.
- **Natural Voice Explanation (TTS)**: Synthesizes spoken Tamil (and English) audio explanations with integrated audio player and MP3 download.
- **Fact Verification**: Validates numerical quantities, dates, scheme names, and official entities between the source and simplified summaries.
- **Interactive Streamlit Dashboard**: Complete web UI to upload documents, review raw and simplified texts, view translations, and listen to synthesized audio.

## 📁 Repository Structure

```
.
├── tn_go_scraper.py            # Scraper for Tamil Nadu GOs
├── stage1_4_pipeline.py        # Ingestion, OCR, normalization & lexicon pipeline
├── stage5_simplifier.py        # IndicBART-based Tamil simplification with guardrails
├── stage6_translator.py        # NLLB Tamil-to-English translation
├── stage7_fact_check.py        # Fact verification & entity validation
├── stage8_tts.py               # Batch Voice Explanation synthesis (gTTS)
├── stage9_dashboard.py         # Streamlit web application with audio player
├── corpora_direct.py           # NLP corpora ingestion helpers
├── corpora_fix.py              # Corpora utilities
├── download_nlp_corpora.py     # NLP corpora downloader
├── dataset_raw_gos/            # Raw Government Order PDFs & metadata
├── dataset_cleaned_text/       # Cleaned & normalized text extracts
├── dataset_processed_records/  # Processed JSON records
├── dataset_simplified_records/ # Simplified JSON records
├── dataset_bilingual_records/  # Bilingual (Tamil + English) output records
├── dataset_audio_outputs/      # Synthesized TTS audio outputs
└── dataset_nlp_corpora/        # Reference corpora (WikiANN, Samanantar)
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HemapriyadharshniG/nlp.git
   cd nlp
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit torch transformers PyMuPDF pytesseract Pillow pandas requests beautifulsoup4 gtts open-tamil
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run stage9_dashboard.py
   ```

## 📄 License

MIT License / Open Source for educational and civic-tech research.
