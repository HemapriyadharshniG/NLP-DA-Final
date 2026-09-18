import os
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "dataset_nlp_corpora")
os.makedirs(CORPUS_DIR, exist_ok=True)

def download_corpora():
    # 1. Named Entity Recognition (Stage 3): WikiANN Tamil Subset
    # Modern Parquet-backed replacement for IndicNER on HF Hub
    print("[*] 1/3 Downloading NER Dataset (WikiANN Tamil subset)...")
    try:
        ner_tam = load_dataset("wikiann", "ta", split="train[:10000]")
        save_path = os.path.join(CORPUS_DIR, "wikiann_ta_sample")
        ner_tam.save_to_disk(save_path)
        print(f"    [✓] Saved {len(ner_tam)} NER records to: {save_path}")
    except Exception as e:
        print(f"    [!] Failed to download WikiANN: {e}")

    # 2. Parallel Translation Corpus (Stage 6): AI4Bharat BPCC / Samanantar (Tamil-English)
    print("\n[*] 2/3 Downloading Tamil-English Translation Sample (Samanantar / BPCC)...")
    try:
        # Samanantar parquet streaming to avoid multi-gigabyte uncompressed local overhead
        trans_dataset = load_dataset("ai4bharat/samanantar", "ta", split="train", streaming=True)
        samples = []
        for idx, item in enumerate(trans_dataset):
            samples.append(item)
            if idx >= 10000:  # Sample 10k aligned sentence pairs
                break
        
        # Save sample as parquet/arrow locally
        from datasets import Dataset
        sample_ds = Dataset.from_list(samples)
        save_path = os.path.join(CORPUS_DIR, "samanantar_ta_sample")
        sample_ds.save_to_disk(save_path)
        print(f"    [✓] Saved {len(sample_ds)} bilingual sentence pairs to: {save_path}")
    except Exception as e:
        print(f"    [!] Failed downloading translation pairs: {e}")

    # 3. Summarization / Simplification (Stage 5): Tamil abstractive summaries
    print("\n[*] 3/3 Downloading Summarization/Simplification Benchmark...")
    try:
        # Using XL-Sum (BBC Tamil news summarization benchmark - modern parquet format)
        sum_dataset = load_dataset("csebuetnlp/xlsum", "tamil", split="train[:5000]")
        save_path = os.path.join(CORPUS_DIR, "xlsum_tamil_sample")
        sum_dataset.save_to_disk(save_path)
        print(f"    [✓] Saved {len(sum_dataset)} summarization pairs to: {save_path}")
    except Exception as e:
        print(f"    [!] Failed to download summarization corpus: {e}")

    print(f"\n[✓] Ingestion complete. All NLP datasets saved in: {CORPUS_DIR}")

if __name__ == "__main__":
    download_corpora()