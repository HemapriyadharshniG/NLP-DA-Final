import os
from datasets import load_dataset

CORPUS_DIR = "/Users/hemapriyadharshni/Documents/tamil_doc_simplifier/dataset_nlp_corpora"
os.makedirs(CORPUS_DIR, exist_ok=True)

# 1. Modern Native Parquet NER (WikiANN Tamil official Parquet conversion)
print("[*] 1/2 Downloading Parquet-native Tamil NER dataset...")
try:
    ner_ta = load_dataset("Babelscape/wikiann", "ta", split="train[:5000]")
    ner_ta.save_to_disk(os.path.join(CORPUS_DIR, "wikiann_ta_sample"))
    print(f"    [✓] Saved {len(ner_ta)} NER records.")
except Exception as e:
    print(f"    [!] NER failed: {e}")

# 2. Modern Native Parquet Summarization (Indic Headline Generation - Tamil)
print("\n[*] 2/2 Downloading Parquet-native Tamil Summarization benchmark...")
try:
    sum_ta = load_dataset("google/indic_headline_generation", "ta", split="train[:2000]")
    sum_ta.save_to_disk(os.path.join(CORPUS_DIR, "xlsum_tamil_sample"))
    print(f"    [✓] Saved {len(sum_ta)} summarization pairs.")
except Exception as e:
    print(f"    [!] Summarization failed: {e}")

print("\n[✓] Finished updating missing NLP corpora.")
