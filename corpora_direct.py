import os
from datasets import load_dataset

CORPUS_DIR = "/Users/hemapriyadharshni/Documents/tamil_doc_simplifier/dataset_nlp_corpora"
os.makedirs(CORPUS_DIR, exist_ok=True)

# 1. WikiANN Tamil (Stage 3 NER) - Direct Parquet Shard
print("[*] 1/2 Fetching WikiANN Tamil NER via direct Parquet...")
try:
    ner_url = "https://huggingface.co/datasets/unimelb-nlp/wikiann/resolve/refs%2Fconvert%2Fparquet/ta/train/0000.parquet"
    ner_ds = load_dataset("parquet", data_files={"train": ner_url}, split="train[:5000]")
    ner_path = os.path.join(CORPUS_DIR, "wikiann_ta_sample")
    ner_ds.save_to_disk(ner_path)
    print(f"    [✓] Successfully saved {len(ner_ds)} NER rows to {ner_path}")
except Exception as e:
    print(f"    [!] NER download failed: {e}")

# 2. XL-Sum Tamil (Stage 5 Summarization) - Direct Parquet Shard
print("\n[*] 2/2 Fetching XL-Sum Tamil Summarization via direct Parquet...")
try:
    xlsum_url = "https://huggingface.co/datasets/GEM/xlsum/resolve/main/tamil/xlsum-train.parquet"
    sum_ds = load_dataset("parquet", data_files={"train": xlsum_url}, split="train[:2000]")
    sum_path = os.path.join(CORPUS_DIR, "xlsum_tamil_sample")
    sum_ds.save_to_disk(sum_path)
    print(f"    [✓] Successfully saved {len(sum_ds)} Summarization pairs to {sum_path}")
except Exception as e:
    print(f"    [!] Summarization download failed: {e}")

print("\n[✓] Finished. You have all reference corpora ready.")
