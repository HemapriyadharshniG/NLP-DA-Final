import os
import json
import re
from gtts import gTTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "dataset_bilingual_records")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_audio_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_for_speech(text: str) -> str:
    """Strip markdown markers and excessive whitespace for natural spoken voice."""
    text = re.sub(r'[*_#`\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def run_tts():
    files = [f for f in sorted(os.listdir(INPUT_DIR)) if f.endswith(".json")]
    print(f"[*] Processing {len(files)} records for Stage 8 Voice Synthesis...")

    for f in files:
        file_path = os.path.join(INPUT_DIR, f)
        with open(file_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        text_to_speak = data.get("simplified_plain_tamil", "")
        if not text_to_speak or len(text_to_speak.strip()) < 10:
            print(f" [!] Skipping empty/short text: {f}")
            continue

        clean_text = clean_for_speech(text_to_speak)[:1200]
        audio_filename = f.replace(".json", ".mp3")
        audio_path = os.path.join(OUTPUT_DIR, audio_filename)

        print(f"[*] Synthesizing voice for: {audio_filename}...")
        try:
            tts = gTTS(text=clean_text, lang="ta", slow=False)
            tts.save(audio_path)
            data["audio_file"] = audio_path

            # Update JSON with audio path
            with open(file_path, "w", encoding="utf-8") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=2)

            print(f" [✓] Audio saved: {audio_filename}")
        except Exception as e:
            print(f" [!] Failed synthesizing audio for {f}: {e}")

    print(f"\n[✓] Stage 8 Complete! Synthesized audio files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_tts()
