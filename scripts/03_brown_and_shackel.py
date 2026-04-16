import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from textblob import TextBlob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. DYNAMIC PATHING ---
# This looks at where the script is and moves "up" to the anthroWorkshop folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Input: data/transcripts (Relative to the project root)
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"

# Setup Output Archive
SCRIPT_NAME = Path(__file__).stem
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"{SCRIPT_NAME}_{TIMESTAMP}"
OUTPUT_DIR = BASE_DIR / "data" / "analysis_results" / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nltk.download('stopwords', quiet=True)


def brown_shackel_cleaner(text):
    text = str(text).lower()
    text = re.sub(r"\'s", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    stop_words = set(stopwords.words('english'))
    custom_stops = {
        'interviewed', 'huh', 'yea', 'yeah', 'guess', 'uh', 'um', 'ok', 'okay',
        'well', 'going', 'got', 'went', 'said', 'back', 'like', 'know', 'think'
    }
    stop_words.update(custom_stops)

    tokens = text.split()
    return [w for w in tokens if w not in stop_words]


# --- 2. DATA PROCESSING ---
def run_workshop_pipeline():
    # Automatically finds .txt files in the relative data/transcripts folder
    transcript_files = list(TRANSCRIPT_DIR.glob("*.txt"))

    if not transcript_files:
        print(f"--- ERROR ---")
        print(f"No .txt files found in: {TRANSCRIPT_DIR}")
        print("Ensure your audio transcripts are in the 'data/transcripts' folder.")
        return

    print(f"--- Running Workshop Analysis: {RUN_ID} ---")
    print(f"Processing {len(transcript_files)} transcripts...")

    all_data = []
    for file_path in transcript_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            clean_tokens = brown_shackel_cleaner(raw_text)
            sentiment = TextBlob(raw_text).sentiment.polarity

            all_data.append({
                'filename': file_path.name[:20] + "...",
                'full_filename': file_path.name,
                'tokens': clean_tokens,
                'sentiment': sentiment
            })
        except Exception as e:
            print(f"Could not read {file_path.name}: {e}")

    df = pd.DataFrame(all_data)
    all_tokens = [t for sublist in df['tokens'] for t in sublist]

    # --- 3. ANALYTICS ---
    word_counts = Counter(all_tokens).most_common(15)
    unigram_df = pd.DataFrame(word_counts, columns=['Word', 'Count']).sort_values('Count', ascending=True)

    all_bigrams = [bg for tokens in df['tokens'] for bg in list(ngrams(tokens, 2))]
    bg_counts = Counter(all_bigrams).most_common(15)
    bigram_df = pd.DataFrame([{"Bigram": " ".join(k), "Count": v} for k, v in bg_counts]).sort_values('Count',
                                                                                                      ascending=True)

    # --- 4. VISUALIZATION ---
    print("Generating charts...")
    plt.figure(figsize=(10, 6))
    plt.barh(unigram_df['Word'], unigram_df['Count'], color='#3498db')
    plt.title(f'Frequent Words Archive: {TIMESTAMP}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_words.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(bigram_df['Bigram'], bigram_df['Count'], color='#e74c3c')
    plt.title(f'Frequent Bigrams Archive: {TIMESTAMP}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_bigrams.png")
    plt.close()

    # --- 5. EXPORT ---
    unigram_df.to_csv(OUTPUT_DIR / "unigram_counts.csv", index=False)
    bigram_df.to_csv(OUTPUT_DIR / "bigram_counts.csv", index=False)
    df.drop(columns=['tokens']).to_csv(OUTPUT_DIR / "master_report.csv", index=False)

    print(f"\nSUCCESS! Results are ready in your project folder:")
    print(f"data/analysis_results/{RUN_ID}/")


if __name__ == "__main__":
    run_workshop_pipeline()