import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from textblob import TextBlob
from pathlib import Path
from datetime import datetime

# --- 1. DYNAMIC PATH CONFIGURATION ---
# Assumes script is in: anthroWorkshop/scripts/
BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"

# Setup Timestamped Output Folder
SCRIPT_NAME = Path(__file__).stem
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"{SCRIPT_NAME}_{TIMESTAMP}"
OUTPUT_DIR = BASE_DIR / "data" / "analysis_results" / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nltk.download('stopwords', quiet=True)


def raw_text_cleaner(text):
    """
    Standardizes raw Whisper output:
    Lowercase, remove punctuation, and filter standard/custom stopwords.
    """
    text = str(text).lower()
    text = re.sub(r"\'s", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    stop_words = set(stopwords.words('english'))
    # Adding 'th' and common filler words to see how they affect the 'raw' count
    custom_stops = {
        'interviewed', 'huh', 'yea', 'yeah', 'well', 'going', 'got', 'went',
        'said', 'ok', 'okay', 'um', 'uh', 'like', 'know', 'back'
    }
    stop_words.update(custom_stops)

    tokens = text.split()
    return [w for w in tokens if w not in stop_words]


# --- 2. THEMATIC ANALYSIS ENGINE ---
def analyze_raw_transcripts():
    # Identify all individual text files
    transcript_files = list(TRANSCRIPT_DIR.glob("*.txt"))

    if not transcript_files:
        print(f"--- ERROR ---")
        print(f"No .txt files found in: {TRANSCRIPT_DIR}")
        print("Please ensure your transcripts are in the 'data/transcripts' folder.")
        return

    print(f"--- Running Raw Analysis: {RUN_ID} ---")
    print(f"Processing {len(transcript_files)} individual files...")

    all_data = []

    # Iterating through each file to build the dataset
    for file_path in transcript_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            clean_tokens = raw_text_cleaner(raw_text)
            sentiment = TextBlob(raw_text).sentiment.polarity

            all_data.append({
                'filename': file_path.name,
                'transcript': raw_text,
                'tokens': clean_tokens,
                'sentiment_score': sentiment
            })
        except Exception as e:
            print(f"Could not read {file_path.name}: {e}")

    # Convert to DataFrame for aggregate analysis
    df = pd.DataFrame(all_data)
    all_tokens = [t for sublist in df['tokens'] for t in sublist]

    # A. WORD FREQUENCY (REPRODUCING FIG 1)
    word_freq = Counter(all_tokens).most_common(50)
    unigram_df = pd.DataFrame(word_freq, columns=['Word', 'Count'])
    unigram_df.to_csv(OUTPUT_DIR / "raw_top_words.csv", index=False)

    # B. BIGRAMS & TRIGRAMS (REPRODUCING TABLE 1 & 2)
    all_bigrams = [bg for tokens in df['tokens'] for bg in list(ngrams(tokens, 2))]
    all_trigrams = [tg for tokens in df['tokens'] for tg in list(ngrams(tokens, 3))]

    bigram_df = pd.DataFrame([{"Bigram": " ".join(k), "Count": v} for k, v in Counter(all_bigrams).most_common(50)])
    bigram_df.to_csv(OUTPUT_DIR / "raw_top_bigrams.csv", index=False)

    trigram_df = pd.DataFrame([{"Trigram": " ".join(k), "Count": v} for k, v in Counter(all_trigrams).most_common(50)])
    trigram_df.to_csv(OUTPUT_DIR / "raw_top_trigrams.csv", index=False)

    # C. SAVE MASTER SUMMARY
    # Dropping 'tokens' for a cleaner CSV output
    df.drop(columns=['tokens']).to_csv(OUTPUT_DIR / "raw_thematic_summary.csv", index=False)

    # --- 3. CONSOLE REPORT ---
    print("\n" + "=" * 40)
    print("RAW DATA ANALYSIS REPORT")
    print("=" * 40)
    print("\nTop 10 RAW Trigrams (Thematic Context):")
    for tg, count in Counter(all_trigrams).most_common(10):
        print(f"- {' '.join(tg):25} | {count}")

    print("\nSentiment Summary:")
    print(df[['filename', 'sentiment_score']].to_string(index=False))
    print("\n" + "=" * 40)
    print(f"Results saved to project folder:")
    print(f"data/analysis_results/{RUN_ID}/")


if __name__ == "__main__":
    analyze_raw_transcripts()