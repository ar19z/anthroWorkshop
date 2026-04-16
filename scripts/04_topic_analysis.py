import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import gensim
from gensim import corpora
from datetime import datetime
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# --- 1. DYNAMIC PATHING ---
BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"

SCRIPT_NAME = Path(__file__).stem
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"{SCRIPT_NAME}_{TIMESTAMP}"
OUTPUT_DIR = BASE_DIR / "data" / "analysis_results" / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nltk.download('stopwords', quiet=True)


def preprocess_for_lda(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    stop_words = set(stopwords.words('english'))
    custom_stops = {
        'yeah', 'get', 'well', 'going', 'got', 'went', 'said', 'back',
        'know', 'think', 'one', 'come', 'right', 'would', 'could', 'didnt',
        'dont', 'time', 'us', 'oh', 'lot', 'something', 'really', 'much', 'th'
    }
    stop_words.update(custom_stops)
    return [w for w in text.split() if w not in stop_words and len(w) > 3]


# --- 2. TOPIC MODELING ENGINE ---
def run_topic_analysis(num_topics=4):
    transcript_files = list(TRANSCRIPT_DIR.glob("*.txt"))
    if not transcript_files:
        print(f"Error: No transcripts found in {TRANSCRIPT_DIR}")
        return

    print(f"--- Running Interactive Topic Analysis: {RUN_ID} ---")

    processed_docs = []
    filenames = []
    for file_path in transcript_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        processed_docs.append(preprocess_for_lda(content))
        filenames.append(file_path.name)

    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    if not corpus:
        print("Error: Corpus is empty.")
        return

    # Train LDA Model
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=20
    )

    # --- 3. INTERACTIVE VISUALIZATION (pyLDAvis) ---
    print("Generating pyLDAvis interactive HTML...")
    # This prepares the mathematical model for visual representation
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds')

    # Save the dashboard
    vis_output_path = OUTPUT_DIR / "interactive_topic_map.html"
    pyLDAvis.save_html(vis_data, str(vis_output_path))

    # --- 4. EXPORT DATA ---
    topics_list = [{"Topic_ID": idx, "Words": topic} for idx, topic in lda_model.print_topics(-1)]
    pd.DataFrame(topics_list).to_csv(OUTPUT_DIR / "discovered_topics.csv", index=False)

    print(f"\nSUCCESS!")
    print(f"1. Open your browser and drag the file: {vis_output_path}")
    print(f"2. Use the 'Lambda' slider to adjust word relevance.")


"""
pyLDAvis is an interactive web dashboard that transforms complex topic 
modeling math into a spatial map, where circles on the left represent 
themes and bar charts on the right display their defining words. 
The Lambda slider is a "tuning" tool that allows researchers 
to adjust the relevance of words: setting it to 1.0 shows the most frequent 
words in a topic, which are often generic, while sliding it toward 
0.0 highlights unique, "signature" words that distinguish that specific topic from 
others. In an anthropological context, this allows you to move beyond high-level 
summaries and discover specific cultural nuances—like a veteran's unique slang 
or specific geographic memories—by finding the optimal balance 
(usually around 0.6) between word frequency and exclusivity.
"""


if __name__ == "__main__":
    run_topic_analysis(num_topics=4)