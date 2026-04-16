import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# --- 1. SETUP ---
# Path logic to stay inside the anthroWorkshop folder
BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_DIR / "data" / "analysis_results" / f"dispersion_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_lexical_dispersion(target_word="war"):
    transcript_files = sorted(list(TRANSCRIPT_DIR.glob("*.txt")))

    if not transcript_files:
        print(f"Error: No transcripts found in {TRANSCRIPT_DIR}")
        return

    plt.figure(figsize=(12, 8))

    for i, file_path in enumerate(transcript_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = f.read().lower().split()

            # Find every index where the target word appears
            # Using 'word.strip()...' to handle punctuation attached to words
            offsets = [index for index, word in enumerate(words) if target_word in word.strip(",.?!()\"")]

            # Plot those points as vertical lines (ticks)
            plt.vlines(offsets, i + 0.5, i + 1.5, color='#2c3e50', alpha=0.8, linewidth=1.5)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Styling the plot for a professional slide look
    plt.yticks(range(1, len(transcript_files) + 1), [f.name[:25] for f in transcript_files])
    plt.ylim(0.5, len(transcript_files) + 0.5)
    plt.title(f"Narrative Timeline: Occurrences of '{target_word}'", fontsize=14, pad=20)
    plt.xlabel("Progress through Interview (Word Count Index)", fontsize=12)
    plt.ylabel("Veteran Interview File", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save with the target word in the filename
    save_path = OUTPUT_DIR / f"dispersion_{target_word}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"--- Visualization Complete ---")
    print(f"Target Word: {target_word}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    # NOW IT LOOKS FOR WAR
    plot_lexical_dispersion(target_word="war")

    # Optional: Run it twice so you have two charts for your slides!
    # plot_lexical_dispersion(target_word="home")