import whisper
import pandas as pd
from pathlib import Path

# --- 1. DYNAMIC PATHING ---
# Assumes this script is in: anthroWorkshop/scripts/01_transcribe.py
# .parent is 'scripts', .parent.parent is 'anthroWorkshop'
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "data" / "unzipped_audio"
OUTPUT_DIR = BASE_DIR / "data" / "transcripts"

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def transcribe_vhp_collection():
    # 2. VERIFY INPUT DATA
    if not INPUT_DIR.exists():
        print(f"--- ERROR ---")
        print(f"Directory not found: {INPUT_DIR}")
        print("Please ensure you have a 'data/unzipped_audio' folder in your project.")
        return

    audio_files = list(INPUT_DIR.glob("*.mp3"))

    if not audio_files:
        print(f"--- ERROR ---")
        print(f"No .mp3 files found in: {INPUT_DIR}")
        return

    # 3. INITIALIZE WHISPER
    # Note: 'medium' requires about 5GB of VRAM/RAM.
    # For older faculty laptops, 'base' or 'small' is a safer fallback if 'medium' fails.
    print("Initializing Whisper Model...")
    model = whisper.load_model("medium")

    # Priming the AI with context-specific vocabulary to help with military jargon
    vhp_prompt = "Veterans History Project, World War II, Vietnam, Korea, deployment, infantry, battalion, ammunition, igloo."

    print(f"Found {len(audio_files)} files. Starting transcription...")
    results = []

    # 4. TRANSCRIPTION LOOP
    for file_path in audio_files:
        print(f"Processing: {file_path.name}")

        try:
            # Transcribe with the context prompt
            result = model.transcribe(
                str(file_path),
                initial_prompt=vhp_prompt,
                verbose=False,
                fp16=False  # Set to False to prevent warnings on CPUs without GPUs (common on faculty laptops)
            )

            # Save individual text file (The 'Raw' transcript)
            txt_output = OUTPUT_DIR / f"{file_path.stem}.txt"
            with open(txt_output, "w", encoding="utf-8") as f:
                f.write(result["text"])

            results.append({
                "filename": file_path.name,
                "transcript": result["text"]
            })
        except Exception as e:
            print(f"Failed to transcribe {file_path.name}: {e}")

    # 5. SAVE MASTER DATASET
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_DIR / "vhp_master_transcripts.csv", index=False)
        print(f"\nSUCCESS! {len(results)} files transcribed.")
        print(f"Transcripts saved to: {OUTPUT_DIR}")
    else:
        print("Transcription failed for all files.")


if __name__ == "__main__":
    transcribe_vhp_collection()