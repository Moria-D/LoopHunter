import json
import pandas as pd
import os
import sys
from division import AudioRemixer

# Configuration
AUDIO_PATH = r"C:\Users\Administrator\Desktop\audio\music\bgm\hip.mp3"
OUTPUT_DIR = "batch_output"
OUTPUT_FILE_PREFIX = "instrument_analysis"

def main():
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: File not found at {AUDIO_PATH}")
        return

    print(f"Initializing AudioRemixer for {AUDIO_PATH}...")
    try:
        remixer = AudioRemixer(AUDIO_PATH)
        # Note: We don't need full 'analyze()' if we just want stems, but 'analyze' does pre-processing.
        # Let's run full analyze to be safe as it sets up duration and pre-processing.
        remixer.analyze() 
        
        print("Analyzing instrument stems (Drums, Bass, Melody)...")
        events = remixer.analyze_stems()
        
        # Prepare data for Export
        # We will create a list of all events with Type
        all_rows = []
        for instr_type, event_list in events.items():
            for evt in event_list:
                all_rows.append({
                    "Instrument": instr_type,
                    "Start Time (s)": round(evt['start'], 3),
                    "Duration (s)": round(evt['duration'], 3),
                    "End Time (s)": round(evt['start'] + evt['duration'], 3)
                })
        
        # Sort by start time
        all_rows.sort(key=lambda x: x["Start Time (s)"])
        
        # 1. Save to JSON
        json_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_PREFIX}.json")
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(events, f, indent=4, ensure_ascii=False)
        print(f"Saved instrument analysis to JSON: {json_path}")
        
        # 2. Save to Excel
        excel_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_PREFIX}.xlsx")
        df = pd.DataFrame(all_rows)
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Saved instrument analysis to Excel: {excel_path}")
        
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()

