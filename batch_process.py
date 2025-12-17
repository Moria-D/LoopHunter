import os
import sys
import json
import time
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from division import AudioRemixer

# Configuration
AUDIO_PATH = r"C:\Users\Administrator\Desktop\audio\music\bgm\hip.mp3"
OUTPUT_DIR = "batch_output"

def save_remix_waveform(remix_y, sr, timeline, total_remix_dur, output_path):
    """Generates and saves a waveform image with highlighted segments and jump points."""
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.set_xlim(0, total_remix_dur)
    ax.set_xlabel("Time (s)", color='#8b949e')
    ax.tick_params(axis='x', colors='#8b949e')
    ax.set_yticks([])
    
    step = 100 # Downsample for plotting speed
    
    for i, seg in enumerate(timeline):
        start_time = seg['remix_start']
        duration = seg['duration']
        end_time = start_time + duration
        
        label = seg['type']
        # Colors matching the app style
        color = '#238636' # Default Green (Linear)
        if label == 'Loop Extension': color = '#1f6feb' # Blue
        if seg.get('is_jump'): color = '#d2a8ff' # Purple (Tail/Jump)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        if end_sample > len(remix_y): end_sample = len(remix_y)
        if start_sample >= end_sample: continue
        
        segment_y = remix_y[start_sample:end_sample:step]
        segment_times = np.linspace(start_time, end_time, num=len(segment_y))
        
        ax.plot(segment_times, segment_y, color=color, linewidth=0.8, alpha=0.9)
        ax.axvline(x=start_time, color='white', linestyle=':', linewidth=0.8, alpha=0.6)
        
        # Mark jump points
        if seg.get('xfade', 0) > 0 or seg.get('is_jump'):
             ax.scatter([start_time], [0], color='white', s=20, zorder=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#30363d')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: File not found at {AUDIO_PATH}")
        return

    print(f"Initializing AudioRemixer for {AUDIO_PATH}...")
    try:
        remixer = AudioRemixer(AUDIO_PATH)
        remixer.analyze()
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return
    
    original_duration = remixer.duration
    print(f"Original duration: {original_duration:.2f}s")
    
    start_target = 30
    end_target = int(original_duration)
    
    if start_target > end_target:
        print(f"Warning: Audio duration ({original_duration:.2f}s) is shorter than start target (30s).")
        return

    print(f"Starting batch processing from {start_target}s to {end_target}s...")
    
    total_start_time = time.time()
    all_info = {}

    # Run Stem Analysis ONCE
    print("Analyzing instrument stems (Drums/Bass/Melody)...")
    stem_events = remixer.analyze_stems()
    
    # Iterate through each second
    for target_dur in range(start_target, end_target + 1):
        iter_start_time = time.time()
        print(f"[{target_dur}/{end_target}] Processing target duration: {target_dur}s...")
        
        try:
            # 1. Plan Remix
            timeline, actual_dur = remixer.plan_multi_loop_remix(target_dur)
            
            # 2. Render Audio
            audio = remixer.render_remix(timeline)
            
            # Normalize audio
            mx = np.max(np.abs(audio))
            if mx > 0: audio = audio / mx * 0.95
            
            base_name = f"remix_{target_dur}s"
            audio_filename = f"{base_name}.wav"
            image_filename = f"{base_name}.png"
            
            audio_out_path = os.path.join(OUTPUT_DIR, audio_filename)
            image_out_path = os.path.join(OUTPUT_DIR, image_filename)
            
            # Save Audio
            sf.write(audio_out_path, audio, remixer.sr)
            
            # 3. Save Waveform
            save_remix_waveform(audio, remixer.sr, timeline, actual_dur, image_out_path)
            
            # 4. Collect Jump Points, Composition, and Instrument Events
            jump_points = []
            composition = []
            
            # Map Stem Events to Remix Timeline
            remix_stem_events = {k: [] for k in stem_events.keys()}
            
            for seg in timeline:
                # Jump Points
                if seg.get('xfade', 0) > 0 or seg.get('is_jump') or seg['type'] == 'Loop Extension':
                     jump_points.append({
                         'time': round(seg['remix_start'], 2),
                         'type': seg['type'],
                         'xfade': seg.get('xfade', 0)
                     })
                
                # Composition
                content_type = remixer._classify_segment(seg['source_start'], seg['source_end'])
                composition.append({
                    "start": round(seg['remix_start'], 2),
                    "duration": round(seg['duration'], 2),
                    "content_type": content_type,
                    "structure_type": seg['type']
                })
                
                # Map Stem Events
                # For this segment (source_start -> source_end), find events in source
                seg_s = seg['source_start']
                seg_e = seg['source_end']
                remix_offset = seg['remix_start'] - seg_s
                
                for inst_name, events in stem_events.items():
                    for evt in events:
                        # Check if event start is within this source segment
                        # Relax boundary slightly for events starting right on edge
                        if evt['start'] >= seg_s and evt['start'] < seg_e:
                            mapped_evt = {
                                "start": round(evt['start'] + remix_offset, 3),
                                "duration": round(evt['duration'], 3)
                            }
                            remix_stem_events[inst_name].append(mapped_evt)

            all_info[str(target_dur)] = {
                "target_duration": target_dur,
                "actual_duration": round(actual_dur, 2),
                "process_time": round(time.time() - iter_start_time, 4),
                "jump_points": jump_points,
                "composition": composition,
                "instrument_events": remix_stem_events, # Added mapped events
                "files": {
                    "audio": audio_filename,
                    "image": image_filename
                }
            }
            
            iter_duration = time.time() - iter_start_time
            print(f"   -> Completed in {iter_duration:.2f}s")
            
        except Exception as e:
            print(f"Error processing {target_dur}s: {e}")
            continue
        
    # Save collected info to JSON
    total_duration = time.time() - total_start_time
    
    final_output = {
        "summary": {
            "total_process_time": round(total_duration, 4),
            "item_count": len(all_info)
        },
        "data": all_info
    }
    
    json_path = os.path.join(OUTPUT_DIR, "batch_info.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"Batch processing complete. Results saved in '{OUTPUT_DIR}'")
    print(f"Total Processing Time: {total_duration:.2f}s")
    print(f"Summary JSON saved to '{json_path}'")

if __name__ == "__main__":
    main()

