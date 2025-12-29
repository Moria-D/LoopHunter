import numpy as np
import librosa

def detect_first_transient(y, sr, threshold_db=-60):
    """
    Detects the first significant transient (onset) in the audio.
    Uses a "Backwards Noise-Floor Scan" strategy:
    1. Finds coarse "Loud" point.
    2. Estimates noise floor from the preceding quiet section.
    3. Walks backwards from the Loud point until amplitude drops to the noise floor.
    
    Args:
        y: Audio time series
        sr: Sample rate
        threshold_db: Threshold relative to max peak to consider as silence
    
    Returns:
        float: Time in seconds of the first transient
    """
    if y is None or len(y) == 0:
        return 0.0

    try:
        # 1. Coarse detection (Find where the main body of sound is)
        # Use a fairly high threshold to ensure we are inside the sound, not in the noise.
        # top_db=60 is standard, but for this reverse-walk strategy, we want to be well inside.
        non_silent_intervals = librosa.effects.split(y, top_db=60, frame_length=1024, hop_length=256)
        if len(non_silent_intervals) == 0:
            return 0.0
            
        coarse_start_sample = non_silent_intervals[0][0]
        
        # If the file starts loud immediately, return 0
        if coarse_start_sample < 512:
            return 0.0
            
        # 2. Estimate Noise Floor
        # Analyze the region *before* the coarse start (leave a 50ms buffer to avoid the attack tail)
        buffer_samples = int(0.05 * sr)
        noise_region_end = max(0, coarse_start_sample - buffer_samples)
        
        # Default low threshold (absolute silence/quantization noise)
        base_threshold = 0.0002
        
        if noise_region_end > 1024:
            # Check noise floor in the "silence"
            noise_chunk = np.abs(y[:noise_region_end])
            # Use 3 * RMS as a safe noise threshold, or max if it's sparse clicks
            noise_rms = np.sqrt(np.mean(noise_chunk**2))
            # Dynamic threshold: slightly above background noise
            threshold = max(base_threshold, noise_rms * 4.0)
            
            # Safety cap: don't let threshold get too high if "silence" is actually loud
            # (e.g. if coarse detection was late)
            threshold = min(threshold, 0.01)
        else:
            # Not enough pre-audio to estimate, use base
            threshold = base_threshold

        # 3. Precise Backtracking (Reverse Walk)
        # Walk backwards from coarse_start_sample until signal drops below threshold
        # We look back up to 500ms
        max_lookback = int(0.5 * sr)
        search_start = max(0, coarse_start_sample - max_lookback)
        
        # Extract the region of interest: [search_start ... coarse_start + small_buffer]
        # Include a small buffer forward just in case split was very early (unlikely)
        roi_end = min(len(y), coarse_start_sample + 1024)
        roi = np.abs(y[search_start : roi_end])
        
        # We scan BACKWARDS from the coarse point (relative to ROI)
        start_idx_rel = coarse_start_sample - search_start
        
        # Window for checking "sustained silence" (e.g. 1ms) for tighter precision
        silence_window = int(0.001 * sr)
        
        detected_idx = start_idx_rel
        
        # Iterate backwards
        # We look for the point where the signal *was* below threshold for `silence_window` samples
        # i.e., we are in the signal, walking left. We stop when we hit the "shore" of silence.
        
        for i in range(start_idx_rel, silence_window, -1):
            # Check a small window to the left
            # If max(window) < threshold, we found silence.
            # The start point is i.
            
            # Optimization: check sample `i`. If it's loud, continue.
            if roi[i] > threshold:
                continue
                
            # If sample is quiet, check if it's just a zero crossing or real silence
            # Look at [i - window : i]
            window = roi[i - silence_window : i]
            if np.max(window) < threshold:
                # Found the noise floor!
                # The signal starts at i + 1 (approx)
                detected_idx = i
                break
                
        # 4. Final Micro-Refinement (Zero-Crossing)
        # We found the point where amplitude rises above noise floor.
        # Now find the nearest zero-crossing to the left (within small margin) to start cleanly.
        # Actually, `detected_idx` is already in the noise. The first signal sample is `detected_idx + 1` or so.
        # Let's scan forward from `detected_idx` to find the first upward trend? 
        # No, `detected_idx` is likely the best cut point (silence).
        
        # Just ensure we didn't go back too far (to 0) if it wasn't necessary.
        final_sample = search_start + detected_idx
        
        return float(final_sample) / sr

    except Exception:
        # Fallback to coarse
        return 0.0

def calculate_global_beat_duration(beat_times, total_duration, bpm_override=None, y=None, sr=None):
    """
    Calculates the optimal global beat duration (period) and alignment offset.
    Uses linear regression on valid beat times to minimize drift.
    
    NEW: If y and sr are provided, it detects the actual audio start (first transient)
    and forces the grid to align with it, ensuring the first slice starts exactly at the sound.
    """
    # 1. Determine base period estimate
    period_est = 0.5
    
    if bpm_override is not None and bpm_override > 0:
        period_est = 60.0 / bpm_override
    else:
        # Infer from beat_times using median difference
        if beat_times is not None and len(beat_times) > 1:
            diffs = np.diff(np.sort(beat_times))
            # Filter unlikely intervals (assuming 40-250 BPM -> 0.24s - 1.5s)
            valid_diffs = diffs[(diffs > 0.2) & (diffs < 1.5)]
            if len(valid_diffs) > 0:
                period_est = np.median(valid_diffs)
    
    # 2. Period Refinement (using beat_times if available)
    best_period = period_est
    best_offset = 0.0
    
    # Calculate best_period using regression on beat_times (for stability of Tempo)
    if beat_times is not None and len(beat_times) > 0:
        beats = np.sort(beat_times)
        beats = beats[(beats >= 0) & (beats <= total_duration)]
        
        if len(beats) > 1 and not bpm_override:
            # Linear Regression to refine Period
            first_beat = beats[0]
            approx_indices = np.round((beats - first_beat) / period_est)
            
            n = len(beats)
            if n >= 2:
                A = np.vstack([approx_indices, np.ones(n)]).T
                m, c = np.linalg.lstsq(A, beats, rcond=None)[0]
                if 0.8 * period_est < m < 1.2 * period_est:
                    best_period = m
                    # We tentatively use c as offset, but will override below if transients found
                    best_offset = c 
                else:
                    best_offset = first_beat
            else:
                best_offset = first_beat
        elif bpm_override:
            # Trust the override period
            best_period = period_est
            # Tentative offset
            if len(beats) > 0:
                 best_offset = beats[0]
    
    # 3. Offset Refinement (The "Start Point" optimization)
    # The regression offset might be mathematically optimal for minimizing error sum,
    # but practically, users want the first slice to start EXACTLY at the first sound.
    # We detect the first transient and force the grid to align with it.
    
    if y is not None and sr is not None:
        first_transient_time = detect_first_transient(y, sr)
        
        # If the detected transient is reasonable (e.g. not silence at 0)
        # We assume this is the start of the first beat (or first relevant event)
        # We align the grid such that one grid line falls exactly on first_transient_time.
        
        # Current grid model: t = best_offset + k * best_period
        # New constraint: first_transient_time = best_offset_new + 0 * best_period
        # So best_offset_new = first_transient_time
        
        # Wait, what if first_transient is actually the 2nd beat?
        # We compare first_transient with the regression's predicted first beat.
        
        # Regression prediction for first beat (k=0 relative to regression)
        # reg_start = best_offset
        
        # If first_transient is close to reg_start (within half a period), we snap to it.
        # If first_transient is far (e.g. intro is silence + pickup), we still want to respect it.
        
        # Ideally, "Offset" should represent the start of the audio content.
        # Let's set best_offset to first_transient_time.
        # This effectively shifts the whole grid to align with the start of the audio.
        # This satisfies the user requirement: "offset=0.2 if sound starts at 0.2".
        
        # Validation: Is first_transient_time consistent with the beat grid?
        # If the audio has a pickup (anacrusis) that is NOT on the grid, this might shift the grid incorrectly.
        # However, for loop slicing, usually the first sound IS the first downbeat (or the user expects it to be).
        
        # Let's trust the physical start of sound as the Anchor.
        best_offset = first_transient_time

        # 4. Tail Optimization (Micro-alignment based on Last Slice)
        # Check if the grid alignment causes a tiny "remainder" slice at the end or cuts off slightly early.
        # This assumes the total file duration is rhythmically significant (e.g. an exact loop).
        
        # Calculate where the last grid point falls relative to total_duration
        # grid points: t = best_offset + k * best_period
        
        # Find the last grid point <= total_duration
        if best_period > 0:
            # Shift offset into [0, Period) relative to total_duration just for modulo check
            # remainder = (total_duration - best_offset) % best_period
            
            # More robustly: calculate duration of the potential last slice
            # Last full beat index
            last_k = int(np.floor((total_duration - best_offset) / best_period))
            last_grid_time = best_offset + last_k * best_period
            
            tail_duration = total_duration - last_grid_time
            
            # Logic:
            # If tail_duration is very small (e.g. 0.02s of a 0.5s beat), 
            # it likely means we started too early (shifted left), so the grid finished just before the end.
            # We should SHIFT RIGHT (add tail_duration) to close the gap.
            #
            # If tail_duration is very large (close to a full beat, e.g. 0.46s of 0.48s),
            # it likely means we started too late (shifted right), so the last beat got cut off.
            # We should SHIFT LEFT (subtract (Period - tail_duration)) to include the full beat.
            
            # Thresholds: 
            # - Small tail: < 5% of period or < 50ms (whichever is larger, but bounded)
            # - Cutoff beat: > 95% of period
            
            # Let's use the user's logic: 0.02 vs 0.48. 0.02 is ~4%.
            
            micro_shift = 0.0
            
            # Case A: Tiny tail (Grid finished early) -> Shift Offset Right (+)
            if tail_duration > 0 and tail_duration < (0.1 * best_period): 
                # Limit correction to avoid massive jumps if it's just a random file
                if tail_duration < 0.1: # Max 100ms correction
                    micro_shift = tail_duration
            
            # Case B: Almost full beat (Grid started late) -> Shift Offset Left (-)
            # The "missing part" is (best_period - tail_duration)
            missing_part = best_period - tail_duration
            if missing_part > 0 and missing_part < (0.1 * best_period):
                 if missing_part < 0.1:
                    micro_shift = -missing_part
            
            if abs(micro_shift) > 0.00001:
                best_offset += micro_shift
        
    else:
        # Fallback if no audio provided: Ensure offset isn't negative
        pass

    return best_period, best_offset
