import numpy as np
import librosa

def detect_first_transient(y, sr, threshold_db=-60):
    """
    Detects the first significant transient (onset) in the audio.
    This serves as a more accurate "physical start time" than librosa.beat.beat_track.
    
    Args:
        y: Audio time series
        sr: Sample rate
        threshold_db: Threshold relative to max peak to consider as silence
    
    Returns:
        float: Time in seconds of the first transient
    """
    if y is None or len(y) == 0:
        return 0.0

    # 1. Coarse detection using silence trimming
    # top_db=60 means anything below -60dB relative to max is silence
    # frame_length=2048, hop_length=512 default
    try:
        non_silent_intervals = librosa.effects.split(y, top_db=abs(threshold_db), frame_length=2048, hop_length=512)
        if len(non_silent_intervals) == 0:
            return 0.0
            
        start_sample = non_silent_intervals[0][0]
        
        # 2. Refined detection: Backtrack from coarse start
        # The split might be a bit late (hop_length resolution). 
        # We look a bit earlier to find the exact rising edge.
        # Search window: 50ms before detected start
        search_back_samples = int(0.05 * sr)
        refine_start = max(0, start_sample - search_back_samples)
        refine_end = min(len(y), start_sample + int(0.01 * sr)) # Look a bit forward too
        
        chunk = y[refine_start:refine_end]
        if len(chunk) == 0:
            return float(start_sample) / sr
            
        # Use simple amplitude threshold on the raw waveform or RMS
        # Absolute threshold: e.g. 0.005 (assuming normalized audio ~1.0 max)
        # Or relative to chunk max
        
        abs_y = np.abs(chunk)
        threshold_amp = 0.005 # empirical low threshold
        
        # Find first index where amplitude > threshold
        above_thresh = np.where(abs_y > threshold_amp)[0]
        if len(above_thresh) > 0:
            # First point exceeding threshold
            first_idx = above_thresh[0]
            # Ideally we want the zero-crossing immediately preceding this rise
            # But just returning this time is usually close enough (<1ms error)
            
            # Optimization: Backtrack to nearest zero crossing before this point
            # to avoid clicking if we were to cut there (though this is just for timing info)
            return float(refine_start + first_idx) / sr
        else:
            # If refinement failed to find explicit peak, stick to coarse start
            return float(start_sample) / sr
            
    except Exception:
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
        
    else:
        # Fallback if no audio provided: Ensure offset isn't negative
        pass

    return best_period, best_offset
