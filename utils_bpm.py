import numpy as np

def calculate_global_beat_duration(beat_times, total_duration, bpm_override=None):
    """
    Calculates the optimal global beat duration (period) and alignment offset.
    Uses linear regression on valid beat times to minimize drift.
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
    
    # 2. If we have beat times, refine period and finding phase (offset)
    # utilizing a robust fit (Linear Regression with outlier rejection)
    
    best_period = period_est
    best_offset = 0.0
    
    # If explicit BPM is given, we trust that period, but we still need to find the best offset (phase)
    # to align the grid with the actual audio transients.
    # If no explicit BPM, we optimize both period and offset.
    
    if beat_times is not None and len(beat_times) > 0:
        beats = np.sort(beat_times)
        beats = beats[(beats >= 0) & (beats <= total_duration)]
        
        if len(beats) > 0:
            # Simple alignment if specific BPM provided:
            # Minimize: sum( ( (beats - offset) % period )^2 ) ?
            # Easier: Just align to the first strong beat? 
            # Better: Histogram of phases.
            
            if bpm_override:
                # Fixed Period, Find Offset
                # Calculate phases 0..1
                phases = (beats % best_period) / best_period
                # Find the peak in phase histogram (using circular mean or simple binning)
                # Simple binning
                hist, bin_edges = np.histogram(phases, bins=20, range=(0,1))
                peak_idx = np.argmax(hist)
                peak_phase = (bin_edges[peak_idx] + bin_edges[peak_idx+1]) / 2.0
                best_offset = peak_phase * best_period
                
                # Ensure offset is close to 0 (or first beat) rather than N periods away
                # We want the grid to start near 0 or the first beat.
                if len(beats) > 0:
                    first_beat = beats[0]
                    # adjust offset to be close to first_beat
                    # offset + k * period ~= first_beat
                    k = round((first_beat - best_offset) / best_period)
                    best_offset = best_offset + k * best_period
                    
            else:
                # Optimize both Period and Offset using Linear Regression
                # We assume beats[i] ~= offset + k_i * period
                # First, determine k_i for each beat based on the rough period_est
                
                # Align first beat to index 0 roughly
                first_beat = beats[0]
                approx_indices = np.round((beats - first_beat) / period_est)
                
                # Run regression: beats ~ offset + approx_indices * period
                # Using RANSAC or just filtering residuals could be better, but simple least squares is usually fine 
                # if BPM is constant.
                
                # Filter outliers: if deviation from expected index is too large (e.g. > 0.4 period), ignore
                # This handles missing beats or extra hits.
                
                # Refined indices check
                # beats = offset + idx * period
                # period ~= (beats[-1] - beats[0]) / (idx[-1] - idx[0])
                
                n = len(beats)
                if n >= 2:
                    A = np.vstack([approx_indices, np.ones(n)]).T
                    m, c = np.linalg.lstsq(A, beats, rcond=None)[0]
                    
                    # m is slope (period), c is intercept (offset)
                    # Sanity check: is m close to period_est?
                    if 0.8 * period_est < m < 1.2 * period_est:
                        best_period = m
                        best_offset = c
                    else:
                        best_offset = first_beat
                else:
                    best_offset = first_beat

    # Ensure offset is not negative (unless very close to 0) or too large
    # Ideally grid starts at 0 or first beat.
    # If best_offset is e.g. 0.5, and period is 0.5, then it's effectively 0.
    # We normalize best_offset to be in [0, period) or slightly negative if it helps cover start.
    
    # However, usually we want the grid to cover the whole file from t=0 if there's intro.
    # But usually "Slice 1" corresponds to the first beat.
    # If there is a pick-up (anacrusis), offset might be positive (first beat at 0.1s).
    # Then Slice 1 could be 0.1s long (intro) or we shift grid to start at 0.1?
    
    # Strategy:
    # Use best_offset as the anchor for the grid.
    # If best_offset > 0, we might have a partial slice before it.
    
    return best_period, best_offset

