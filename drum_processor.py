import numpy as np
import librosa
import scipy.signal
from scipy.ndimage import median_filter

class DrumLoopExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
        self.bands = {
            'kick': (20, 150), 'snare_perc': (200, 3500), 'cymbals': (3500, None),
            'bass': (20, 500), 'instruments': (200, 8000)
        }

    def _get_musical_musicality_score(self, y):
        """乐理分数计算：检测片段的音色纯度，过滤 Lofi 环境音效"""
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        y_harm, _ = librosa.effects.hpss(y)
        harm_energy = np.sum(y_harm**2)
        total_energy = np.sum(y**2) + 1e-6
        return (1.0 - flatness) * (harm_energy / total_energy)

    def _detect_active_regions(self, y_audio, threshold=0.04):
        """[核心优化] 识别能量活跃区间，剥离中间的无声/空白片段"""
        mse = librosa.feature.rms(y=y_audio, frame_length=2048, hop_length=512)[0]
        max_mse = np.max(mse) if np.max(mse) > 0 else 1.0
        active_mask = mse > (max_mse * threshold)
        
        regions = []
        is_active, start_f = False, 0
        for f, val in enumerate(active_mask):
            if val and not is_active:
                start_f, is_active = f, True
            elif not val and is_active:
                regions.append((start_f, f))
                is_active = False
        if is_active: regions.append((start_f, len(active_mask)))
        return [(librosa.frames_to_time(r[0], hop_length=512, sr=self.sr), 
                 librosa.frames_to_time(r[1], hop_length=512, sr=self.sr)) for r in regions]

    def find_musical_boundary(self, y_audio, onset_env, chroma, start_f, expected_p_f, is_melody=False):
        """寻找衔接最自然的切分点：打击乐寻找能量低谷，旋律寻找和声转换"""
        search_range = int(expected_p_f * 0.12)
        ideal_end = start_f + expected_p_f
        s_start, s_end = max(0, ideal_end - search_range), min(len(onset_env), ideal_end + search_range)
        
        if s_start >= s_end: return ideal_end

        if not is_melody:
            # 打击乐：利用能量低谷(Valley)防止底鼓泄露
            y_region = y_audio[s_start * 512 : s_end * 512]
            if len(y_region) == 0: return ideal_end
            energy = librosa.feature.rms(y=y_region, frame_length=1024, hop_length=1)[0]
            return s_start + np.argmin(energy)
        else:
            # 旋律：基于 Chroma 相似度确保无缝衔接
            ref_chroma = chroma[:, start_f % chroma.shape[1]]
            search_chromas = chroma[:, s_start:s_end]
            norm_ref = np.linalg.norm(ref_chroma) + 1e-6
            norm_search = np.linalg.norm(search_chromas, axis=0) + 1e-6
            sims = np.dot(ref_chroma, search_chromas) / (norm_ref * norm_search)
            return s_start + np.argmax(sims)

    def process_single_track(self, name, y_audio, beat_times, total_dur):
        """[核心逻辑] 识别活跃区间，并将长段落切分为 6s 以内的短 Loop"""
        is_melody = name in ['bass', 'instruments']
        active_regions = self._detect_active_regions(y_audio, threshold=0.04 if not is_melody else 0.06)
        
        markers, unique_samples = [], {}
        if not active_regions:
            markers.append({"Start Time (s)": 0.0, "End Time (s)": round(total_dur, 3), "Duration (s)": round(total_dur, 3), "Type": f"{name}_silence"})
            return markers, unique_samples

        onset_env = librosa.onset.onset_strength(y=y_audio, sr=self.sr)
        chroma = librosa.feature.chroma_stft(y=y_audio, sr=self.sr) if is_melody else None
        avg_beat_dur = np.mean(np.diff(beat_times))
        
        # 锁定循环步长：确保在 6 秒以内
        p_beats = 8
        if avg_beat_dur * p_beats > 6.0: p_beats = 4
        expected_p_f = int(librosa.time_to_frames(avg_beat_dur * p_beats, sr=self.sr))
        
        known_patterns, last_end_t = [], 0.0

        for r_start, r_end in active_regions:
            # 处理区间前的空白/静音
            if r_start - last_end_t > 0.5:
                markers.append({"Start Time (s)": round(last_end_t, 3), "End Time (s)": round(r_start, 3), 
                                "Duration (s)": round(r_start - last_end_t, 3), "Type": f"{name}_sparse_gap"})

            # 在活动区间内进行 Loop 划分
            curr_f = int(librosa.time_to_frames(r_start, sr=self.sr))
            reg_end_f = int(librosa.time_to_frames(r_end, sr=self.sr))
            
            while curr_f + expected_p_f < reg_end_f + int(expected_p_f * 0.5):
                next_f = self.find_musical_boundary(y_audio, onset_env, chroma, curr_f, expected_p_f, is_melody)
                st_t, en_t = librosa.frames_to_time(curr_f, sr=self.sr), librosa.frames_to_time(next_f, sr=self.sr)
                en_t = min(en_t, total_dur)
                
                y_seg = y_audio[int(st_t * self.sr):int(en_t * self.sr)].copy()
                if len(y_seg) > 500:
                    f_l = int(0.005 * self.sr)
                    y_seg[:f_l] *= np.linspace(0, 1, f_l); y_seg[-f_l:] *= np.linspace(1, 0, f_l)

                fp = np.mean(librosa.feature.mfcc(y=y_seg, sr=self.sr, n_mfcc=13), axis=1) if len(y_seg) > 1024 else np.zeros(13)
                matched_id = -1
                for pid, kfp in enumerate(known_patterns):
                    if np.linalg.norm(fp - kfp) < 14.0: matched_id = pid + 1; break
                
                if matched_id == -1:
                    known_patterns.append(fp); matched_id = len(known_patterns)
                    unique_samples[f"{name}_loop_{matched_id}"] = y_seg 

                markers.append({"Start Time (s)": round(st_t, 3), "End Time (s)": round(en_t, 3), 
                                "Duration (s)": round(en_t - st_t, 3), "Type": f"{name}_loop_{matched_id}"})
                curr_f = next_f
                if en_t >= r_end: break
            
            last_end_t = librosa.frames_to_time(curr_f, sr=self.sr)

        if last_end_t < total_dur:
            markers.append({"Start Time (s)": round(last_end_t, 3), "End Time (s)": round(total_dur, 3), 
                            "Duration (s)": round(total_dur - last_end_t, 3), "Type": f"{name}_outro"})
        return markers, unique_samples

    def find_sync_intervals(self, track_results, subset_keys, total_dur, is_global=False):
        """鲁棒同步划分：允许轨道处于 '被动区域' (Gap/Intro/Outro) 时对齐"""
        pts = [0.0, total_dur]
        for k in subset_keys:
            if k in track_results:
                for m in track_results[k]['markers']: pts.append(m['End Time (s)'])
        
        unique_pts = sorted(list(set([round(p, 3) for p in pts])))
        sync_blocks, last_pt = [], 0.0
        tol = 0.08 if not is_global else 0.12
        
        for pt in unique_pts:
            if pt <= last_pt + 1.0: continue
            
            is_valid = True
            for k in subset_keys:
                if k not in track_results: continue
                mks = track_results[k]['markers']
                at_boundary = any(abs(pt - m['End Time (s)']) < tol for m in mks)
                is_passive = any(m['Type'].endswith(('gap', 'intro', 'outro', 'silence')) and m['Start Time (s)'] <= pt <= m['End Time (s)'] for m in mks)
                if not (at_boundary or is_passive): is_valid = False; break
            
            if is_valid:
                sync_blocks.append({"Start Time (s)": last_pt, "End Time (s)": pt, 
                                   "Duration (s)": round(pt - last_pt, 3), "Type": "SYNC_BLOCK"})
                last_pt = pt
        return sync_blocks

    def process(self, y_drums, beat_times):
        """Simplified processor for app_drums.py specific needs"""
        results = {}
        if y_drums.ndim > 1: y_drums = np.mean(y_drums, axis=0)
        total_dur = len(y_drums) / self.sr
        
        parts = ['kick', 'snare', 'cymbals'] 
        
        for name in parts:
            if name == 'kick': 
                sos = scipy.signal.butter(4, 150, 'low', fs=self.sr, output='sos')
                internal_name = 'kick'
            elif name == 'snare': 
                sos = scipy.signal.butter(4, [200, 3500], 'band', fs=self.sr, output='sos')
                internal_name = 'snare_perc'
            else: 
                sos = scipy.signal.butter(4, 3500, 'high', fs=self.sr, output='sos')
                internal_name = 'cymbals'
                
            y_f = scipy.signal.sosfilt(sos, y_drums)
            
            m, s = self.process_single_track(internal_name, y_f, beat_times, total_dur)
            
            best_loop = None
            loop_durations = {}
            first_occurrence = {}
            
            for marker in m:
                t_type = marker['Type']
                if 'loop' in t_type:
                    dur = marker['Duration (s)']
                    loop_durations[t_type] = loop_durations.get(t_type, 0) + dur
                    if t_type not in first_occurrence:
                        first_occurrence[t_type] = marker
            
            if loop_durations:
                best_type = max(loop_durations, key=loop_durations.get)
                marker = first_occurrence[best_type]
                best_loop = {
                    'start': marker['Start Time (s)'],
                    'end': marker['End Time (s)'],
                    'duration': marker['Duration (s)'],
                    'confidence': min(1.0, loop_durations[best_type] / total_dur * 2)
                }
            
            results[name] = {
                'audio': y_f,
                'loop': best_loop
            }
            
        return results

    def process_all_tracks(self, stem_audio_dict, beat_times):
        results = {}
        y_drums = stem_audio_dict.get('drums', np.zeros(1024))
        if y_drums.ndim > 1: y_drums = np.mean(y_drums, axis=0)
        total_dur = len(y_drums) / self.sr
        
        track_list = ['kick', 'snare_perc', 'cymbals', 'bass', 'instruments']
        for part in track_list:
            if part in ['kick', 'snare_perc', 'cymbals']:
                low, high = self.bands[part]
                if part == 'kick': sos = scipy.signal.butter(4, 150, 'low', fs=self.sr, output='sos')
                elif part == 'snare_perc': sos = scipy.signal.butter(4, [200, 4000], 'band', fs=self.sr, output='sos')
                else: sos = scipy.signal.butter(4, 4000, 'high', fs=self.sr, output='sos')
                y_f = scipy.signal.sosfilt(sos, y_drums)
                m, s = self.process_single_track(part, y_f, beat_times, total_dur)
                results[part] = {"markers": m, "audio": y_f, "samples": s}
            elif part in stem_audio_dict:
                y_t = stem_audio_dict[part]
                if y_t.ndim > 1: y_t = np.mean(y_t, axis=0)
                m, s = self.process_single_track(part, y_t, beat_times, total_dur)
                results[part] = {"markers": m, "audio": y_t, "samples": s}

        results['drum_sync'] = self.find_sync_intervals(results, ['kick', 'snare_perc', 'cymbals'], total_dur)
        results['global_sync'] = self.find_sync_intervals(results, ['kick', 'snare_perc', 'cymbals', 'bass', 'instruments'], total_dur, is_global=True)
        return results
