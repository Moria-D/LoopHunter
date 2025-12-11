import librosa
import numpy as np
import warnings
import random
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

class AudioRemixer:
    def __init__(self, audio_path, sr=22050):
        self.path = audio_path
        self.sr = sr
        print(f"Loading audio: {audio_path}...")
        self.y, self.sr = librosa.load(audio_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        self.beat_times = None
        self.beat_features = None 
        self.loops = []
        
        # 计算 Onset Envelope 用于瞬态检测
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)

    def _refine_cut_point(self, time_sec, search_window_ms=50):
        """
        [关键优化] 微调切点：瞬态回溯 + 过零点锁定
        """
        center_sample = int(time_sec * self.sr)
        search_samples = int((search_window_ms / 1000) * self.sr)
        
        start_search = max(0, center_sample - search_samples)
        end_search = min(len(self.y), center_sample + search_samples)
        
        if start_search >= end_search: return center_sample
        
        # 瞬态检测
        onset_frame_center = librosa.samples_to_frames(center_sample)
        search_frame_rad = librosa.samples_to_frames(search_samples)
        
        f_start = max(0, onset_frame_center - search_frame_rad)
        f_end = min(len(self.onset_env), onset_frame_center + search_frame_rad)
        
        if f_start < f_end:
            local_onset_idx = f_start + np.argmax(self.onset_env[f_start:f_end])
            transient_sample = librosa.frames_to_samples(local_onset_idx)
            # 往回倒推 10ms，保留 Attack
            pre_roll = int(0.01 * self.sr) 
            target_sample = max(0, transient_sample - pre_roll)
        else:
            target_sample = center_sample

        # 过零点锁定
        zc_window = 200
        z_start = max(0, target_sample - zc_window)
        z_end = min(len(self.y), target_sample + zc_window)
        
        chunk = self.y[z_start:z_end]
        zero_crossings = np.where(np.diff(np.signbit(chunk)))[0]
        
        if len(zero_crossings) > 0:
            best_zc = zero_crossings[np.argmin(np.abs(zero_crossings - (target_sample - z_start)))]
            return z_start + best_zc
        
        return target_sample

    def analyze(self):
        """分析音频结构"""
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=self.sr)
        
        if len(beat_frames) < 16:
            beat_frames = np.linspace(0, len(self.y), int(self.duration * 2), dtype=int)
            
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        # 特征提取
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y_percussive, sr=self.sr, n_mfcc=13)
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        mfcc_sync = librosa.util.sync(mfcc, beat_frames, aggregate=np.median)
        chroma_sync = librosa.util.normalize(chroma_sync, axis=1)
        mfcc_sync = librosa.util.normalize(mfcc_sync, axis=1)
        features = np.vstack([chroma_sync, mfcc_sync])
        
        # [修复] 保存特征用于缩短算法
        self.beat_features = features.T
        
        # Shingling & Recurrence
        stack_size = 4 
        features_stacked = librosa.feature.stack_memory(features, n_steps=stack_size, delay=1)
        R = librosa.segment.recurrence_matrix(features_stacked, width=3, mode='affinity', sym=True)
        
        self.loops = []
        n_beats = R.shape[0]
        thresholds = [0.85, 0.75, 0.65, 0.55] 
        
        for thresh in thresholds:
            if len(self.loops) > 40: break 
            for lag in range(8, n_beats // 2):
                diag = np.diagonal(R, offset=lag)
                diag_smooth = median_filter(diag, size=3)
                high_sim_indices = np.where(diag_smooth > thresh)[0]
                if len(high_sim_indices) == 0: continue
                segments = np.split(high_sim_indices, np.where(np.diff(high_sim_indices) != 1)[0] + 1)
                for seg in segments:
                    if len(seg) >= 8:
                        start_beat = seg[0]
                        time_start = self.beat_times[start_beat]
                        time_end = self.beat_times[start_beat + lag]
                        
                        if time_end > self.duration - 0.5: continue
                        
                        is_dup = False
                        for l in self.loops:
                            if abs(l['start'] - time_start) < 1.0 and abs(l['end'] - time_end) < 1.0:
                                is_dup = True; break
                        if not is_dup:
                            self.loops.append({
                                "start": time_start,
                                "end": time_end,
                                "duration": time_end - time_start,
                                "score": np.mean(diag[seg]),
                                "beats_len": lag,
                                "type": "loop"
                            })
        self.loops = sorted(self.loops, key=lambda x: x['score'], reverse=True)

    def generate_loop_preview(self, loop_data, repetitions=4):
        """生成预览音频"""
        s_idx = self._refine_cut_point(loop_data['start'])
        e_idx = self._refine_cut_point(loop_data['end'])
        
        chunk = self.y[s_idx:e_idx]
        if len(chunk) == 0: return np.array([])

        output = chunk
        fade_len = int(0.02 * self.sr)
        
        for _ in range(repetitions - 1):
            if len(chunk) < fade_len:
                output = np.concatenate((output, chunk))
            else:
                prev_tail = output[-fade_len:]
                curr_head = chunk[:fade_len]
                lin = np.linspace(0, 1, fade_len)
                w_in = np.sin(lin * np.pi / 2)
                w_out = np.cos(lin * np.pi / 2)
                overlap = prev_tail * w_out + curr_head * w_in
                output = np.concatenate((output[:-fade_len], overlap, chunk[fade_len:]))
                
        return output

    def _filter_loops(self):
        if not self.loops: return []
        sorted_loops = sorted(self.loops, key=lambda x: x['score'], reverse=True)
        selected_loops = []
        for candidate in sorted_loops:
            is_overlap = False
            for selected in selected_loops:
                start_max = max(candidate['start'], selected['start'])
                end_min = min(candidate['end'], selected['end'])
                if end_min > start_max:
                    overlap_len = end_min - start_max
                    min_len = min(candidate['duration'], selected['duration'])
                    if overlap_len > 0.3 * min_len:
                        is_overlap = True; break
            if not is_overlap:
                candidate['repeats'] = 0 
                selected_loops.append(candidate)
        return sorted(selected_loops, key=lambda x: x['start'])

    def _find_best_cut_points(self, target_duration):
        """
        [修复] 找回缩短逻辑：弹性搜索最佳缝合点
        """
        n_beats = len(self.beat_times)
        remove_amount = self.duration - target_duration
        
        best_score = -999.0
        best_cut = None 
        
        min_idx = 4
        max_idx = n_beats - 4
        
        # 听感优先配置
        time_tolerance = 5.0 if target_duration <= 30 else 3.0
        time_penalty_weight = 0.02 
        
        for i in range(min_idx, max_idx):
            time_a = self.beat_times[i]
            
            ideal_time_b = time_a + remove_amount
            if ideal_time_b >= self.beat_times[max_idx]: break
                
            j_approx = np.searchsorted(self.beat_times, ideal_time_b)
            search_window_beats = int(time_tolerance * 2)
            
            start_j = max(i + 8, j_approx - search_window_beats)
            end_j = min(n_beats - 4, j_approx + search_window_beats)
            
            if start_j >= end_j: continue
            
            feat_a = self.beat_features[i].reshape(1, -1)
            feats_candidates = self.beat_features[start_j:end_j]
            
            dists = cdist(feat_a, feats_candidates, metric='cosine')[0]
            sims = 1.0 - dists
            
            candidate_times = self.beat_times[start_j:end_j]
            est_durations = time_a + (self.duration - candidate_times)
            time_errors = np.abs(est_durations - target_duration)
            
            final_scores = sims - (time_errors * time_penalty_weight)
            
            local_best_idx = np.argmax(final_scores)
            local_best_score = final_scores[local_best_idx]
            
            if local_best_score > best_score:
                best_score = local_best_score
                real_j = start_j + local_best_idx
                best_cut = (i, real_j)
        
        return best_cut, best_score

    def _plan_hard_cut(self, target):
        """[修复] 找回硬切保底逻辑"""
        cut = target / 2
        return [
            {"source_start":0, "source_end":cut, "duration":cut, "type":"Head", "remix_start":0, "refine_end": True},
            {"source_start":self.duration-(target-cut), "source_end":self.duration, "duration":target-cut, "type":"Tail", "xfade":50, "is_jump":True, "remix_start":cut, "refine_start": True}
        ]

    def plan_multi_loop_remix(self, target_duration):
        """
        高级路径规划：支持延长 (Loop Back) 和 缩短 (Skip Forward)
        """
        # ==========================================
        # 1. 缩短模式 (Target < Original) - [修复] 找回丢失的分支
        # ==========================================
        if target_duration < self.duration:
            if target_duration < 5.0:
                return self._plan_hard_cut(target_duration), target_duration

            cut_indices, score = self._find_best_cut_points(target_duration)
            
            timeline = []
            if cut_indices:
                idx_a, idx_b = cut_indices
                time_a = self.beat_times[idx_a]
                time_b = self.beat_times[idx_b]
                
                # Head (0 -> A)
                timeline.append({
                    "source_start": 0.0,
                    "source_end": time_a,
                    "duration": time_a,
                    "type": "Head",
                    "xfade": 0,
                    "remix_start": 0.0,
                    "refine_end": True # 开启微调
                })
                
                # Tail (B -> End)
                tail_dur = self.duration - time_b
                timeline.append({
                    "source_start": time_b,
                    "source_end": self.duration,
                    "duration": tail_dur,
                    "type": "Tail",
                    "xfade": 30, # 缝合点 Fade
                    "remix_start": time_a,
                    "is_jump": True,
                    "refine_start": True # 开启微调
                })
                
                actual_dur = time_a + tail_dur
                return timeline, actual_dur
            else:
                return self._plan_hard_cut(target_duration), target_duration

        # ==========================================
        # 2. 延长模式 (Target > Original)
        # ==========================================
        else:
            if not self.loops: return None, 0
            
            active_loops = self._filter_loops()
            if not active_loops:
                active_loops = [self.loops[0]]
                active_loops[0]['repeats'] = 0

            current_total = self.duration
            time_diff = target_duration - current_total
            
            if time_diff > 0:
                while time_diff > 0:
                    best_idx = -1
                    best_score = -1
                    for i, loop in enumerate(active_loops):
                        penalty = 1.0 / (loop['repeats'] + 1)
                        w_score = loop['score'] * penalty * loop['duration']
                        if w_score > best_score:
                            best_score = w_score; best_idx = i
                    if best_idx != -1:
                        active_loops[best_idx]['repeats'] += 1
                        time_diff -= active_loops[best_idx]['duration']
                    else: break

            timeline = []
            cursor = 0.0      
            source_cursor = 0.0
            
            for loop in active_loops:
                # A. Linear part
                if source_cursor < loop['end']:
                    seg_dur = loop['end'] - source_cursor
                    timeline.append({
                        "source_start": source_cursor,
                        "source_end": loop['end'],
                        "duration": seg_dur,
                        "type": "Linear",
                        "xfade": 0,
                        "remix_start": cursor,
                        "refine_start": True if len(timeline)>0 else False,
                        "refine_end": True 
                    })
                    cursor += seg_dur
                    source_cursor = loop['end']
                
                # B. Loop Extension
                if loop['repeats'] > 0:
                    for i in range(loop['repeats']):
                        timeline.append({
                            "source_start": loop['start'],
                            "source_end": loop['end'],
                            "duration": loop['duration'],
                            "type": "Loop Extension",
                            "xfade": 25,
                            "remix_start": cursor,
                            "loop_id": loop.get('start'),
                            "refine_start": True,
                            "refine_end": True
                        })
                        cursor += loop['duration']
            
            # C. Outro
            if source_cursor < self.duration:
                timeline.append({
                    "source_start": source_cursor,
                    "source_end": self.duration,
                    "duration": self.duration - source_cursor,
                    "type": "Outro",
                    "xfade": 0,
                    "remix_start": cursor,
                    "refine_start": True,
                    "refine_end": False
                })
                cursor += (self.duration - source_cursor)
                
            return timeline, cursor

    def render_remix(self, timeline):
        if not timeline: return np.array([])
        
        total_samples = int(sum([s['duration'] for s in timeline]) * self.sr)
        output = np.zeros(total_samples + 44100) # Buffer
        cursor = 0
        
        for i, seg in enumerate(timeline):
            s_time = seg['source_start']
            e_time = seg['source_end']
            
            if seg.get('refine_start', False):
                s_idx = self._refine_cut_point(s_time)
            else:
                s_idx = int(s_time * self.sr)
                
            if seg.get('refine_end', False):
                e_idx = self._refine_cut_point(e_time)
            else:
                e_idx = int(e_time * self.sr)
                
            if s_idx >= e_idx: continue
            if e_idx > len(self.y): e_idx = len(self.y)
            
            chunk = self.y[s_idx:e_idx]
            
            fade_ms = seg.get('xfade', 0)
            fade_len = int((fade_ms / 1000) * self.sr)
            
            if i == 0 or fade_len == 0:
                output[cursor:cursor+len(chunk)] = chunk
                cursor += len(chunk)
            else:
                overlap = cursor - fade_len
                if overlap < 0: overlap = 0
                prev = output[overlap:cursor]
                curr = chunk[:fade_len]
                n = min(len(prev), len(curr))
                
                if n > 1:
                    lin = np.linspace(0, 1, n)
                    w_in = np.sin(lin * np.pi / 2)
                    w_out = np.cos(lin * np.pi / 2)
                    mix = prev*w_out + curr*w_in
                    output[overlap:overlap+n] = mix
                
                output[cursor:cursor+len(chunk)-fade_len] = chunk[fade_len:]
                cursor += len(chunk) - fade_len
                
        return output[:cursor]