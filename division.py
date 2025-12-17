import librosa
import numpy as np
import warnings
import random
import json
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
        
        # 计算全局特征用于相对比较 (初始计算)
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.global_rms = np.sqrt(np.mean(self.y**2))

    def _trim_silence(self):
        """
        [保留功能] 预处理：切除尾部静音
        """
        # 1. 计算短时能量
        mse = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]
        db = librosa.amplitude_to_db(mse, ref=np.max)
        
        # 2. 设定阈值
        silence_thresh = -60
        
        # 3. 从后往前扫描
        last_frame = len(db) - 1
        for i in range(len(db)-1, 0, -1):
            if db[i] > silence_thresh:
                last_frame = i
                break
        
        # 4. 转换并预留 0.5s 混响余量
        last_sample = librosa.frames_to_samples(last_frame, hop_length=512)
        padding = int(0.5 * self.sr) 
        trim_end = min(len(self.y), last_sample + padding)
        
        # 5. 应用切除
        if trim_end < len(self.y) - int(0.1 * self.sr):
            print(f"Trimming silence: {self.duration:.2f}s -> {trim_end/self.sr:.2f}s")
            self.y = self.y[:trim_end]
            self.duration = trim_end / self.sr
            
            # 更新全局特征以保持状态一致
            self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            self.global_rms = np.sqrt(np.mean(self.y**2))

    def _refine_cut_point(self, time_sec, search_window_ms=50):
        """[保留功能] 微调切点：瞬态回溯 + 过零点锁定"""
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

    def _get_event_durations(self, onsets, envelope, sr, max_dur=2.0, min_dur=0.1):
        """
        [新增] 根据能量包络计算每个事件的持续时间
        """
        durations = []
        n_frames = len(envelope)
        onset_frames = librosa.time_to_frames(onsets, sr=sr)
        
        for i, start_frame in enumerate(onset_frames):
            # 确定硬性边界（下一个音符开始或最大时长）
            if i < len(onset_frames) - 1:
                hard_limit = onset_frames[i+1]
            else:
                hard_limit = min(n_frames, start_frame + librosa.time_to_frames(max_dur, sr=sr))
                
            # 限制最大搜索范围
            search_limit = min(n_frames, start_frame + librosa.time_to_frames(max_dur, sr=sr))
            actual_limit = int(min(hard_limit, search_limit))
            
            if start_frame >= n_frames:
                durations.append(min_dur)
                continue

            # 找到局部峰值（onset 触发后能量可能会继续上升一点）
            lookahead = 5 # frames
            local_search_end = min(n_frames, start_frame + lookahead)
            if local_search_end > start_frame:
                peak_offset = np.argmax(envelope[start_frame:local_search_end])
                peak_idx = start_frame + peak_offset
                peak_val = envelope[peak_idx]
            else:
                peak_idx = start_frame
                peak_val = envelope[start_frame]
            
            # 寻找能量衰减点 (阈值法)
            threshold = peak_val * 0.25 # 能量降至 25% 视为结束
            end_frame = actual_limit
            
            # 从峰值开始向后搜索
            for f in range(peak_idx, actual_limit):
                if envelope[f] < threshold:
                    end_frame = f
                    break
            
            # 计算时长
            dur_frames = end_frame - start_frame
            dur_time = librosa.frames_to_time(dur_frames, sr=sr)
            
            # 确保在合理范围内
            dur_time = max(min_dur, min(dur_time, max_dur))
            durations.append(dur_time)
            
        return durations

    def _classify_segment(self, start_time, end_time):
        """[保留功能] 基于音频特征动态分类 Loop 类型"""
        s_idx = int(start_time * self.sr)
        e_idx = int(end_time * self.sr)
        chunk = self.y[s_idx:e_idx]
        
        if len(chunk) == 0: return "melody"
        
        chunk_rms = np.sqrt(np.mean(chunk**2))
        energy_ratio = chunk_rms / (self.global_rms + 1e-6)
        zcr = np.mean(librosa.feature.zero_crossing_rate(chunk))
        
        if energy_ratio > 1.1: return "climax"
        if energy_ratio < 0.6: return "atmosphere"
        if zcr > 0.08: return "beats"
        return "melody"

    def analyze(self):
        """分析音频结构"""
        
        # 1. [保留] 先执行静音切除
        self._trim_silence()
        
        # ========================================================
        # 2. [新增完善] 信号预处理：去直流 + 归一化
        # ========================================================
        
        # 消除直流偏移 (DC Offset Removal)
        # 作用：让波形中心严格对准 0，确保 _refine_cut_point 的过零点检测极其精准，消除爆音。
        self.y = self.y - np.mean(self.y)
        
        # 峰值归一化 (Peak Normalization)
        # 作用：将最大音量拉到 1.0。确保 _classify_segment 中的能量阈值(1.1/0.6)对所有音量的歌曲都有效。
        if np.max(np.abs(self.y)) > 0:
            self.y = self.y / np.max(np.abs(self.y))
            
        # [同步更新] 因为波形数值变了，必须更新全局特征，否则分类器会失效
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.global_rms = np.sqrt(np.mean(self.y**2))
        # ========================================================
        
        # 3. 后续常规分析 (保持不变)
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=self.sr)
        
        if len(beat_frames) < 16:
            beat_frames = np.linspace(0, len(self.y), int(self.duration * 2), dtype=int)
            
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y_percussive, sr=self.sr, n_mfcc=13)
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        mfcc_sync = librosa.util.sync(mfcc, beat_frames, aggregate=np.median)
        chroma_sync = librosa.util.normalize(chroma_sync, axis=1)
        mfcc_sync = librosa.util.normalize(mfcc_sync, axis=1)
        features = np.vstack([chroma_sync, mfcc_sync])
        
        # [重要] 保存特征用于缩短算法的全网格搜索
        self.beat_features = features.T
        
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
                            # 调用分类器
                            loop_type = self._classify_segment(time_start, time_end)
                            self.loops.append({
                                "start": time_start,
                                "end": time_end,
                                "duration": time_end - time_start,
                                "score": np.mean(diag[seg]),
                                "beats_len": lag,
                                "type": loop_type
                            })
        self.loops = sorted(self.loops, key=lambda x: x['score'], reverse=True)

    def analyze_stems(self):
        """
        Analyze independent stems (Percussive/Harmonic) to find instrument events.
        Simplified approach without deep learning models (Demucs/Spleeter).
        """
        # 1. Separate Harmonic and Percussive
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        
        events = {
            "drums": [],
            "bass": [],
            "melody": []
        }
        
        # --- A. Drums (Percussive Component) ---
        # Use simple onset detection on percussive component
        onset_env_perc = librosa.onset.onset_strength(y=y_percussive, sr=self.sr)
        onsets_perc = librosa.onset.onset_detect(onset_envelope=onset_env_perc, sr=self.sr, units='time', backtrack=True)
        
        if len(onsets_perc) > 0:
            # Drums duration is usually short
            durs_perc = self._get_event_durations(onsets_perc, onset_env_perc, self.sr, max_dur=0.4, min_dur=0.05)
            for t, d in zip(onsets_perc, durs_perc):
                events["drums"].append({
                    "start": float(t),
                    "duration": float(d)
                })

        # --- B. Bass (Low Freq Harmonic) ---
        # Low-pass filter harmonic component to isolate bass (< 250 Hz)
        # Using a simple spectral cut for efficiency
        S_h = librosa.stft(y_harmonic)
        freqs = librosa.fft_frequencies(sr=self.sr)
        bass_mask = freqs < 250
        S_bass = S_h * bass_mask[:, np.newaxis]
        y_bass = librosa.istft(S_bass)
        
        onset_env_bass = librosa.onset.onset_strength(y=y_bass, sr=self.sr)
        onsets_bass = librosa.onset.onset_detect(onset_envelope=onset_env_bass, sr=self.sr, units='time', backtrack=False)
        
        if len(onsets_bass) > 0:
            durs_bass = self._get_event_durations(onsets_bass, onset_env_bass, self.sr, max_dur=2.0)
            for t, d in zip(onsets_bass, durs_bass):
                events["bass"].append({
                    "start": float(t),
                    "duration": float(d)
                })

        # --- C. Melody/Other (High Freq Harmonic) ---
        # High-pass filter (> 250 Hz)
        melody_mask = freqs >= 250
        S_melody = S_h * melody_mask[:, np.newaxis]
        y_melody = librosa.istft(S_melody)
        
        onset_env_mel = librosa.onset.onset_strength(y=y_melody, sr=self.sr)
        onsets_mel = librosa.onset.onset_detect(onset_envelope=onset_env_mel, sr=self.sr, units='time', backtrack=False)
        
        if len(onsets_mel) > 0:
            durs_mel = self._get_event_durations(onsets_mel, onset_env_mel, self.sr, max_dur=4.0)
            for t, d in zip(onsets_mel, durs_mel):
                events["melody"].append({
                    "start": float(t),
                    "duration": float(d)
                })
                
        return events

    def export_analysis_data(self, source_filename="audio.wav"):
        """[保留功能] 生成分析报告文档"""
        looping_points = []
        for loop in self.loops:
            looping_points.append({
                "duration": round(loop['duration'], 2),
                "start_position": round(loop['start'], 2),
                "type": loop['type'], 
                "confidence": round(loop['score'], 2)
            })
        raw_data = {
            "source_music": source_filename,
            "total_duration": round(self.duration, 2),
            "looping_points": looping_points
        }
        lines = []
        lines.append(f"========== AUDIO ANALYSIS REPORT ==========")
        lines.append(f"Source File : {source_filename}")
        lines.append(f"Duration    : {self.duration:.2f} seconds")
        lines.append(f"Loops Found : {len(self.loops)}")
        lines.append("")
        lines.append("--- DETAILED LOOP POINTS ---")
        lines.append(f"{'#':<4} | {'Start':<9} | {'Dur':<8} | {'Type':<10} | {'Conf':<6}")
        lines.append("-" * 50)
        for i, pt in enumerate(looping_points):
            lines.append(f"#{i+1:02d}  | {pt['start_position']:6.2f}s  | {pt['duration']:5.2f}s  | {pt['type'].upper():<10} | {pt['confidence']:.2f}")
        lines.append("")
        lines.append("========== END OF REPORT ==========")
        return raw_data, "\n".join(lines)

    def generate_loop_preview(self, loop_data, repetitions=4):
        """[保留功能] 生成预览音频"""
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
        """[保留功能] 弹性搜索最佳缝合点"""
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
        cut = target / 2
        return [
            {"source_start":0, "source_end":cut, "duration":cut, "type":"Head", "remix_start":0, "refine_end": True},
            {"source_start":self.duration-(target-cut), "source_end":self.duration, "duration":target-cut, "type":"Tail", "xfade":50, "is_jump":True, "remix_start":cut, "refine_start": True}
        ]

    def plan_multi_loop_remix(self, target_duration):
        """
        [保留功能] 高级路径规划：支持延长 (Loop Back) 和 缩短 (Skip Forward)
        """
        
        # === A. 缩短模式 (Target < Original) ===
        if target_duration < self.duration:
            if target_duration < 5.0:
                return self._plan_hard_cut(target_duration), target_duration

            # 弹性搜索最佳切点
            cut_indices, score = self._find_best_cut_points(target_duration)
            
            timeline = []
            if cut_indices:
                idx_a, idx_b = cut_indices
                time_a = self.beat_times[idx_a]
                time_b = self.beat_times[idx_b]
                
                # Head: 0 -> A
                timeline.append({
                    "source_start": 0.0,
                    "source_end": time_a,
                    "duration": time_a,
                    "type": "Head",
                    "xfade": 0,
                    "remix_start": 0.0,
                    "refine_end": True
                })
                
                # Tail: B -> End
                tail_dur = self.duration - time_b
                timeline.append({
                    "source_start": time_b,
                    "source_end": self.duration,
                    "duration": tail_dur,
                    "type": "Tail",
                    "xfade": 30, # 跳跃点 Crossfade
                    "remix_start": time_a,
                    "is_jump": True,
                    "refine_start": True
                })
                
                actual_dur = time_a + tail_dur
                return timeline, actual_dur
            else:
                return self._plan_hard_cut(target_duration), target_duration

        # === B. 延长模式 (Target > Original) ===
        else:
            if not self.loops:
                return [{"source_start":0, "source_end":self.duration, "duration":self.duration, "type":"Original", "remix_start":0}], self.duration
            
            active_loops = self._filter_loops()
            if not active_loops:
                active_loops = [self.loops[0]]
                active_loops[0]['repeats'] = 0
            
            current_total = self.duration
            time_diff = target_duration - current_total
            
            if time_diff > 0:
                while time_diff > 0:
                    best_idx = -1; best_score = -1
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
                # 线性部分
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
                
                # 循环部分
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
            
            # 尾部
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
        output = np.zeros(total_samples + 44100)
        cursor = 0
        for i, seg in enumerate(timeline):
            s_time = seg['source_start']
            e_time = seg['source_end']
            
            if seg.get('refine_start', False): s_idx = self._refine_cut_point(s_time)
            else: s_idx = int(s_time * self.sr)
            
            if seg.get('refine_end', False): e_idx = self._refine_cut_point(e_time)
            else: e_idx = int(e_time * self.sr)
            
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