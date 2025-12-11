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

    def analyze(self):
        """分析 Beat 结构并保存特征"""
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=self.sr)
        
        if len(beat_frames) < 16:
            beat_frames = np.linspace(0, len(self.y), int(self.duration * 2), dtype=int)
            
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        # 1. 提取特征
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y_percussive, sr=self.sr, n_mfcc=13)
        
        # 2. 同步到 Beat
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        mfcc_sync = librosa.util.sync(mfcc, beat_frames, aggregate=np.median)
        
        chroma_sync = librosa.util.normalize(chroma_sync, axis=1)
        mfcc_sync = librosa.util.normalize(mfcc_sync, axis=1)
        
        # 保存特征
        self.beat_features = np.vstack([chroma_sync, mfcc_sync]).T
        
        # 3. 寻找 Loops (仅用于延长模式)
        stack_size = 4 
        features = np.vstack([chroma_sync, mfcc_sync])
        features_stacked = librosa.feature.stack_memory(features, n_steps=stack_size, delay=1)
        R = librosa.segment.recurrence_matrix(features_stacked, width=3, mode='affinity', sym=True)
        
        self.loops = []
        n_beats = R.shape[0]
        thresholds = [0.85, 0.75, 0.65, 0.55] 
        
        for thresh in thresholds:
            if len(self.loops) > 30: break 
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
                        
                        self.loops.append({
                            "start": time_start,
                            "end": time_end,
                            "duration": time_end - time_start,
                            "score": np.mean(diag[seg]),
                            "beats_len": lag,
                            "type": "loop"
                        })
        
        self.loops = sorted(self.loops, key=lambda x: x['score'], reverse=True)

    def _find_best_cut_points(self, target_duration):
        """
        弹性搜索：寻找最佳缝合点 (A -> B)
        策略：Score = Similarity * 1.0 - Time_Error * Penalty
        允许用时长换音质
        """
        n_beats = len(self.beat_times)
        remove_amount = self.duration - target_duration
        
        best_score = -999.0
        best_cut = None 
        
        # 保护区间：头尾至少保留 4 beats (约2秒)
        min_idx = 4
        max_idx = n_beats - 4
        
        # 设定误差容忍度：如果是极短目标(<=30s)，容忍度大一点(5s)；否则小一点(3s)
        time_tolerance = 5.0 if target_duration <= 30 else 3.0
        
        # 惩罚系数：每偏差1秒，扣除多少相似度分数 (0.02 = 2%)
        # 如果用户更看重音质，这个值越小越好
        time_penalty_weight = 0.02 
        
        for i in range(min_idx, max_idx):
            time_a = self.beat_times[i]
            
            # 理想切入点
            ideal_time_b = time_a + remove_amount
            if ideal_time_b >= self.beat_times[max_idx]: break
                
            j_approx = np.searchsorted(self.beat_times, ideal_time_b)
            
            # 搜索窗口：在理想点前后找相似度最高的 beat
            # 窗口大小根据 Tolerance 动态决定
            search_window_beats = int(time_tolerance * 2) # 估算 beats 数
            
            start_j = max(i + 8, j_approx - search_window_beats)
            end_j = min(n_beats - 4, j_approx + search_window_beats)
            
            if start_j >= end_j: continue
            
            # 批量计算相似度
            feat_a = self.beat_features[i].reshape(1, -1)
            feats_candidates = self.beat_features[start_j:end_j]
            
            # 1. 相似度分数 (0-1)
            dists = cdist(feat_a, feats_candidates, metric='cosine')[0]
            sims = 1.0 - dists
            
            # 2. 时长误差分数
            candidate_times = self.beat_times[start_j:end_j]
            est_durations = time_a + (self.duration - candidate_times)
            time_errors = np.abs(est_durations - target_duration)
            
            # 3. 综合评分 = 相似度 - (误差 * 惩罚)
            final_scores = sims - (time_errors * time_penalty_weight)
            
            # 找到局部最优
            local_best_idx = np.argmax(final_scores)
            local_best_score = final_scores[local_best_idx]
            
            if local_best_score > best_score:
                best_score = local_best_score
                real_j = start_j + local_best_idx
                best_cut = (i, real_j)
        
        return best_cut, best_score

    def plan_multi_loop_remix(self, target_duration):
        # 1. 缩短模式
        if target_duration < self.duration:
            # 极端情况检查
            if target_duration < 5.0:
                return self._plan_hard_cut(target_duration), target_duration

            # 弹性搜索
            cut_indices, score = self._find_best_cut_points(target_duration)
            
            timeline = []
            if cut_indices:
                idx_a, idx_b = cut_indices
                time_a = self.beat_times[idx_a]
                time_b = self.beat_times[idx_b]
                
                # Part 1: Head
                timeline.append({
                    "source_start": 0.0, "source_end": time_a, "duration": time_a,
                    "type": "Head", "xfade": 0, "remix_start": 0.0
                })
                
                # Part 2: Tail
                tail_dur = self.duration - time_b
                timeline.append({
                    "source_start": time_b, "source_end": self.duration, "duration": tail_dur,
                    "type": "Tail", "xfade": 30, "remix_start": time_a, "is_jump": True
                })
                
                actual_dur = time_a + tail_dur
                return timeline, actual_dur
            else:
                return self._plan_hard_cut(target_duration), target_duration

        # 2. 延长模式 (逻辑不变)
        else:
            if not self.loops:
                return [{"source_start":0, "source_end":self.duration, "duration":self.duration, "type":"Original", "remix_start":0}], self.duration

            best_loop = self.loops[0]
            loop_dur = best_loop['duration']
            head_dur = best_loop['start']
            tail_dur = self.duration - best_loop['end']
            
            needed = target_duration - (head_dur + tail_dur)
            if needed <= 0: repeats = 1
            else:
                repeats = int(round(needed / loop_dur))
                if repeats < 1: repeats = 1
            
            timeline = []
            cursor = 0.0
            
            timeline.append({"source_start":0.0, "source_end":best_loop['end'], "duration":best_loop['end'], "type":"Head", "xfade":0, "remix_start":0.0})
            cursor += best_loop['end']
            
            for i in range(repeats):
                timeline.append({
                    "source_start":best_loop['start'], "source_end":best_loop['end'], "duration":loop_dur,
                    "type":"Loop Extension", "xfade":30, "is_jump":True, "remix_start":cursor
                })
                cursor += loop_dur
                
            timeline.append({
                "source_start":best_loop['end'], "source_end":self.duration, "duration":tail,
                "type":"Tail", "xfade":0, "remix_start":cursor
            })
            
            return timeline, cursor + tail

    def _plan_hard_cut(self, target):
        """极端保底：根据 target 比例保留头尾"""
        # 优先保留更多的 Tail (Outro)，因为结尾突兀比开头突兀更难受
        tail_ratio = 0.6 
        head_ratio = 0.4
        
        head_dur = target * head_ratio
        tail_dur = target * tail_ratio
        
        # 寻找最近的 Beat 进行切割，稍微保证一点节奏感
        if self.beat_times is not None:
            idx_head = np.argmin(np.abs(self.beat_times - head_dur))
            head_dur = self.beat_times[idx_head]
            
            target_tail_start = self.duration - tail_dur
            idx_tail = np.argmin(np.abs(self.beat_times - target_tail_start))
            # 确保不重叠
            if idx_tail <= idx_head: idx_tail = idx_head + 1
            if idx_tail < len(self.beat_times):
                tail_start = self.beat_times[idx_tail]
                tail_dur = self.duration - tail_start
        else:
            tail_start = self.duration - tail_dur

        return [
            {"source_start":0, "source_end":head_dur, "duration":head_dur, "type":"Head", "remix_start":0},
            {"source_start":tail_start, "source_end":self.duration, "duration":tail_dur, "type":"Tail", "xfade":50, "is_jump":True, "remix_start":head_dur}
        ]

    def render_remix(self, timeline):
        if not timeline: return np.array([])
        total_samples = int(sum([s['duration'] for s in timeline]) * self.sr)
        output = np.zeros(total_samples + 44100)
        cursor = 0
        
        for i, seg in enumerate(timeline):
            s = int(seg['source_start'] * self.sr)
            e = int(seg['source_end'] * self.sr)
            if s >= e: continue
            if e > len(self.y): e = len(self.y)
            
            chunk = self.y[s:e]
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
                if n > 0:
                    lin = np.linspace(0, 1, n)
                    mix = prev*(1-lin) + curr*lin
                    output[overlap:overlap+n] = mix
                output[cursor:cursor+len(chunk)-fade_len] = chunk[fade_len:]
                cursor += len(chunk) - fade_len
        return output[:cursor]