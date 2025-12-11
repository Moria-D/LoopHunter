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
        
        self.beat_features = np.vstack([chroma_sync, mfcc_sync]).T
        
        # 3. 寻找 Loops
        stack_size = 4 
        features = np.vstack([chroma_sync, mfcc_sync])
        features_stacked = librosa.feature.stack_memory(features, n_steps=stack_size, delay=1)
        R = librosa.segment.recurrence_matrix(features_stacked, width=3, mode='affinity', sym=True)
        
        self.loops = []
        n_beats = R.shape[0]
        thresholds = [0.85, 0.75, 0.65, 0.55] 
        
        for thresh in thresholds:
            if len(self.loops) > 100: break # 获取更多 Loop 供用户挑选
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
        
        # 按分数排序，去除极其相似的重叠项
        self.loops = sorted(self.loops, key=lambda x: x['score'], reverse=True)
        unique_loops = []
        for l in self.loops:
            is_dup = False
            for u in unique_loops:
                if abs(l['start'] - u['start']) < 0.5 and abs(l['end'] - u['end']) < 0.5:
                    is_dup = True
                    break
            if not is_dup:
                unique_loops.append(l)
        self.loops = unique_loops

    def generate_loop_preview(self, loop_data, repetitions=4):
        """
        生成循环预览音频：将 Loop 片段重复 N 次并应用 Crossfade
        """
        s_idx = int(loop_data['start'] * self.sr)
        e_idx = int(loop_data['end'] * self.sr)
        
        chunk = self.y[s_idx:e_idx]
        if len(chunk) == 0: return np.array([])

        output = chunk
        fade_len = int(0.03 * self.sr)
        
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

    def _find_best_cut_points(self, target_duration):
        n_beats = len(self.beat_times)
        remove_amount = self.duration - target_duration
        best_score = -999.0
        best_cut = None 
        min_idx = 4
        max_idx = n_beats - 4
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

    def plan_multi_loop_remix(self, target_duration):
        if target_duration < self.duration:
            if target_duration < 5.0:
                return self._plan_hard_cut(target_duration), target_duration
            cut_indices, score = self._find_best_cut_points(target_duration)
            timeline = []
            if cut_indices:
                idx_a, idx_b = cut_indices
                time_a = self.beat_times[idx_a]
                time_b = self.beat_times[idx_b]
                timeline.append({"source_start": 0.0, "source_end": time_a, "duration": time_a, "type": "Head", "xfade": 0, "remix_start": 0.0})
                tail_dur = self.duration - time_b
                timeline.append({"source_start": time_b, "source_end": self.duration, "duration": tail_dur, "type": "Tail", "xfade": 30, "remix_start": time_a, "is_jump": True})
                return timeline, time_a + tail_dur
            else:
                return self._plan_hard_cut(target_duration), target_duration
        else:
            if not self.loops:
                return [{"source_start":0, "source_end":self.duration, "duration":self.duration, "type":"Original", "remix_start":0}], self.duration
            best_loop = self.loops[0]
            loop_dur = best_loop['duration']
            head_dur = best_loop['start']
            tail_dur = self.duration - best_loop['end']
            needed = target_duration - (head_dur + tail_dur)
            repeats = 1 if needed <= 0 else max(1, int(round(needed / loop_dur)))
            timeline = []
            cursor = 0.0
            timeline.append({"source_start":0.0, "source_end":best_loop['end'], "duration":best_loop['end'], "type":"Head", "xfade":0, "remix_start":0.0})
            cursor += best_loop['end']
            for i in range(repeats):
                timeline.append({"source_start":best_loop['start'], "source_end":best_loop['end'], "duration":loop_dur, "type":"Loop Extension", "xfade":30, "is_jump":True, "remix_start":cursor})
                cursor += loop_dur
            timeline.append({"source_start":best_loop['end'], "source_end":self.duration, "duration":tail_dur, "type":"Tail", "xfade":0, "remix_start":cursor})
            return timeline, cursor + tail_dur

    def _plan_hard_cut(self, target):
        cut = target / 2
        return [
            {"source_start":0, "source_end":cut, "duration":cut, "type":"Head", "remix_start":0},
            {"source_start":self.duration-(target-cut), "source_end":self.duration, "duration":target-cut, "type":"Tail", "xfade":50, "is_jump":True, "remix_start":cut}
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
                if n > 1:
                    lin = np.linspace(0, 1, n)
                    mix = prev*(1-lin) + curr*lin
                    output[overlap:overlap+n] = mix
                output[cursor:cursor+len(chunk)-fade_len] = chunk[fade_len:]
                cursor += len(chunk) - fade_len
        return output[:cursor]