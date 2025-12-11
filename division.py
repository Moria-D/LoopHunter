import librosa
import numpy as np
import warnings
import random
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

def snap_to_transient(y, idx, search_window=500):
    """瞬态对齐"""
    if idx < search_window: return max(0, idx)
    if idx >= len(y) - search_window: return min(len(y), idx)
    
    window = y[idx - search_window : idx + search_window]
    energy = librosa.feature.rms(y=window, frame_length=64, hop_length=16)[0]
    if len(energy) == 0: return idx
    local_peak_frame = np.argmax(energy)
    shift = (local_peak_frame * 16) - search_window
    
    # 过零点微调
    new_idx = idx + shift
    zc_window = y[max(0, new_idx - 50) : min(len(y), new_idx + 50)]
    if len(zc_window) > 0:
        zc = np.where(np.diff(np.signbit(zc_window)))[0]
        if len(zc) > 0:
            closest_zc = zc[np.argmin(np.abs(zc - 50))]
            return new_idx - 50 + closest_zc
            
    return new_idx

def get_bar_similarity(features, idx_a, idx_b):
    """计算两个小节的声学相似度 (0-1)"""
    if idx_a >= len(features) or idx_b >= len(features): return 0.0
    # Cosine distance: 0 is same, 1 is diff
    dist = cdist([features[idx_a]], [features[idx_b]], metric='cosine')[0][0]
    return 1.0 - dist

class AudioRemixer:
    def __init__(self, audio_path, sr=22050):
        self.path = audio_path
        self.sr = sr
        print(f"Loading audio: {audio_path}...")
        self.y, self.sr = librosa.load(audio_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        self.bars = []
        self.bar_features = None

    def analyze(self):
        """分析小节与特征"""
        print("Analyzing structure...")
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=self.sr)
        if np.ndim(tempo) > 0: tempo = tempo.item()
        
        beats = librosa.frames_to_time(beat_frames, sr=self.sr)
        if len(beats) > 0 and beats[0] > 0.5:
            beats = np.insert(beats, 0, 0.0)
            
        # 构建小节 (4/4拍)
        self.bars = []
        beats_per_bar = 4
        
        if len(beats) < 4:
            self.bars.append({"index": 0, "start": 0.0, "end": self.duration, "duration": self.duration})
        else:
            for i in range(0, len(beats), beats_per_bar):
                if i + beats_per_bar < len(beats):
                    start_t = beats[i]
                    end_t = beats[i + beats_per_bar]
                    self.bars.append({
                        "index": len(self.bars),
                        "start": start_t,
                        "end": end_t,
                        "duration": end_t - start_t
                    })
            
            # 强制包含文件尾部
            last_bar = self.bars[-1]
            if last_bar['end'] < self.duration:
                if self.duration - last_bar['end'] > 1.0:
                     self.bars.append({
                        "index": len(self.bars),
                        "start": last_bar['end'],
                        "end": self.duration,
                        "duration": self.duration - last_bar['end']
                    })
                else:
                    last_bar['end'] = self.duration
                    last_bar['duration'] = self.duration - last_bar['start']

        # 提取小节特征 (用于计算相似度)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr)
        features = np.vstack([chroma, mfcc])
        
        n_bars = len(self.bars)
        self.bar_features = np.zeros((n_bars, features.shape[0]))
        
        for i, bar in enumerate(self.bars):
            s_frame = librosa.time_to_frames(bar['start'], sr=self.sr)
            e_frame = librosa.time_to_frames(bar['end'], sr=self.sr)
            if e_frame > s_frame:
                self.bar_features[i] = np.mean(features[:, s_frame:e_frame], axis=1)

    def generate_path(self, target_duration):
        if not self.bars: self.analyze()
        if not self.bars: return []
        
        # 分流逻辑
        if target_duration < self.duration:
            return self._generate_shortening_path(target_duration)
        else:
            return self._generate_extending_path(target_duration)

    def _generate_shortening_path(self, target):
        """
        缩短模式：寻找最佳单点剪辑 (Single Cut)
        Head (0->A) + Tail (B->End) = Target
        """
        n_bars = len(self.bars)
        needed_remove = self.duration - target
        
        # 极端情况：目标太短，直接取开头或结尾
        if target < 10.0:
            # 这种情况下只保留 Outro
            trim_start = max(0, self.duration - target)
            return [{
                "source_start": trim_start, "source_end": self.duration,
                "duration": target, "type": "Tail Only", "xfade": 0, "remix_start": 0.0
            }]

        best_score = -1.0
        best_cut = None # (idx_a, idx_b)
        
        # 限制搜索范围，保护 Intro 和 Outro
        # A 点 (跳出点): 至少保留前 2 小节
        min_a = 2
        # B 点 (跳入点): 至少保留后 2 小节
        max_b = n_bars - 2
        
        # 遍历所有可能的 A 点
        for i in range(min_a, n_bars - 4):
            # 计算理想的 B 点位置
            # time(B) ≈ time(A) + needed_remove
            time_a = self.bars[i]['end']
            ideal_time_b = time_a + needed_remove
            
            # 在 ideal_time_b 附近寻找最近的小节线 j
            # 我们可以通过查找 bar start time 来快速定位
            # 简单遍历优化：只看 i 之后的部分
            
            for j in range(i + 1, max_b):
                time_b = self.bars[j]['start']
                
                # 如果这个 B 点会导致总时长误差太大 (> 5秒)，跳过
                est_duration = time_a + (self.duration - time_b)
                if abs(est_duration - target) > 5.0:
                    continue
                
                # 计算 A -> B 的衔接分数
                # 1. 声学相似度 (下一小节 i+1 和 j 的相似度)
                # 如果 bar[i+1] 和 bar[j] 很像，说明从 i 跳到 j 听起来像是在继续播放 i+1
                # 或者：比较 bar[i] 和 bar[j-1] (前文相似性)
                
                # 我们比较 "应该接什么" (i+1) 和 "实际接了什么" (j)
                sim_score = get_bar_similarity(self.bar_features, i + 1, j)
                
                # 2. 能量匹配度
                # 3. 节奏相位 (都是 Bar Start，天然对齐)
                
                if sim_score > best_score:
                    best_score = sim_score
                    best_cut = (i, j)

        # 构建 Timeline
        timeline = []
        t_cursor = 0.0
        
        if best_cut:
            idx_a, idx_b = best_cut
            # Part 1: 0 -> A.end
            dur_a = self.bars[idx_a]['end']
            timeline.append({
                "source_start": 0.0,
                "source_end": dur_a,
                "duration": dur_a,
                "type": "Head",
                "xfade": 0,
                "remix_start": 0.0
            })
            t_cursor += dur_a
            
            # Part 2: B.start -> End
            start_b = self.bars[idx_b]['start']
            dur_b = self.duration - start_b
            timeline.append({
                "source_start": start_b,
                "source_end": self.duration,
                "duration": dur_b,
                "type": "Tail",
                "xfade": 25, # 切割点给一个标准的 Crossfade
                "remix_start": t_cursor
            })
        else:
            # Fallback: 直接硬切中间
            cut_point = target / 2
            tail_len = target / 2
            timeline.append({
                "source_start": 0.0, "source_end": cut_point, "duration": cut_point, 
                "type": "Head", "xfade": 0, "remix_start": 0.0
            })
            timeline.append({
                "source_start": self.duration - tail_len, "source_end": self.duration, 
                "duration": tail_len, "type": "Tail", "xfade": 40, "remix_start": cut_point
            })
            
        return timeline

    def _generate_extending_path(self, target_duration):
        """延长模式：循环 Body (保留之前的逻辑)"""
        n_bars = len(self.bars)
        
        # Intro/Outro 锁定
        intro_len = min(4, n_bars)
        intro_bars = self.bars[:intro_len]
        
        outro_count = 4 if n_bars > 16 else 2
        outro_start_idx = max(intro_len, n_bars - outro_count)
        outro_bars = self.bars[outro_start_idx:]
        
        intro_dur = sum(b['duration'] for b in intro_bars)
        outro_dur = sum(b['duration'] for b in outro_bars)
        
        body_start_idx = intro_len
        body_end_idx = outro_start_idx - 1
        
        timeline = []
        current_dur = 0.0
        
        # Intro
        for b in intro_bars:
            timeline.append({**b, "type": "Intro", "xfade": 0, "remix_start": 0.0})
        current_dur += intro_dur
        
        # Body Loop
        fill_target = target_duration - outro_dur
        if body_end_idx >= body_start_idx:
            curr_idx = body_start_idx
            while current_dur < fill_target:
                bar = self.bars[curr_idx]
                prev = timeline[-1]
                is_natural = (bar['index'] == prev['index'] + 1)
                xfade = 0 if is_natural else 20
                label = "Body" if is_natural else "Loop ⟳"
                
                timeline.append({**bar, "type": "Body", "label": label, "xfade": xfade})
                current_dur += bar['duration']
                
                curr_idx += 1
                if curr_idx > body_end_idx:
                    curr_idx = body_start_idx # Loop back
        
        # Outro
        for i, b in enumerate(outro_bars):
            prev = timeline[-1]
            is_natural = (b['index'] == prev['index'] + 1)
            xfade = 0 if is_natural else 40
            timeline.append({**b, "type": "Outro", "xfade": xfade})
            current_dur += b['duration']
            
        # Timestamp update
        t_c = 0.0
        for x in timeline:
            x['remix_start'] = t_c
            t_c += x['duration']
            
        return timeline

    def render(self, timeline):
        if not timeline: return np.array([])
        
        total_samples = int(sum([seg['duration'] for seg in timeline]) * self.sr)
        output = np.zeros(total_samples + 44100)
        
        cursor = 0
        for i, seg in enumerate(timeline):
            s_idx = int(seg['source_start'] * self.sr)
            e_idx = int(seg['source_end'] * self.sr)
            
            # 瞬态对齐 (除了文件末尾)
            if i > 0: s_idx = snap_to_transient(self.y, s_idx)
            # 只有当这不是文件真正的末尾时才对齐结束点
            if seg['source_end'] < self.duration - 0.5:
                e_idx = snap_to_transient(self.y, e_idx)
            
            if e_idx > len(self.y): e_idx = len(self.y)
            chunk = self.y[s_idx:e_idx]
            
            fade_ms = seg.get('xfade', 0)
            fade_pts = int((fade_ms / 1000) * self.sr)
            
            if i == 0 or fade_pts == 0:
                output[cursor:cursor+len(chunk)] = chunk
                cursor += len(chunk)
            else:
                overlap_start = cursor - fade_pts
                if overlap_start < 0: overlap_start = 0
                prev = output[overlap_start:cursor]
                curr = chunk[:fade_pts]
                n = min(len(prev), len(curr))
                
                if n > 0:
                    lin = np.linspace(0, 1, n)
                    output[overlap_start:overlap_start+n] = prev[:n]*(1-lin) + curr[:n]*lin
                rest = chunk[n:]
                output[cursor:cursor+len(rest)] = rest
                cursor += len(rest)
                
        return output[:cursor]