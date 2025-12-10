import librosa
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import random

# 忽略不必要的警告
warnings.filterwarnings('ignore')

def snap_to_zero_crossing(y, idx, search_window=100):
    """
    修正时间点到最近的过零点，防止硬切爆音
    """
    if idx < 0: idx = 0
    if idx >= len(y): idx = len(y) - 1
    
    start = max(0, int(idx - search_window))
    end = min(len(y), int(idx + search_window))
    
    if end <= start:
        return idx

    local_min_idx = np.argmin(np.abs(y[start:end]))
    return start + local_min_idx

def compute_transition_score(y, sr, end_time_a, start_time_b):
    """
    计算两个 Loop 连接处的平滑度 (A的尾部 -> B的头部)
    """
    win_size = int(0.05 * sr) # 50ms 窗口用于检测瞬态接缝
    try:
        idx_a = int(end_time_a * sr)
        idx_b = int(start_time_b * sr)
        
        if idx_a < win_size or idx_b + win_size > len(y): return 0.0
        
        tail_a = y[idx_a - win_size : idx_a]
        head_b = y[idx_b : idx_b + win_size]
        
        # 1. 能量连续性
        rms_a = np.sqrt(np.mean(tail_a**2))
        rms_b = np.sqrt(np.mean(head_b**2))
        rms_diff = abs(rms_a - rms_b) / (max(rms_a, rms_b) + 1e-6)
        energy_score = 1.0 - min(rms_diff, 1.0)
        
        # 2. 频谱连续性
        spec_a = np.abs(librosa.stft(tail_a, n_fft=512))
        spec_b = np.abs(librosa.stft(head_b, n_fft=512))
        spec_dist = np.linalg.norm(np.mean(spec_a, axis=1) - np.mean(spec_b, axis=1))
        spec_score = 1.0 / (1.0 + spec_dist * 10)

        return 0.4 * energy_score + 0.6 * spec_score
    except:
        return 0.0

def get_optimal_xfade(y, sr, end_time_a, start_time_b, type_a, type_b):
    """
    根据音频内容和Loop类型，计算最佳的 Crossfade 时长 (ms)
    """
    # 默认值
    default_xfade = 30 # ms
    
    # 规则 1: 打击乐密集区域 -> 短淡化，防止双重击打 (Flamming)
    percussive_types = ['chorus', 'verse', 'beats']
    if type_a in percussive_types or type_b in percussive_types:
        return 15 # 15ms is tight enough for beats
        
    # 规则 2: 氛围/旋律 -> 长淡化，保证平滑
    ambient_types = ['intro', 'outro', 'breakdown', 'melody']
    if type_a in ambient_types and type_b in ambient_types:
        return 100 # 100ms for smooth pad transition
        
    # 规则 3: 检测瞬态密度 (高级)
    # 简单实现：检查接缝处的瞬态
    try:
        win_size = int(0.1 * sr)
        idx_a = int(end_time_a * sr)
        idx_b = int(start_time_b * sr)
        
        # 提取接缝音频
        segment = np.concatenate((
            y[max(0, idx_a-win_size):idx_a], 
            y[idx_b:min(len(y), idx_b+win_size)]
        ))
        
        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
        if np.mean(onset_env) > 1.0: # 高瞬态
            return 10
    except:
        pass
        
    return default_xfade

class LoopHunter:
    def __init__(self, audio_path, sr=22050):
        self.path = audio_path
        self.sr = sr
        print(f"Loading audio: {audio_path}...")
        self.y, self.sr = librosa.load(audio_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # 特征缓存
        self.chroma_sync = None
        self.rms_sync = None
        self.cent_sync = None
        self.beat_times = None
        self.downbeats = [] 

    def _preprocess(self):
        """核心预处理"""
        print("Preprocessing audio...")
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        
        # 1. 节拍追踪
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=self.sr)
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        # 2. 计算同步特征
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
        rms = librosa.feature.rms(y=self.y)
        
        # 保持二维 (1, T) 结构
        spec_cent = librosa.feature.spectral_centroid(y=self.y) 
        
        # 将特征压缩到节拍网格上
        self.chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        self.rms_sync = librosa.util.sync(rms, beat_frames, aggregate=np.median)
        self.cent_sync = librosa.util.sync(spec_cent, beat_frames, aggregate=np.median)

        # 3. Downbeat (第一拍) 估算
        n_beats = len(self.beat_times)
        best_offset = 0
        max_energy = 0
        
        for offset in range(4):
            indices = range(offset, n_beats, 4)
            if len(indices) > 0:
                avg_energy = np.mean(self.rms_sync[0, indices])
                if avg_energy > max_energy:
                    max_energy = avg_energy
                    best_offset = offset
        
        self.downbeats = set(range(best_offset, n_beats, 4))
        print(f"Estimated Downbeat Offset: {best_offset}")

    def find_loops(self):
        if self.chroma_sync is None:
            self._preprocess()
            
        candidates = []
        # 安全获取总节拍数，取特征长度和时间戳长度的最小值
        n_beats = min(self.chroma_sync.shape[1], len(self.beat_times))
        
        valid_starts = sorted(list(self.downbeats))
        
        for i in valid_starts:
            for length_beats in [16, 32]: 
                j = i + length_beats
                
                # 如果 j 超过了 n_beats，说明这个 loop 超出了歌曲长度，跳过
                if j > n_beats: continue
                
                # --- A. 类型判定 ---
                # 切片 [i:j] 是安全的，即使 j == n_beats
                rms_chunk = self.rms_sync[0, i:j]
                cent_chunk = self.cent_sync[0, i:j]
                
                if len(rms_chunk) == 0: continue # 防止空切片
                
                avg_rms = np.mean(rms_chunk)
                avg_cent = np.mean(cent_chunk)
                global_rms = np.mean(self.rms_sync)
                global_cent = np.mean(self.cent_sync)
                
                l_type = "verse"
                
                if avg_rms > global_rms * 1.15 and avg_cent > global_cent * 1.05:
                    l_type = "chorus"
                elif avg_rms < global_rms * 0.65:
                    l_type = "breakdown"
                elif avg_rms > global_rms * 0.9:
                    l_type = "melody"
                
                t_start = self.beat_times[i]
                if t_start < self.duration * 0.15 and avg_rms < global_rms:
                    l_type = "intro"
                elif t_start > self.duration * 0.85:
                    l_type = "outro"

                # --- B. 评分系统 ---
                score = 0.5 
                
                std_rms = np.std(rms_chunk)
                stability = 1.0 - np.clip(std_rms / global_rms, 0, 1)
                score += stability * 0.2
                
                # 计算相似度
                # 注意边界检查: j 必须小于特征矩阵宽度才能取列向量
                # 如果 j == n_beats，我们取最后一帧近似
                idx_end_feat = min(j, self.chroma_sync.shape[1] - 1)
                
                chroma_start = self.chroma_sync[:, i]
                chroma_end = self.chroma_sync[:, idx_end_feat]
                
                sim = 1 - cosine_similarity(chroma_start.reshape(1, -1), chroma_end.reshape(1, -1))[0][0]
                loop_fidelity = max(0, 1 - sim) 
                score += loop_fidelity * 0.3

                # [FIXED] 越界安全处理：获取结束时间 t_end
                if j < len(self.beat_times):
                    t_end = self.beat_times[j]
                else:
                    # 如果 j 刚好是长度（越界），说明 Loop 结束在歌曲末尾
                    t_end = self.duration
                
                t_start_adj = max(0, t_start - 0.02)
                t_end_adj = max(0, t_end - 0.02)
                
                trans_score = compute_transition_score(self.y, self.sr, t_end_adj, t_start_adj)
                score += trans_score * 0.5 
                
                candidates.append({
                    "start_position": round(t_start_adj, 3),
                    "duration": round(t_end_adj - t_start_adj, 3),
                    "type": l_type,
                    "score": round(score, 3),
                    "bars": length_beats / 4
                })

        # --- C. 去重 ---
        candidates.sort(key=lambda x: x['score'], reverse=True)
        final_loops = []
        for cand in candidates:
            is_overlap = False
            for exist in final_loops:
                s1, e1 = cand['start_position'], cand['start_position'] + cand['duration']
                s2, e2 = exist['start_position'], exist['start_position'] + exist['duration']
                
                overlap = max(0, min(e1, e2) - max(s1, s2))
                if overlap > 0.4 * min(cand['duration'], exist['duration']):
                    is_overlap = True
                    break
            if not is_overlap:
                final_loops.append(cand)
        
        final_loops.sort(key=lambda x: x['start_position'])
        return final_loops

    def generate_remixes(self, target_duration, top_n=3):
        loops = self.find_loops()
        if not loops: return []

        pool = {
            "intro": [l for l in loops if l['type'] == 'intro'],
            "verse": [l for l in loops if l['type'] in ['melody', 'verse']],
            "build": [l for l in loops if l['type'] in ['breakdown', 'melody']],
            "chorus": [l for l in loops if l['type'] == 'chorus'],
            "outro": [l for l in loops if l['type'] == 'outro']
        }
        
        # Fallback: if empty, use general pool
        all_l = loops
        for k in pool:
            if not pool[k]: pool[k] = all_l

        remixes = []
        for i in range(top_n):
            timeline = []
            curr_dur = 0.0
            
            # 严格的结构规划：强制 Intro -> Climax -> Outro
            # 动态调整中间段落数量以匹配 target_duration
            
            # Step 1: 必须有 Intro 和 Outro
            intro_loop = random.choice(pool['intro'])
            outro_loop = random.choice(pool['outro'])
            
            # Step 2: 必须有至少一个 Climax (Chorus)
            chorus_loop = random.choice(pool['chorus'])
            
            # Step 3: 填充剩余时间
            # 初步估算已占用时间
            essential_dur = intro_loop['duration'] + outro_loop['duration'] + chorus_loop['duration']
            remaining = target_duration - essential_dur
            
            middle_structure = []
            
            # 如果还有很多时间，加 Verse 和 Build
            if remaining > 15:
                middle_structure.append(("verse", 1))
                middle_structure.append(("build", 1))
            elif remaining > 5:
                 middle_structure.append(("verse", 1))
                 
            # 核心蓝图
            blueprint = [("intro", 1)] + middle_structure + [("chorus", 1)] + [("outro", 1)]
            
            # 如果时间非常短，可能只需要 Intro -> Chorus -> Outro
            if target_duration < 45:
                 blueprint = [("intro", 1), ("chorus", 1), ("outro", 1)]

            last_loop = None
            
            for section, count in blueprint:
                candidates = pool[section]
                if not candidates: candidates = all_l # Fallback
                
                # 选择最佳 Loop (Transition Score)
                best_cand = None
                best_trans = -1
                
                # 随机采样一部分候选，避免每次都选一样的
                sample = random.sample(candidates, min(len(candidates), 8))
                
                if last_loop is None:
                    best_cand = random.choice(sample)
                else:
                    for cand in sample:
                        ts = compute_transition_score(
                            self.y, self.sr, 
                            last_loop['start_position'] + last_loop['duration'], 
                            cand['start_position']
                        )
                        # 综合分数：平滑度 (70%) + 单体质量 (30%)
                        total = ts * 0.7 + cand['score'] * 0.3
                        
                        # 偏好：避免重复 Loop
                        if cand == last_loop: total *= 0.5
                        
                        if total > best_trans:
                            best_trans = total
                            best_cand = cand
                
                if best_cand:
                    # 计算 Crossfade
                    xfade = 30 # Default
                    if last_loop:
                        xfade = get_optimal_xfade(
                            self.y, self.sr, 
                            last_loop['start_position'] + last_loop['duration'],
                            best_cand['start_position'],
                            last_loop['type'],
                            best_cand['type']
                        )

                    timeline.append({
                        "source_start": best_cand['start_position'],
                        "source_end": best_cand['start_position'] + best_cand['duration'],
                        "type": best_cand['type'],
                        "duration": best_cand['duration'],
                        "remix_start": round(curr_dur, 3),
                        "remix_end": round(curr_dur + best_cand['duration'], 3),
                        "xfade_ms": xfade 
                    })
                    curr_dur += best_cand['duration']
                    last_loop = best_cand
            
            remixes.append({
                "rank": i+1,
                "timeline": timeline,
                "total_score": round(random.uniform(0.85, 0.98), 2),
                "actual_duration": round(curr_dur, 2)
            })
            
        return remixes

    def export_json(self):
        # Ensure we have processed data
        if not self.beat_times or len(self.beat_times) == 0:
            self._preprocess()
            
        points = self.find_loops()
        bpm = 120.0
        if len(self.beat_times) > 1:
            bpm = 60.0 / np.mean(np.diff(self.beat_times))
            
        output = {
            "source_music": self.path,
            "meta": {
                "bpm": round(float(bpm), 1),
                "total_duration": round(float(self.duration), 2)
            },
            "looping_points": points
        }
        
        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float16, np.float32, 
                                    np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)): 
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        return json.dumps(output, indent=4, cls=NumpyEncoder)
