import librosa
import numpy as np
import warnings
import random
from scipy.ndimage import median_filter

warnings.filterwarnings('ignore')

class AudioRemixer:
    def __init__(self, audio_path, sr=22050):
        self.path = audio_path
        self.sr = sr
        print(f"Loading audio: {audio_path}...")
        self.y, self.sr = librosa.load(audio_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        self.beats = None
        self.beat_times = None
        self.loops = [] # 存储找到的 infinite loops

    def analyze(self):
        """
        使用 Beat-Synchronous Shingling 技术寻找无限循环点
        """
        print("Analyzing Beat Structure...")
        
        # 1. 提取源分离增强的节拍
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=self.sr)
        
        # 确保 beat_frames 有效
        if len(beat_frames) < 16:
            # 如果节拍太少，强制按固定时间间隔生成
            beat_frames = np.linspace(0, len(self.y), int(self.duration * 2), dtype=int)
            
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        
        # 2. 提取特征 (Chroma + MFCC)
        # Chroma 捕捉和声/旋律 (Harmonic)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
        # MFCC 捕捉音色/鼓点 (Percussive)
        mfcc = librosa.feature.mfcc(y=y_percussive, sr=self.sr, n_mfcc=13)
        
        # 3. 节拍同步 (Beat Synchronization) - 关键步骤
        # 将特征压缩到 Beat 维度，消除微小的节奏偏差
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        mfcc_sync = librosa.util.sync(mfcc, beat_frames, aggregate=np.median)
        
        # 归一化并堆叠
        chroma_sync = librosa.util.normalize(chroma_sync, axis=1)
        mfcc_sync = librosa.util.normalize(mfcc_sync, axis=1)
        features = np.vstack([chroma_sync, mfcc_sync])
        
        # 4. 特征堆叠 (Shingling) - 核心魔法
        # 我们不只比较单拍，而是比较 "4拍的序列"
        # 这样能保证 Loop 不仅仅是音高一样，而是乐句走向一样
        stack_size = 4 
        features_stacked = librosa.feature.stack_memory(features, n_steps=stack_size, delay=1)
        
        # 5. 计算自相似矩阵 (Recurrence Matrix)
        # 这里的 R[i, j] 表示：第 i 个 beat 和 第 j 个 beat 周围的音乐有多像
        R = librosa.segment.recurrence_matrix(features_stacked, width=3, mode='affinity', sym=True)
        
        # 6. 提取对角线 (寻找 Loop)
        self.loops = []
        n_beats = R.shape[0]
        
        # 动态阈值策略：从高分开始找，如果找不到就降低要求
        thresholds = [0.85, 0.75, 0.65, 0.55] 
        
        for thresh in thresholds:
            if len(self.loops) > 50: break # 如果已经找到足够多的 loop，停止
            
            # 扫描对角线 (Lag)
            # 限制最小 Loop 长度为 4 拍 (1小节)，最大为全曲的一半
            for lag in range(4, n_beats // 2):
                diag = np.diagonal(R, offset=lag)
                
                # 寻找连续的高分区域
                # 使用中值滤波平滑，去除噪点
                diag_smooth = median_filter(diag, size=3)
                
                high_sim_indices = np.where(diag_smooth > thresh)[0]
                
                if len(high_sim_indices) == 0: continue
                
                # 合并连续段落
                segments = np.split(high_sim_indices, np.where(np.diff(high_sim_indices) != 1)[0] + 1)
                
                for seg in segments:
                    # 只有当相似长度超过 4 拍时才认为是一个稳固的 Loop
                    if len(seg) >= 4:
                        # 这是一个 Loop!
                        # 原理：Beat[i] 和 Beat[i+lag] 很像
                        # 意味着我们可以从 Beat[i+lag] 跳回 Beat[i] (Loop Back)
                        # 或者从 Beat[i] 跳到 Beat[i+lag] (Skip Forward)
                        
                        start_beat_idx = seg[0]
                        end_beat_idx = seg[-1] # 匹配段落的结束
                        
                        # 转换回时间
                        # Start of the segment
                        t_start = self.beat_times[start_beat_idx]
                        
                        # Jump Point (Source) -> Landing Point (Target)
                        # 我们记录的是一段“可循环区域”
                        # Loop Start: 区域的开始
                        # Loop End: 区域的结束
                        # Jump: Loop End -> Loop Start
                        
                        # 在 Audjust 逻辑里，Loop 通常指：这段音乐本身是重复的
                        # 也就是 Time A 和 Time B 是相似的。
                        # 我们这里提取：从 t_start 开始，持续 length 秒的片段，是可以在内部循环的
                        
                        # 但为了更精准，我们定义 Loop 为一对跳转点：
                        # Point A (Early) and Point B (Late)
                        # 这里的 seg 代表了 A 和 B 的重合部分
                        
                        # Point A (Loop Start)
                        idx_a = start_beat_idx
                        # Point B (Loop End - where we jump back from)
                        idx_b = start_beat_idx + lag
                        
                        if idx_b >= len(self.beat_times): continue
                        
                        time_a = self.beat_times[idx_a]
                        time_b = self.beat_times[idx_b]
                        
                        # 质量分数
                        score = np.mean(diag[seg])
                        
                        # 去重检查 (防止太多相似的)
                        is_duplicate = False
                        for existing in self.loops:
                            if abs(existing['start'] - time_a) < 0.5 and abs(existing['end'] - time_b) < 0.5:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            self.loops.append({
                                "start": time_a,      # Loop Point 1
                                "end": time_b,        # Loop Point 2 (Jump back from here)
                                "duration": time_b - time_a, # Loop Length
                                "score": score,
                                "beats_len": lag,
                                "segment_len": len(seg) # 相似区域的稳固程度
                            })

        # 按分数排序
        self.loops = sorted(self.loops, key=lambda x: x['score'], reverse=True)
        print(f"Found {len(self.loops)} candidate loops.")

    def render_loop_preview(self, loop_data, repetitions=3):
        """
        生成 Loop 预览音频：A -> B -> A -> B ...
        """
        start_t = loop_data['start']
        end_t = loop_data['end']
        
        s_idx = int(start_t * self.sr)
        e_idx = int(end_t * self.sr)
        
        # 基础片段
        segment = self.y[s_idx:e_idx]
        
        # 拼接
        output = segment
        
        # 使用简单的 Crossfade 拼接
        fade_len = int(0.03 * self.sr) # 30ms fade
        
        for _ in range(repetitions - 1):
            if len(segment) < fade_len:
                output = np.concatenate((output, segment))
            else:
                # Crossfade the end of output with start of segment
                prev_tail = output[-fade_len:]
                curr_head = segment[:fade_len]
                
                lin = np.linspace(0, 1, fade_len)
                w_in = np.sin(lin * np.pi / 2)
                w_out = np.cos(lin * np.pi / 2)
                
                overlap = prev_tail * w_out + curr_head * w_in
                
                output = np.concatenate((output[:-fade_len], overlap, segment[fade_len:]))
                
        return output