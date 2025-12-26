import os
# 解决 OpenMP 库冲突问题（Windows 环境）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import librosa
import numpy as np
import warnings
import math
try:
    import torch
    from demucs.separate import main as demucs_main
    DEMUCS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    torch = None
    demucs_main = None
    DEMUCS_AVAILABLE = False
from functools import reduce
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

class AudioRemixer:
    def __init__(self, audio_path, sr=22050):
        self.path = audio_path
        self.sr = sr
        self.y, self.sr = librosa.load(audio_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.beat_times = None
        self.beat_features = None 
        self.loops = []
        self.stem_loops = {}  # 存储分轨后的 Loop
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.global_rms = np.sqrt(np.mean(self.y**2))
        self.current_bpm = 120.0

    def _trim_silence(self):
        mse = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]
        db = librosa.amplitude_to_db(mse, ref=np.max)
        silence_thresh = -60
        last_frame = len(db) - 1
        for i in range(len(db)-1, 0, -1):
            if db[i] > silence_thresh:
                last_frame = i
                break
        last_sample = librosa.frames_to_samples(last_frame, hop_length=512)
        padding = int(0.5 * self.sr) 
        trim_end = min(len(self.y), last_sample + padding)
        if trim_end < len(self.y) - int(0.1 * self.sr):
            self.y = self.y[:trim_end]
            self.duration = trim_end / self.sr
            self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            self.global_rms = np.sqrt(np.mean(self.y**2))

    def _refine_cut_point(self, time_sec, search_window_ms=50):
        """[保留] 微调切点：瞬态回溯 + 过零点锁定"""
        center_sample = int(time_sec * self.sr)
        search_samples = int((search_window_ms / 1000) * self.sr)
        
        start_search = max(0, center_sample - search_samples)
        end_search = min(len(self.y), center_sample + search_samples)
        
        if start_search >= end_search: return center_sample
        
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

        zc_window = 200
        z_start = max(0, target_sample - zc_window)
        z_end = min(len(self.y), target_sample + zc_window)
        
        chunk = self.y[z_start:z_end]
        zero_crossings = np.where(np.diff(np.signbit(chunk)))[0]
        
        if len(zero_crossings) > 0:
            best_zc = zero_crossings[np.argmin(np.abs(zero_crossings - (target_sample - z_start)))]
            return z_start + best_zc
        
        return target_sample

    def _classify_segment(self, start_time, end_time):
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
        self._trim_silence()
        y_harmonic, y_percussive = librosa.effects.hpss(self.y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=self.sr)
        self.current_bpm = float(tempo)
        
        if len(beat_frames) < 16:
            beat_frames = np.linspace(0, len(self.y), int(self.duration * 2), dtype=int)
            
        self.beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y_percussive, sr=self.sr, n_mfcc=13)
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        mfcc_sync = librosa.util.sync(mfcc, beat_frames, aggregate=np.median)
        features = np.vstack([librosa.util.normalize(chroma_sync), librosa.util.normalize(mfcc_sync)])
        self.beat_features = features.T
        
        R = librosa.segment.recurrence_matrix(librosa.feature.stack_memory(features, n_steps=4), width=3, mode='affinity', sym=True)
        self.loops = []
        n_beats = R.shape[0]
        
        for thresh in [0.85, 0.75, 0.65]:
            for lag in range(8, n_beats // 2):
                diag = median_filter(np.diagonal(R, offset=lag), size=3)
                indices = np.where(diag > thresh)[0]
                if len(indices) == 0: continue
                segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
                for seg in segments:
                    if len(seg) >= 8:
                        t_start, t_end = self.beat_times[seg[0]], self.beat_times[seg[0] + lag]
                        if t_end > self.duration - 0.5: continue
                        self.loops.append({
                            "start": float(t_start), "end": float(t_end),
                            "duration": float(t_end - t_start),
                            "score": float(np.mean(diag[seg])), "type": self._classify_segment(t_start, t_end)
                        })
        self.loops = sorted(self.loops, key=lambda x: x['score'], reverse=True)

    def export_analysis_data(self, source_filename="audio.wav"):
        """生成分析报告文档"""
        looping_points = []
        for loop in self.loops:
            looping_points.append({
                "duration": round(float(loop['duration']), 2),
                "start_position": round(float(loop['start']), 2),
                "type": loop['type'], 
                "confidence": round(float(loop['score']), 2)
            })
        raw_data = {
            "source_music": source_filename,
            "total_duration": round(float(self.duration), 2),
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

    def separate_stems(self, output_dir="stems"):
        """使用 Demucs API 进行分轨"""
        # 运行时重新检查，因为 Streamlit 可能会重新加载模块
        demucs_main_func = None
        try:
            from demucs.separate import main as demucs_main_func
        except (ImportError, ModuleNotFoundError):
            # 如果运行时导入失败，尝试使用模块级别的导入
            if DEMUCS_AVAILABLE and demucs_main is not None:
                demucs_main_func = demucs_main
            else:
                raise ImportError("demucs is not installed. Please install it with: pip install demucs")
        
        if demucs_main_func is None:
            raise ImportError("demucs is not installed. Please install it with: pip install demucs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 调用 Demucs 分离 4 轨: drums, bass, other, vocals
        # -n 选择模型, --out 指定输出目录, --mp3 可选（如果支持）
        # 注意：不使用 torchcodec 相关选项，避免依赖问题
        try:
            demucs_main_func(["-n", "htdemucs", "--out", output_dir, self.path])
        except Exception as e:
            error_msg = str(e)
            if "torchcodec" in error_msg.lower():
                raise ImportError("torchcodec is required but not installed. Please install it with: pip install torchcodec")
            elif "demucs" in error_msg.lower() or "import" in error_msg.lower():
                raise ImportError("demucs is not installed. Please install it with: pip install demucs")
            else:
                raise
        
        # 获取生成的音轨路径 (Demucs 默认会按模型名建子文件夹)
        model_name = "htdemucs"
        base_name = os.path.splitext(os.path.basename(self.path))[0]
        stem_path = os.path.join(output_dir, model_name, base_name)
        
        stems = {}
        # 保存 stem 文件路径，供后续混音使用
        self.stem_paths = {}
        for part in ['drums', 'bass', 'other', 'vocals']:
            p = os.path.join(stem_path, f"{part}.wav")
            if os.path.exists(p):
                y, _ = librosa.load(p, sr=self.sr)
                stems[part] = y
                self.stem_paths[part] = p
            else:
                # 如果某个音轨文件不存在，记录警告但继续
                import warnings
                warnings.warn(f"Stem file not found: {p}")
        
        if not stems:
            raise ValueError(f"No stem files found in {stem_path}. Please check if Demucs completed successfully.")
        
        return stems

    def _find_raw_loops(self, y):
        """(简化版内部搜索)"""
        # 使用与 analyze 方法类似的逻辑
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=self.sr)
        
        if len(beat_frames) < 16:
            beat_frames = np.linspace(0, len(y), int(len(y) / self.sr * 2), dtype=int)
        
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y_percussive, sr=self.sr, n_mfcc=13)
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        mfcc_sync = librosa.util.sync(mfcc, beat_frames, aggregate=np.median)
        features = np.vstack([librosa.util.normalize(chroma_sync), librosa.util.normalize(mfcc_sync)])
        
        R = librosa.segment.recurrence_matrix(librosa.feature.stack_memory(features, n_steps=4), width=3, mode='affinity', sym=True)
        raw_loops = []
        n_beats = R.shape[0]
        duration = len(y) / self.sr
        
        for thresh in [0.85, 0.75, 0.65]:
            for lag in range(8, n_beats // 2):
                diag = median_filter(np.diagonal(R, offset=lag), size=3)
                indices = np.where(diag > thresh)[0]
                if len(indices) == 0: continue
                segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
                for seg in segments:
                    if len(seg) >= 8:
                        t_start, t_end = beat_times[seg[0]], beat_times[seg[0] + lag]
                        if t_end > duration - 0.5: continue
                        raw_loops.append({
                            "start": float(t_start), "end": float(t_end),
                            "duration": float(t_end - t_start),
                            "score": float(np.mean(diag[seg]))
                        })
        return sorted(raw_loops, key=lambda x: x['score'], reverse=True)

    def analyze_with_stems(self, stems):
        """对每个音轨独立分析并量化"""
        # 1. 以鼓点轨确定全局 BPM
        y_ref = stems.get('drums', self.y)
        tempo, _ = librosa.beat.beat_track(y=y_ref, sr=self.sr)
        self.current_bpm = float(tempo)
        beat_dur = 60.0 / self.current_bpm
        
        self.loops = []
        for name, y_stem in stems.items():
            # 这里调用简化的 Loop 查找逻辑
            raw_loops = self._find_raw_loops(y_stem)
            
            for l in raw_loops:
                # --- 核心改进：量化 ---
                # 将秒数转换为整数拍数 (Quantize to Beats)
                q_beats = int(round(l['duration'] / beat_dur))
                if q_beats % 2 != 0 and q_beats > 1:  # 音乐上通常是偶数拍
                    q_beats = q_beats + (1 if q_beats % 4 == 3 else -1)
                
                if q_beats > 0:
                    l['q_beats'] = q_beats
                    l['duration'] = q_beats * beat_dur  # 强制对齐时长
                    l['stem_type'] = name
                    self.loops.append(l)

    def get_beat_metrics(self):
        """计算全曲层面的 GCD 和 LCM"""
        if not self.loops: return None
        
        beat_dur = 60.0 / self.current_bpm
        # 获取所有已量化的拍数
        beat_counts = [l['q_beats'] for l in self.loops if 'q_beats' in l]
        
        if not beat_counts: return None
        
        # 计算数学关系
        gcd_val = reduce(math.gcd, beat_counts)
        lcm_val = reduce(lambda a, b: abs(a * b) // math.gcd(a, b), beat_counts)
        
        return {
            "bpm": round(self.current_bpm, 1),
            "beat_duration": beat_dur,
            "gcd_beats": gcd_val,
            "lcm_beats": lcm_val,
            "cycle_duration": lcm_val * beat_dur,
            "processed_loops": self.loops,
            "type_map": {
                "drums": {"name": "Drums (Kick/Snare)", "color": "#238636"},
                "bass": {"name": "Bassline", "color": "#1f6feb"},
                "other": {"name": "Melody/Synth", "color": "#d2a8ff"},
                "vocals": {"name": "Vocals/FX", "color": "#f85149"}
            }
        }

    def render_lcm_remix(self, metrics, n_cycles=4):
        """生成基于 LCM 的多轨对齐混音"""
        if not hasattr(self, 'stem_paths') or not self.stem_paths:
            # 如果没有分轨路径，无法进行多轨混音
            return None
            
        lcm_beats = metrics['lcm_beats']
        beat_dur = metrics['beat_duration']
        cycle_dur = lcm_beats * beat_dur
        total_samples = int(cycle_dur * n_cycles * self.sr)
        
        # 立体声输出 buffer
        output = np.zeros(total_samples)
        
        # 按类型分组 loops
        tracks = {}
        for l in self.loops:
            t_type = l.get('stem_type')
            if t_type:
                if t_type not in tracks: tracks[t_type] = []
                tracks[t_type].append(l)
        
        # 为每个轨道选择最佳 loop 并填充
        for t_type, loops in tracks.items():
            if t_type not in self.stem_paths: continue
            
            # 策略：选择最长的 Loop，优先覆盖更多拍数
            best_loop = max(loops, key=lambda x: x['duration'])
            
            # 读取原始音频
            try:
                y_stem, _ = librosa.load(self.stem_paths[t_type], sr=self.sr)
            except Exception:
                continue
                
            # 提取 Loop 片段
            s_idx = int(best_loop['start'] * self.sr)
            e_idx = int(best_loop['end'] * self.sr)
            loop_audio = y_stem[s_idx:e_idx]
            
            if len(loop_audio) == 0: continue
            
            # 计算在 LCM 周期内需要重复多少次
            # 理论上 loop_dur = q_beats * beat_dur
            # LCM / q_beats = 重复次数
            q_beats = best_loop['q_beats']
            repeats_per_cycle = lcm_beats // q_beats
            
            # 构建单周期音频
            cycle_audio = np.tile(loop_audio, repeats_per_cycle)
            
            # 裁剪或填充至精确的周期长度
            cycle_samples = int(cycle_dur * self.sr)
            if len(cycle_audio) > cycle_samples:
                cycle_audio = cycle_audio[:cycle_samples]
            elif len(cycle_audio) < cycle_samples:
                cycle_audio = np.pad(cycle_audio, (0, cycle_samples - len(cycle_audio)))
            
            # 重复 n_cycles 次
            track_audio = np.tile(cycle_audio, n_cycles)
            
            # 叠加到总输出 (简单的加法混音)
            # 确保长度一致
            if len(track_audio) > len(output):
                track_audio = track_audio[:len(output)]
            elif len(track_audio) < len(output):
                track_audio = np.pad(track_audio, (0, len(output) - len(track_audio)))
                
            output += track_audio
            
        # 归一化防止爆音
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95
            
        return output

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
        """弹性搜索最佳缝合点"""
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
        """高级路径规划：支持延长 (Loop Back) 和 缩短 (Skip Forward)"""
        
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