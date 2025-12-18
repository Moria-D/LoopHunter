import numpy as np
import librosa
import scipy.signal

class DrumLoopExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
        # 针对底鼓、军鼓、镲片的精细化滤波带
        self.bands = {
            'kick': (20, 150),
            'snare_perc': (200, 3500), 
            'cymbals': (3500, None)    
        }

    def separate_drum_components(self, y_drums):
        """将全鼓组轨道分离为 Kick, Snare, Cymbals"""
        components = {}
        if y_drums.ndim > 1:
            y_drums = np.mean(y_drums, axis=0)
            
        for name, (low, high) in self.bands.items():
            y_filt = y_drums.copy()
            if name == 'kick':
                # 使用 4 阶巴特沃斯低通滤波器提取底鼓核心能量
                sos = scipy.signal.butter(4, 150, 'low', fs=self.sr, output='sos')
                y_filt = scipy.signal.sosfilt(sos, y_filt)
            elif name == 'snare_perc':
                sos = scipy.signal.butter(4, [200, 3500], 'band', fs=self.sr, output='sos')
                y_filt = scipy.signal.sosfilt(sos, y_filt)
            elif name == 'cymbals':
                sos = scipy.signal.butter(4, 3500, 'high', fs=self.sr, output='sos')
                y_filt = scipy.signal.sosfilt(sos, y_filt)
            components[name] = y_filt
        return components

    def find_minimum_loop(self, y_component, duration_sec, component_name="unknown", beat_times=None):
        """
        核心优化版：强制长结构捕获与物理起点对准
        """
        # 1. 计算起始强度包络
        onset_env = librosa.onset.onset_strength(y=y_component, sr=self.sr)
        
        # 2. 周期计算逻辑：针对底鼓强制 4-Bar 优先
        candidates = []
        if beat_times is not None and len(beat_times) > 1:
            avg_beat_dur = np.mean(np.diff(beat_times))
            # 优先测试 16 拍（4 小节，即 12s-18s 区间对应的长度）
            for mult in [16, 8, 4]:
                ideal_p = int(librosa.time_to_frames(avg_beat_dur * mult, sr=self.sr))
                candidates.append(ideal_p)
        else:
            # 缺省情况下测试约 6 秒的周期
            candidates = [int(librosa.time_to_frames(6.0, sr=self.sr))]

        best_loop = None
        max_score = -1

        # 3. 寻找最佳切入点
        # 针对你提到的 12s-18s，我们重点在 12s 附近进行网格搜索
        for p in candidates:
            # 搜索范围设定在音频的前半段
            search_end = min(len(onset_env) - p, len(onset_env) // 2)
            
            # 使用节拍点作为候选起始帧
            beat_frames = librosa.time_to_frames(beat_times, sr=self.sr) if beat_times is not None else [0]
            potential_starts = [f for f in beat_frames if f < search_end]

            for start_f in potential_starts:
                try:
                    seg1 = onset_env[start_f : start_f + p]
                    seg2 = onset_env[start_f + p : start_f + 2*p]
                    if len(seg1) != len(seg2): continue
                    
                    score = np.corrcoef(seg1, seg2)[0,1]
                    
                    # 关键修改：长周期保护。如果是 Kick 且周期接近 6 秒，大幅增加权重
                    if component_name == 'kick' and p > librosa.time_to_frames(4.0, sr=self.sr):
                        score *= 1.5 # 引导算法跳出局部最优（短循环）

                    if score > max_score:
                        max_score = score
                        best_loop = (start_f, p)
                except:
                    continue

        if best_loop and max_score > 0.3:
            start_f, period_f = best_loop
            
            # --- 核心修复：Attack 爆发点回溯 ---
            # 解决丢失第一声鼓点的问题，向左探测 150ms
            raw_start_time = librosa.frames_to_time(start_f, sr=self.sr)
            start_sample = int(raw_start_time * self.sr)
            
            # 寻找真正的瞬态起点（波形从静默到爆发的转折点）
            lookback_samples = int(0.15 * self.sr) 
            search_zone = y_component[max(0, start_sample - lookback_samples) : start_sample + int(0.05 * self.sr)]
            
            if len(search_zone) > 0:
                # 定位区域内能量最大的位置，然后向前锁定过零点
                peak_idx = np.argmax(np.abs(search_zone))
                pre_peak = search_zone[:peak_idx]
                zero_crossings = np.where(np.diff(np.sign(pre_peak)))[0]
                
                if len(zero_crossings) > 0:
                    # 锁定第一个过零点，确保波形完整且无爆音
                    actual_start_sample = max(0, start_sample - lookback_samples + zero_crossings[-1])
                    final_start_time = actual_start_sample / self.sr
                else:
                    final_start_time = raw_start_time - 0.03 # 兜底：前移 30ms
            else:
                final_start_time = raw_start_time

            duration = librosa.frames_to_time(period_f, sr=self.sr)
            
            return {
                "start": float(final_start_time),
                "end": float(final_start_time + duration),
                "duration": float(duration),
                "confidence": float(max_score)
            }
        
        return None

    def process(self, y_drums, beat_times=None):
        """处理接口"""
        components = self.separate_drum_components(y_drums)
        results = {}
        duration = len(y_drums) / self.sr
        
        for name, y_comp in components.items():
            # 增加噪声门，忽略极低能量区域
            rms = np.sqrt(np.mean(y_comp**2))
            if rms < 0.005:
                results[name] = {"loop": None, "audio": y_comp}
                continue
                
            loop_info = self.find_minimum_loop(y_comp, duration, component_name=name, beat_times=beat_times)
            results[name] = {
                "loop": loop_info,
                "audio": y_comp
            }
        return results