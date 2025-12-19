import numpy as np
import librosa
import scipy.signal

class DrumLoopExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
        # 针对不同打击乐器频段的滤波器配置
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
        全自动动态嗅探版：解决静音提取问题，自动识别循环长度并对齐重拍。
        """
        # 1. 动态特征提取与门限自适应
        onset_env = librosa.onset.onset_strength(y=y_component, sr=self.sr)
        # 计算动态能量门限，防止将有效鼓点识别为静音
        rms = librosa.feature.rms(y=y_component)[0]
        dynamic_thresh = np.percentile(rms, 20) # 取能量分布的低分位作为基准

        if beat_times is None:
            # 使用更灵敏的节拍追踪
            tempo, beats = librosa.beat.beat_track(y=y_component, sr=self.sr, tightness=100)
            beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        if len(beat_times) < 2: return None
        avg_beat_dur = np.mean(np.diff(beat_times))

        # 2. 动态探测最佳循环长度
        possible_mults = [4, 8, 12, 16, 32]
        best_p_frames = int(librosa.time_to_frames(avg_beat_dur * 8, sr=self.sr))
        max_ac_score = -1

        for mult in possible_mults:
            p_test = int(librosa.time_to_frames(avg_beat_dur * mult, sr=self.sr))
            if p_test * 2 > len(onset_env): continue
            
            ac_segment = librosa.autocorrelate(onset_env, max_size=p_test + 1)
            score = ac_segment[p_test]
            
            # 权重补偿：针对底鼓轨道，给予 16 拍及其倍数更高权重
            if component_name == 'kick' and mult >= 16:
                score *= 1.3 
                
            if score > max_ac_score:
                max_ac_score = score
                best_p_frames = p_test

        # 3. 相位自校准：锁定重拍（绿色框起始点）
        best_start_f = 0
        max_phase_weight = -1
        
        # 遍历音频前段的节拍点，寻找“爆发力”与“结构一致性”的最佳平衡点
        search_limit = min(32, len(beat_times))
        for i in range(search_limit):
            start_f = librosa.time_to_frames(beat_times[i], sr=self.sr)
            if start_f + best_p_frames * 2 > len(onset_env): break
            
            # 评分因子：起始点的瞬态能量爆发
            energy_hit = np.max(onset_env[start_f : start_f + 5])
            
            # 评分因子：跨周期的自相关匹配度
            seg1 = onset_env[start_f : start_f + best_p_frames]
            seg2 = onset_env[start_f + best_p_frames : start_f + 2 * best_p_frames]
            corr = np.corrcoef(seg1, seg2)[0,1] if len(seg1) == len(seg2) else 0
            
            total_weight = (energy_hit * 0.4) + (corr * 0.6)
            
            if total_weight > max_phase_weight:
                max_phase_weight = total_weight
                best_start_f = start_f

        # 4. 物理起点对准：通过能量梯度回溯锁定波形起跳
        final_start_time = self._refine_start_with_gradient(y_component, best_start_f)
        actual_duration = librosa.frames_to_time(best_p_frames, sr=self.sr)

        return {
            "start": float(final_start_time),
            "end": float(final_start_time + actual_duration),
            "duration": float(actual_duration),
            "confidence": float(max_phase_weight)
        }

    def _refine_start_with_gradient(self, y, start_f):
        """精准回溯底鼓起跳瞬间，消除静音偏差"""
        start_sample = int(librosa.frames_to_samples(start_f))
        # 向左回溯 150ms 捕捉物理爆发
        lookback = int(0.15 * self.sr)
        search_zone = y[max(0, start_sample - lookback) : start_sample + int(0.05 * self.sr)]
        
        if len(search_zone) > 0:
            abs_diff = np.abs(np.diff(np.abs(search_zone)))
            gradient_peak = np.argmax(abs_diff)
            pre_peak = search_zone[:gradient_peak]
            zero_crossings = np.where(np.diff(np.sign(pre_peak)))[0]
            
            if len(zero_crossings) > 0:
                refined_sample = max(0, start_sample - lookback + zero_crossings[-1])
            else:
                refined_sample = max(0, start_sample - lookback + gradient_peak - 5)
            return refined_sample / self.sr
        return librosa.frames_to_time(start_f, sr=self.sr)

    def process(self, y_drums, beat_times=None):
        """主处理接口"""
        components = self.separate_drum_components(y_drums)
        results = {}
        duration = len(y_drums) / self.sr
        
        for name, y_comp in components.items():
            # 动态检测信号活性，防止处理纯静音区域
            rms_val = np.sqrt(np.mean(y_comp**2))
            if rms_val < 0.001: # 极低门限确保捕捉微弱信号
                results[name] = {"loop": None, "audio": y_comp}
                continue
                
            loop_info = self.find_minimum_loop(y_comp, duration, component_name=name, beat_times=beat_times)
            results[name] = {"loop": loop_info, "audio": y_comp}
        return results