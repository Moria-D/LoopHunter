import numpy as np
import librosa
import scipy.signal

class DrumLoopExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
        # 精细化滤波频段
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
                sos = scipy.signal.butter(4, [250, 4000], 'band', fs=self.sr, output='sos')
                y_filt = scipy.signal.sosfilt(sos, y_filt)
            elif name == 'cymbals':
                sos = scipy.signal.butter(4, 4000, 'high', fs=self.sr, output='sos')
                y_filt = scipy.signal.sosfilt(sos, y_filt)
            components[name] = y_filt
        return components

    def find_minimum_loop(self, y_component, duration_sec, component_name="unknown", beat_times=None):
        """
        全自动动态嗅探增强版：增加 Snare 变奏保护与 Cymbals 平滑对齐
        """
        # 1. 提取强度包络
        onset_env = librosa.onset.onset_strength(y=y_component, sr=self.sr)
        
        # 针对 Snare 这种需要极高瞬态灵敏度的组件，进行中值滤波锐化包络
        if component_name == 'snare_perc':
            onset_env = librosa.util.normalize(onset_env)
        
        if beat_times is None:
            tempo, beats = librosa.beat.beat_track(y=y_component, sr=self.sr, tightness=100)
            beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        if len(beat_times) < 2: return None
        avg_beat_dur = np.mean(np.diff(beat_times))

        # 2. 动态探测周期
        # 增加 16/32 拍探测，确保捕获 Snare 的长变奏
        possible_mults = [4, 8, 12, 16, 24, 32]
        best_p_frames = int(librosa.time_to_frames(avg_beat_dur * 8, sr=self.sr))
        max_ac_score = -1

        for mult in possible_mults:
            p_test = int(librosa.time_to_frames(avg_beat_dur * mult, sr=self.sr))
            if p_test * 2 > len(onset_env): continue
            
            ac_segment = librosa.autocorrelate(onset_env, max_size=p_test + 1)
            score = ac_segment[p_test]
            
            # 权重补偿：针对 Kick/Snare/Cymbals 赋予长周期 (16拍以上) 显著权重，以保留变奏细节
            bias = 1.4 if component_name in ['kick', 'snare_perc', 'cymbals'] and mult >= 16 else 1.0
            if score * bias > max_ac_score:
                max_ac_score = score * bias
                best_p_frames = p_test

        # 3. 相位自校准：锁定重拍起始点 (Downbeat Priority)
        best_start_f = 0
        max_phase_weight = -1
        
        search_limit = min(32, len(beat_times))
        for i in range(search_limit):
            start_f = librosa.time_to_frames(beat_times[i], sr=self.sr)
            if start_f + best_p_frames * 2 > len(onset_env): break
            
            energy_hit = np.max(onset_env[start_f : start_f + 5])
            
            seg1 = onset_env[start_f : start_f + best_p_frames]
            seg2 = onset_env[start_f + best_p_frames : start_f + 2 * best_p_frames]
            corr = np.corrcoef(seg1, seg2)[0,1] if len(seg1) == len(seg2) else 0
            
            total_weight = (energy_hit * 0.4) + (corr * 0.6)
            if total_weight > max_phase_weight:
                max_phase_weight = total_weight
                best_start_f = start_f

        # 4. 物理起点与终点精准对准 (解决平滑度问题)
        start_time, end_time = self._refine_loop_boundaries(y_component, best_start_f, best_p_frames, component_name)

        return {
            "start": float(start_time),
            "end": float(end_time),
            "duration": float(end_time - start_time),
            "confidence": float(max_phase_weight)
        }

    def _refine_loop_boundaries(self, y, start_f, p_frames, name):
        """
        通过斜率匹配过零点锁定起始和结束位置，确保循环无缝衔接
        """
        start_sample_target = int(librosa.frames_to_samples(start_f))
        end_sample_target = start_sample_target + int(librosa.frames_to_samples(p_frames))
        
        # 定义搜索窗口 (50ms)
        win = int(0.05 * self.sr)
        
        def find_best_zero_cross(center_sample, preferred_slope=None):
            s_idx = max(0, center_sample - win)
            e_idx = min(len(y), center_sample + win)
            region = y[s_idx:e_idx]
            
            # 找到所有过零点
            zero_crossings = np.where(np.diff(np.sign(region)))[0]
            if len(zero_crossings) == 0:
                return center_sample, 1
            
            # 评估每个过零点：距离中心近且斜率一致
            best_zc = zero_crossings[0]
            min_dist = float('inf')
            final_slope = 1
            
            for zc in zero_crossings:
                dist = abs(zc - win)
                slope = np.sign(y[s_idx + zc + 1] - y[s_idx + zc])
                
                # 如果指定了首选斜率，则强制匹配
                if preferred_slope is not None and slope != preferred_slope:
                    dist += win * 2 # 增加惩罚项
                
                if dist < min_dist:
                    min_dist = dist
                    best_zc = zc
                    final_slope = slope
                    
            return s_idx + best_zc, final_slope

        # 1. 确定起始点及其波形斜率
        refined_start, start_slope = find_best_zero_cross(start_sample_target)
        
        # 2. 确定结束点，并强制其斜率与起始点一致，实现平滑衔接
        refined_end, _ = find_best_zero_cross(end_sample_target, preferred_slope=start_slope)
        
        return refined_start / self.sr, refined_end / self.sr

    def process(self, y_drums, beat_times=None):
        """主处理接口"""
        components = self.separate_drum_components(y_drums)
        results = {}
        duration = len(y_drums) / self.sr
        
        for name, y_comp in components.items():
            rms_val = np.sqrt(np.mean(y_comp**2))
            # Cymbals 的弱音尾部需要极低门限以防被截断
            thresh = 0.0003 if name == 'cymbals' else 0.001
            if rms_val < thresh:
                results[name] = {"loop": None, "audio": y_comp}
                continue
                
            loop_info = self.find_minimum_loop(y_comp, duration, name, beat_times)
            results[name] = {"loop": loop_info, "audio": y_comp}
        return results