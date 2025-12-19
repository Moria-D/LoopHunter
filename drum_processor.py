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
        全自动动态嗅探增强版：优化 Cymbals 的长尾衔接和平滑度
        """
        onset_env = librosa.onset.onset_strength(y=y_component, sr=self.sr)
        
        if component_name == 'snare_perc':
            onset_env = librosa.util.normalize(onset_env)
        
        if beat_times is None:
            tempo, beats = librosa.beat.beat_track(y=y_component, sr=self.sr, tightness=100)
            beat_times = librosa.frames_to_time(beats, sr=self.sr)
        
        if len(beat_times) < 2: return None
        avg_beat_dur = np.mean(np.diff(beat_times))

        # 1. 动态探测周期
        possible_mults = [4, 8, 12, 16, 24, 32]
        best_p_frames = int(librosa.time_to_frames(avg_beat_dur * 8, sr=self.sr))
        max_ac_score = -1

        for mult in possible_mults:
            p_test = int(librosa.time_to_frames(avg_beat_dur * mult, sr=self.sr))
            if p_test * 2 > len(onset_env): continue
            
            ac_segment = librosa.autocorrelate(onset_env, max_size=p_test + 1)
            score = ac_segment[p_test]
            
            bias = 1.5 if component_name in ['kick', 'snare_perc'] and mult >= 16 else 1.0
            # 针对 Cymbals 这种持续性声音，给长周期（32拍）更高的优先级以包含完整余音
            if component_name == 'cymbals' and mult >= 16:
                bias = 1.6

            if score * bias > max_ac_score:
                max_ac_score = score * bias
                best_p_frames = p_test

        # 2. 相位自校准
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

        # 3. 物理起点与结束点精准对准（引入余音保护）
        start_time, end_time = self._refine_loop_boundaries(y_component, best_start_f, best_p_frames, component_name)

        return {
            "start": float(start_time),
            "end": float(end_time),
            "duration": float(end_time - start_time),
            "confidence": float(max_phase_weight),
            "needs_crossfade": True # 标记给下游处理器建议使用交叉渐变
        }

    def _refine_loop_boundaries(self, y, start_f, p_frames, name):
        """
        通过斜率匹配、余音检测和过零点锁定确保 Loop 衔接完美
        """
        start_sample_target = int(librosa.frames_to_samples(start_f))
        end_sample_target = start_sample_target + int(librosa.frames_to_samples(p_frames))
        
        win = int(0.06 * self.sr)
        
        def find_best_zero_cross(center_sample, preferred_slope=None, is_end_point=False):
            # 针对 Cymbals 结束点，大幅度向后搜索以寻找衰减后的平稳期
            if is_end_point and name == 'cymbals':
                look_back = win
                look_ahead = win * 3 
            else:
                look_back = win
                look_ahead = win
                
            s_idx = max(0, center_sample - look_back)
            e_idx = min(len(y), center_sample + look_ahead)
            region = y[s_idx:e_idx]
            
            zero_crossings = np.where(np.diff(np.sign(region)))[0]
            if len(zero_crossings) == 0:
                return center_sample, 1
            
            best_zc = zero_crossings[0]
            min_score = float('inf')
            final_slope = 1
            
            for zc in zero_crossings:
                curr_idx = s_idx + zc
                if curr_idx + 1 >= len(y): continue
                
                # 距离目标的距离
                dist = abs(zc - look_back)
                # 这里的波形走向（斜率）
                slope = np.sign(y[curr_idx + 1] - y[curr_idx])
                # 这里的局部能量
                local_energy = np.abs(y[curr_idx]) + np.abs(y[curr_idx+1])
                
                score = dist
                # 惩罚项：如果斜率不匹配
                if preferred_slope is not None and slope != preferred_slope:
                    score += win * 10
                
                # 针对结束点：优先选择能量更小的过零点（即衰减得更彻底的地方）
                if is_end_point:
                    score += local_energy * 1000

                if score < min_score:
                    min_score = score
                    best_zc = zc
                    final_slope = slope
                    
            return s_idx + best_zc, final_slope

        # 1. 确定起始点斜率
        refined_start, start_slope = find_best_zero_cross(start_sample_target, is_end_point=False)
        
        # 2. 寻找结束点，匹配起始斜率且尽可能在能量低谷
        refined_end, _ = find_best_zero_cross(end_sample_target, preferred_slope=start_slope, is_end_point=True)
        
        return refined_start / self.sr, refined_end / self.sr

    def process(self, y_drums, beat_times=None):
        components = self.separate_drum_components(y_drums)
        results = {}
        duration = len(y_drums) / self.sr
        
        for name, y_comp in components.items():
            rms_val = np.sqrt(np.mean(y_comp**2))
            thresh = 0.0002 if name == 'cymbals' else 0.0008
            if rms_val < thresh:
                results[name] = {"loop": None, "audio": y_comp}
                continue
                
            loop_info = self.find_minimum_loop(y_comp, duration, name, beat_times)
            results[name] = {"loop": loop_info, "audio": y_comp}
        return results