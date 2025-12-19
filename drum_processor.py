import numpy as np
import librosa
import scipy.signal

class DrumLoopExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
        self.bands = {
            'kick': (20, 150),
            'snare_perc': (200, 3500), 
            'cymbals': (3500, None)    
        }

    def separate_drum_components(self, y_drums):
        """保持原有的滤波分轨功能"""
        components = {}
        if y_drums.ndim > 1:
            y_drums = np.mean(y_drums, axis=0)
            
        for name, (low, high) in self.bands.items():
            if name == 'kick':
                sos = scipy.signal.butter(4, 150, 'low', fs=self.sr, output='sos')
            elif name == 'snare_perc':
                sos = scipy.signal.butter(4, [250, 4000], 'band', fs=self.sr, output='sos')
            else: # cymbals
                sos = scipy.signal.butter(4, 4000, 'high', fs=self.sr, output='sos')
            components[name] = scipy.signal.sosfilt(sos, y_drums)
        return components

    def find_real_segment_boundary(self, onset_env, start_f, expected_p_f):
        """动态寻找真实的循环边界，解决时长死板问题"""
        search_range = int(expected_p_f * 0.15) 
        ideal_end = start_f + expected_p_f
        search_start = max(0, ideal_end - search_range)
        search_end = min(len(onset_env), ideal_end + search_range)
        
        if search_start >= search_end:
            return ideal_end
        # 在预期位置附近寻找最强的起始点
        return search_start + np.argmax(onset_env[search_start:search_end])

    def process(self, y_drums, beat_times=None):
        """识别真实的 Intro -> 可变 Loops -> Outro 序列"""
        components = self.separate_drum_components(y_drums)
        results = {}
        total_dur = len(y_drums) / self.sr
        
        if beat_times is None or len(beat_times) < 4:
            tempo, beats = librosa.beat.beat_track(y=y_drums, sr=self.sr)
            beat_times = librosa.frames_to_time(beats, sr=self.sr)

        avg_beat_dur = np.mean(np.diff(beat_times))
        p_beats = 8 # 标准循环长度(拍)

        for name, y_comp in components.items():
            onset_env = librosa.onset.onset_strength(y=y_comp, sr=self.sr)
            mse = librosa.feature.rms(y=y_comp, frame_length=2048, hop_length=512)[0]
            thresh = np.max(mse) * 0.05
            active = np.where(mse > thresh)[0]
            
            if len(active) == 0:
                results[name] = {"markers": [], "audio": y_comp}
                continue

            first_f = int(librosa.time_to_frames(librosa.frames_to_time(active[0], hop_length=512), sr=self.sr))
            last_f = int(librosa.time_to_frames(librosa.frames_to_time(active[-1], hop_length=512), sr=self.sr))
            markers = []
            
            # 1. 标注 Intro
            if first_f > librosa.time_to_frames(0.1, sr=self.sr):
                intro_end = librosa.frames_to_time(first_f, sr=self.sr)
                markers.append({"Start Time (s)": 0.0, "End Time (s)": round(intro_end, 3), 
                                "Duration (s)": round(intro_end, 3), "Type": "intro"})

            # 2. 动态循环识别
            curr_f = first_f
            expected_p_f = int(librosa.time_to_frames(avg_beat_dur * p_beats, sr=self.sr))
            idx = 1
            while curr_f + expected_p_f < last_f:
                next_f = self.find_real_segment_boundary(onset_env, curr_f, expected_p_f)
                start_t, end_t = librosa.frames_to_time(curr_f, sr=self.sr), librosa.frames_to_time(next_f, sr=self.sr)
                markers.append({"Start Time (s)": round(start_t, 3), "End Time (s)": round(end_t, 3), 
                                "Duration (s)": round(end_t - start_t, 3), "Type": f"{name}_loop_{idx}"})
                curr_f, idx = next_f, idx + 1
            
            # 3. 标注 Outro
            outro_start = librosa.frames_to_time(curr_f, sr=self.sr)
            markers.append({"Start Time (s)": round(outro_start, 3), "End Time (s)": round(total_dur, 3), 
                            "Duration (s)": round(total_dur - outro_start, 3), "Type": "outro"})

            results[name] = {"markers": markers, "audio": y_comp}
        return results