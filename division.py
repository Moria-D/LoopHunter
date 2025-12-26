import librosa
import numpy as np
import warnings
import torch
import json
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist

try:
    from demucs import pretrained
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except:
    DEMUCS_AVAILABLE = False

warnings.filterwarnings('ignore')

class AudioRemixer:
    def __init__(self, audio_path, sr=22050):
        self.path, self.sr = audio_path, sr
        print(f"Loading audio: {audio_path}...")
        self.y, _ = librosa.load(audio_path, sr=sr)
        self.duration = librosa.get_duration(y=self.y, sr=sr)
        self.beat_times, self.beat_features, self.loops = None, None, []
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=sr)
        self.global_rms = np.sqrt(np.mean(self.y**2))

    def _update_global_stats(self):
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.global_rms = np.sqrt(np.mean(self.y**2))

    def _refine_cut_point(self, time_sec):
        """精准过零点锁定防止爆音"""
        c_s = int(time_sec * self.sr)
        rad = librosa.samples_to_frames(int(0.05 * self.sr))
        f_c = librosa.samples_to_frames(c_s)
        f_s, f_e = max(0, f_c - rad), min(len(self.onset_env), f_c + rad)
        if f_s < f_e:
            target = librosa.frames_to_samples(f_s + np.argmax(self.onset_env[f_s:f_e]))
            target = max(0, target - int(0.01 * self.sr))
        else: target = c_s
        chunk = self.y[max(0, target-200):min(len(self.y), target+200)]
        zcr = np.where(np.diff(np.signbit(chunk)))[0]
        return max(0, target-200) + zcr[np.argmin(np.abs(zcr - (target - max(0, target-200))))] if len(zcr) > 0 else target

    def _classify_segment(self, start, end):
        """Lofi 自适应分类：通过频谱平坦度识别干扰音"""
        chunk = self.y[int(start*self.sr):int(end*self.sr)]
        if len(chunk) == 0: return "melody"
        flat = np.mean(librosa.feature.spectral_flatness(y=chunk))
        if flat > 0.15: return "atmosphere"
        ratio = np.sqrt(np.mean(chunk**2)) / (self.global_rms + 1e-6)
        return "climax" if ratio > 1.1 else ("atmosphere" if ratio < 0.6 else "melody")

    def analyze(self):
        """DC 去除与归一化逻辑"""
        self.y = self.y - np.mean(self.y)
        if np.max(np.abs(self.y)) > 0:
            self.y = self.y / np.max(np.abs(self.y))
        self._update_global_stats()
        
        y_h, y_p = librosa.effects.hpss(self.y)
        _, beat_f = librosa.beat.beat_track(y=y_p, sr=self.sr)
        self.beat_times = librosa.frames_to_time(beat_f, sr=self.sr)
        
        chroma = librosa.util.normalize(librosa.util.sync(librosa.feature.chroma_cqt(y=y_h, sr=self.sr), beat_f), axis=1)
        mfcc = librosa.util.normalize(librosa.util.sync(librosa.feature.mfcc(y=y_p, sr=self.sr, n_mfcc=13), beat_f), axis=1)
        self.beat_features = np.vstack([chroma, mfcc]).T
        
        R = librosa.segment.recurrence_matrix(librosa.feature.stack_memory(self.beat_features.T, n_steps=4), mode='affinity', sym=True)
        for lag in range(8, len(beat_f)//2):
            diag = median_filter(np.diagonal(R, offset=lag), size=3)
            idx = np.where(diag > 0.8)[0]
            if len(idx) >= 8:
                t_s, t_e = self.beat_times[idx[0]], self.beat_times[idx[0]+lag]
                if not any(abs(l['start'] - t_s) < 1.0 for l in self.loops):
                    self.loops.append({"start": t_s, "end": t_e, "duration": t_e-t_s, "score": np.mean(diag[idx]), "type": self._classify_segment(t_s, t_e)})
        self.loops = sorted(self.loops, key=lambda x: x['score'], reverse=True)

    def analyze_stems(self):
        if DEMUCS_AVAILABLE:
            try:
                model = pretrained.get_model('htdemucs').to("cpu")
                wav = torch.tensor(librosa.resample(self.y, orig_sr=self.sr, target_sr=44100)).float().unsqueeze(0).repeat(2,1).unsqueeze(0)
                sources = apply_model(model, wav)[0]
                stems = {n: librosa.resample(sources[i].mean(0).cpu().numpy(), orig_sr=44100, target_sr=self.sr)[:len(self.y)] for i, n in enumerate(["drums", "bass", "other", "vocals"])}
                return {"drums": stems["drums"], "bass": stems["bass"], "instruments": stems["other"]}
            except: pass
        y_h, y_p = librosa.effects.hpss(self.y)
        return {"drums": y_p, "bass": y_h, "instruments": y_h}

    def render_remix(self, timeline):
        out = np.zeros(int(sum([s['duration'] for s in timeline])*self.sr) + int(self.sr))
        cur = 0
        for i, seg in enumerate(timeline):
            s_idx, e_idx = self._refine_cut_point(seg['source_start']), self._refine_cut_point(seg['source_end'])
            chunk = self.y[s_idx:e_idx].copy()
            f_l = int((seg.get('xfade', 0)/1000)*self.sr)
            if i == 0 or f_l == 0:
                out[cur:cur+len(chunk)] = chunk; cur += len(chunk)
            else:
                ov = cur - f_l; n = min(len(out[ov:cur]), len(chunk[:f_l]))
                lin = np.linspace(0, 1, n)
                out[ov:ov+n] = out[ov:ov+n]*np.cos(lin*np.pi/2) + chunk[:n]*np.sin(lin*np.pi/2)
                out[ov+n:ov+n+len(chunk)-n] = chunk[n:]; cur += len(chunk)-f_l
        return out[:cur]

    def plan_multi_loop_remix(self, target):
        if target < self.duration: return [{"source_start":0, "source_end":target, "duration":target, "type":"Linear", "remix_start":0}], target
        active = sorted(self.loops, key=lambda x: x['score'], reverse=True)[:3]
        for l in active: l['repeats'] = 0
        diff = target - self.duration
        while diff > 0 and active:
            idx = np.argmax([l['score']/(l['repeats']+1) for l in active])
            active[idx]['repeats'] += 1; diff -= active[idx]['duration']
        timeline, cur, src_cur = [], 0.0, 0.0
        for l in sorted(active, key=lambda x: x['start']):
            if src_cur < l['end']:
                d = l['end'] - src_cur; timeline.append({"source_start":src_cur, "source_end":l['end'], "duration":d, "type":"Linear", "remix_start":cur}); cur += d; src_cur = l['end']
            for _ in range(l['repeats']):
                timeline.append({"source_start":l['start'], "source_end":l['end'], "duration":l['duration'], "type":"Loop Extension", "xfade":30, "remix_start":cur}); cur += l['duration']
        if src_cur < self.duration: timeline.append({"source_start":src_cur, "source_end":self.duration, "duration":self.duration-src_cur, "type":"Outro", "remix_start":cur})
        return timeline, cur