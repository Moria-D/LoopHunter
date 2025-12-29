import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import io
import tempfile
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import scipy.signal

# Import our backend modules
from division import AudioRemixer
# from drum_processor import DrumLoopExtractor # No longer needed for slicer only

st.set_page_config(layout="wide", page_title="LoopHunter - BPM Slicer")

st.markdown("""
<style>
    /* 全局深色背景与字体优化 */
    .main { 
        background-color: #0E1117; 
        color: #FAFAFA;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* 按钮样式增强 */
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: 600; 
        border: none;
        padding: 0.6rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    /* 主按钮（Analyze）特殊样式 */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF2B2B 100%);
        box-shadow: 0 2px 4px rgba(255, 75, 75, 0.2);
    }
    
    /* 标题与文字颜色 */
    h1, h2, h3 { color: #F0F2F6 !important; letter-spacing: -0.5px; }
    p, label, .stMarkdown { color: #C4C9D6 !important; }
    
    /* Metric 组件优化 */
    div[data-testid="stMetric"] {
        background-color: #262730;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #363945;
    }
    .stMetricLabel { color: #A3A8B8 !important; font-size: 0.9rem !important; }
    .stMetricValue { color: #4F8BF9 !important; font-weight: 700 !important; }

    /* 下载按钮统一高度 */
    .stDownloadButton button { height: 3.2rem; background-color: #262730; color: #E0E2E6; border: 1px solid #4A4E5A; }
    .stDownloadButton button:hover { border-color: #6C7280; color: #FFFFFF; background-color: #363945; }
    
    /* 分割线颜色 */
    hr { border-color: #363945; margin: 2rem 0; }
    
    /* Expander 样式 */
    .streamlit-expanderHeader { 
        background-color: #262730; 
        border-radius: 6px; 
        color: #E0E2E6;
    }
    
    /* 侧边栏微调 */
    section[data-testid="stSidebar"] {
        background-color: #161920;
    }
</style>
""", unsafe_allow_html=True)

st.title("✂️ LoopHunter - BPM Slicer")
st.caption("AI-powered tool for BPM-based audio slicing.")

if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'beat_slices' not in st.session_state: st.session_state.beat_slices = None
if 'bpm_info' not in st.session_state: st.session_state.bpm_info = None

def estimate_bpm_from_times(times_sec):
    """稳健 BPM 估计：基于切点间隔的中位数（比均值更抗漏拍/噪声）。"""
    if times_sec is None:
        return 0.0
    t = np.array(times_sec, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 0.0
    t = np.sort(np.unique(t))
    if t.size < 2:
        return 0.0
    diffs = np.diff(t)
    # 过滤异常间隔（过小通常是重复点/噪声；过大通常是漏拍/尾奏）
    diffs = diffs[(diffs > 0.12) & (diffs < 2.0)]
    if diffs.size == 0:
        return 0.0
    return float(60.0 / np.median(diffs))

def get_onset_energy(y, sr, time_sec, window_ms=50):
    """计算特定时间点附近的局部能量峰值"""
    center_frame = librosa.time_to_frames(time_sec, sr=sr)
    half_window = int(librosa.time_to_frames(window_ms/1000.0, sr=sr))
    
    # 保护边界
    start_f = max(0, center_frame - half_window)
    end_f = min(len(y), center_frame + half_window) # 注意：这里是音频帧不是onset envelope帧
    
    # 为了简化，我们直接看振幅
    # chunk = y[int(time_sec*sr - sr*0.05) : int(time_sec*sr + sr*0.05)]
    # 但振幅大不一定是 onset，还是得用 onset strength
    return 0 # Placeholder if needed, but we use inline logic below for speed

def estimate_bpm_librosa(y, sr):
    """librosa tempo 估计（整体节奏），对弱拍/漏拍通常更稳。"""
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        if tempo is None or len(tempo) == 0:
            return 0.0
        return float(tempo[0])
    except Exception:
        return 0.0

def estimate_bpm_best(y, sr, bpm_min=75.0, bpm_max=200.0, hop_length=512):
    """
    BPM 估计：使用 JMPerez/beats-audio-api 的算法
    (基于 100-150Hz 低频能量峰值检测与间隔统计)
    
    参数:
    - bpm_min: BPM 下限 (默认 75.0，以避免 134 BPM 被误判为 67)
    - bpm_max: BPM 上限 (默认 200.0)
    """
    try:
        # 1. 滤波：Bandpass 100-150Hz (Lowpass 150 + Highpass 100)
        # Web Audio API 默认 Biquad 是 12dB/oct (2nd order)
        sos_lp = scipy.signal.butter(2, 150, 'low', fs=sr, output='sos')
        y_lp = scipy.signal.sosfilt(sos_lp, y)
        
        sos_hp = scipy.signal.butter(2, 100, 'high', fs=sr, output='sos')
        y_filt = scipy.signal.sosfilt(sos_hp, y_lp)
        
        # 2. 峰值检测 (Get Peaks)
        # 将音频分为 0.5s 的片段，找每段最大值
        part_size = int(sr * 0.5)
        if part_size == 0: return {"bpm": 0.0, "confidence": 0.0, "base_bpm": 0.0, "candidates": []}
        
        parts = len(y_filt) // part_size
        peaks = []
        
        for i in range(parts):
            start = i * part_size
            end = start + part_size
            chunk = y_filt[start:end]
            
            if len(chunk) == 0: continue
            
            # 找最大振幅
            max_idx = np.argmax(np.abs(chunk))
            max_vol = float(np.abs(chunk[max_idx]))
            
            if max_vol > 0:
                peaks.append({
                    'position': start + max_idx,
                    'volume': max_vol
                })
        
        if not peaks:
             return {"bpm": 0.0, "confidence": 0.0, "base_bpm": 0.0, "candidates": []}

        # 按音量降序
        peaks.sort(key=lambda x: x['volume'], reverse=True)
        
        # 取前 50% 最响的
        take_count = max(1, len(peaks) // 2)
        peaks = peaks[:take_count]
        
        # 按位置(时间)重新排序
        peaks.sort(key=lambda x: x['position'])
        
        # 3. 间隔统计 (Get Intervals)
        groups = []
        
        for index, peak in enumerate(peaks):
            # 对比接下来的 10 个峰值
            for i in range(1, 10):
                if index + i >= len(peaks):
                    break
                
                neighbor = peaks[index + i]
                diff_samples = neighbor['position'] - peak['position']
                if diff_samples <= 0: continue
                
                tempo = (60.0 * sr) / diff_samples
                
                # JMPerez 逻辑：归一化到指定范围 (默认 75-200)
                # 如果 bpm_min/bpm_max 参数未指定，则使用 75/200 默认值
                # 原算法是 90-180，这里放宽以支持 80 BPM，并将下限设为 75 以避免 134 被误判为 67
                min_limit = bpm_min if bpm_min > 0 else 75.0
                max_limit = bpm_max if bpm_max > 0 else 200.0
                
                while tempo < min_limit:
                    tempo *= 2
                while tempo > max_limit:
                    tempo /= 2
                    
                tempo = round(tempo)
                
                # 统计
                found = False
                for g in groups:
                    if g['tempo'] == tempo:
                        g['count'] += 1
                        found = True
                        break
                if not found:
                    groups.append({'tempo': tempo, 'count': 1})
        
        if not groups:
            return {"bpm": 0.0, "confidence": 0.0, "base_bpm": 0.0, "candidates": []}

        # 按 count 降序
        groups.sort(key=lambda x: x['count'], reverse=True)
        
        best_group = groups[0]
        best_bpm = float(best_group['tempo'])
        best_count = best_group['count']
        
        # 简单计算置信度
        total_count = sum(g['count'] for g in groups)
        confidence = float(best_count) / total_count if total_count > 0 else 0.0
        
        # 构造 candidates 格式
        candidates = [(float(g['tempo']), float(g['count'])) for g in groups[:5]]
        
        return {
            "bpm": best_bpm,
            "confidence": confidence,
            "base_bpm": best_bpm,
            "candidates": candidates
        }

    except Exception:
        return {"bpm": 0.0, "confidence": 0.0, "base_bpm": 0.0, "candidates": []}

def detect_active_bounds(y, sr, top_db=45):
    """
    估计“有效内容”的起止（去掉开头/结尾的静音或低电平底噪）。
    返回 (start_time, end_time) 秒。
    """
    try:
        intervals = librosa.effects.split(y, top_db=top_db)
        if intervals is None or len(intervals) == 0:
            return 0.0, float(len(y) / sr)
        start_samp = int(intervals[0][0])
        end_samp = int(intervals[-1][1])
        start_t = float(start_samp) / sr
        end_t = float(end_samp) / sr
        # 防御：至少保证有正长度
        if end_t <= start_t:
            return 0.0, float(len(y) / sr)
        return start_t, min(end_t, float(len(y) / sr))
    except Exception:
        return 0.0, float(len(y) / sr)

def refine_time_to_zero_crossing(y, sr, t_sec, window_ms=8):
    """把切点微调到附近过零点/低幅度点，减少切割爆音。"""
    n = len(y)
    if n == 0:
        return float(t_sec)
    idx = int(round(t_sec * sr))
    idx = max(0, min(n - 1, idx))
    w = max(1, int(sr * (window_ms / 1000.0)))
    s = max(0, idx - w)
    e = min(n - 1, idx + w)
    seg = y[s:e+1]
    if seg.size < 3:
        return float(idx) / sr
    # 优先找真正的过零点（符号变化）
    signs = np.sign(seg)
    zc = np.where(np.diff(np.signbit(seg)))[0]
    if zc.size > 0:
        # 选择离中心最近的过零点
        center = idx - s
        best = zc[np.argmin(np.abs(zc - center))]
        return float(s + best) / sr
    # 否则选幅度最小点
    best = int(np.argmin(np.abs(seg)))
    return float(s + best) / sr

def apply_short_fade(x, sr, fade_ms=5):
    """试听用：对切片做极短淡入淡出，进一步避免点击音。"""
    if x is None or len(x) == 0:
        return x
    fade_len = int(sr * (fade_ms / 1000.0))
    fade_len = max(0, min(fade_len, len(x) // 2))
    if fade_len <= 0:
        return x
    x = x.copy()
    win_in = np.linspace(0.0, 1.0, fade_len, endpoint=False)
    win_out = np.linspace(1.0, 0.0, fade_len, endpoint=False)
    x[:fade_len] *= win_in
    x[-fade_len:] *= win_out
    return x

def refine_beat_times(y, sr, beat_times):
    # Calculate onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Detect onsets with backtracking for better transient precision
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    refined = []
    # Optimization: Sort onset_times to speed up search or just iterate (len is small)
    for t in beat_times:
        # Search window +/- 60ms
        diffs = np.abs(onset_times - t)
        if len(diffs) > 0:
            min_idx = np.argmin(diffs)
            if diffs[min_idx] < 0.06:
                refined.append(onset_times[min_idx])
            else:
                refined.append(t)
        else:
            refined.append(t)
    return np.unique(refined) # Remove potential duplicates

def get_beat_slices(y, sr, beat_times, total_duration, bpm_override=None):
    """
    更“好听”的 BPM 切片：
    - 先用能量分割估计有效起止，避免底噪/静音干扰
    - 用 beat_times 推断稳定 period，并生成更规整的 beat 网格（更一致）
    - 每个切点先吸附瞬态，再微调到过零点，减少爆音/截断感
    - 默认丢弃/合并过短的“弱起/尾巴”切片（仍保持按 BPM 网格）
    """
    total_duration = float(total_duration)
    if total_duration <= 0:
        total_duration = float(len(y) / sr)

    # 0) 有效内容范围（更准的全曲起止）
    active_start, active_end = detect_active_bounds(y, sr, top_db=45)
    active_start = max(0.0, min(active_start, total_duration))
    active_end = max(active_start, min(active_end, total_duration))

    # 1) 先做瞬态吸附（避免切在“半山腰”）
    bt = np.array(beat_times, dtype=float) if beat_times is not None else np.array([], dtype=float)
    bt = bt[np.isfinite(bt)]
    bt = bt[(bt >= 0.0) & (bt <= total_duration)]
    bt = np.sort(np.unique(bt))

    if bt.size > 0:
        bt = refine_beat_times(y, sr, bt)
        bt = bt[(bt >= 0.0) & (bt <= total_duration)]
        bt = np.sort(np.unique(bt))

    # 2) 推断 period（优先使用 bpm_override；否则用 beat_times/tempo）
    period = None
    if bpm_override is not None:
        try:
            b = float(bpm_override)
            if np.isfinite(b) and b > 0:
                period = float(60.0 / b)
        except Exception:
            period = None
    if bt.size >= 3:
        diffs = np.diff(bt)
        diffs = diffs[(diffs > 0.12) & (diffs < 2.0)]
        if diffs.size > 0:
            period = float(np.median(diffs))
    if period is None or period <= 0:
        tempo = estimate_bpm_librosa(y, sr)
        if tempo and tempo > 0:
            period = float(60.0 / tempo)
        else:
            # 兜底：假设 120 BPM
            period = 0.5

    # 3) 生成更规整的 beat 网格（让切片长度更一致）
    grid = []
    if bt.size > 0:
        anchor = float(bt[0])
    else:
        anchor = active_start

    # 让网格覆盖有效内容范围（略扩一点，防止边界漏掉）
    start_n = int(np.floor((active_start - anchor) / period)) - 1
    end_n = int(np.ceil((active_end - anchor) / period)) + 1
    for n in range(start_n, end_n + 1):
        t = anchor + n * period
        if active_start - 0.25 * period <= t <= active_end + 0.25 * period:
            grid.append(float(t))
    grid = np.sort(np.unique(np.array(grid, dtype=float)))

            # 3.5) 尝试将网格对齐到检测到的 beat_times（修正累积漂移）
    # 线性网格容易在后面产生累积误差，导致切点偏离（如 slice 10 偏后）。
    # 这里利用 librosa 检测到的 beat_times（通常更贴合音频变化）来修正网格位置。
    if bt.size > 0:
        synced_grid = []
        # 允许最大漂移窗口：周期的一半或固定值，取较小值防止跳拍
        # 0.35 * period 能容忍一定程度的 tempo 变化，同时避免吸附到相邻拍
        sync_window = 0.35 * period if period and period > 0 else 0.15
        
        # 针对末尾部分（slice 15+）可能出现的更大累积漂移，适当放宽末端窗口
        # 或者增加全局同步的强度：不仅仅是单次比对，而是“拉链式”同步
        
        for i, g in enumerate(grid):
            # 找最近的检测 beat
            # 改进：不仅找绝对距离最近的，还要考虑方向性
            # 如果 g 已经明显晚于最近的 beat（即 g > nearest），说明网格偏后了，应该拉回来
            # 如果 g 明显早于最近的 beat，说明网格偏前了，也应该拉过去
            
            idx = (np.abs(bt - g)).argmin()
            nearest = bt[idx]
            dist = abs(nearest - g)
            
            # 动态调整窗口：越靠后的点，允许的漂移可能稍微大一点（因为线性误差会累积）
            # 但也不能无限大，否则会跳拍。这里尝试简单的线性增加权重，或者保持固定但略微放宽。
            current_window = sync_window * (1.0 + 0.05 * i) # 稍微随索引增加一点容忍度
            current_window = min(current_window, 0.45 * period if period else 0.25)

            # 如果在允许范围内，说明检测到了对应的 beat，优先使用检测值（因为它已经包含瞬态对齐）
            if dist < current_window:
                synced_grid.append(nearest)
            else:
                # 即使没有完全匹配上 beat，如果它离前一个 synced 点太近（< 0.5 period），可能是一个错误的中间点
                # 这里简单保留原网格，后续靠 refine_beat_times 微调
                synced_grid.append(g)
        
        # 重新排序并去重
        grid = np.array(synced_grid)
        grid = np.sort(np.unique(grid))

    # NEW: 增加一个额外的强制对齐步骤
    # 有时候同步窗口还是太保守，或者 beat_times 检测本身有微小误差。
    # 这里我们引入一个更激进的策略：如果一个网格点 g 距离 ideal_g (n * period) 偏差超过 10%，
    # 并且附近有更强的 onset 能量峰值，强制吸附到峰值。
    
    # 4) 把网格吸附到最近瞬态（小窗口内），避免机械切割
    # 再次运行 refine 以确保即使是未同步的网格点也能贴合瞬态
    # 并且再次过滤掉超出范围的点
    grid = refine_beat_times(y, sr, grid)
    
    # 4.2) 再次进行更严格的网格修正 (Post-refinement Check)
    # 这是为了解决 Slice 11 这种微小“滑移”到下一个瞬态的问题。
    if period and period > 0:
        strict_grid = []
        if len(grid) > 0:
            strict_grid.append(grid[0])
            for k in range(1, len(grid)):
                prev = strict_grid[-1]
                curr = grid[k]
                diff = curr - prev
                
                # 如果间隔偏大 (1.08x - 1.4x)，很可能是吸附到了稍晚的瞬态
                if 1.08 * period < diff < 1.4 * period:
                     ideal_pos = prev + period
                     # 在 ideal_pos 附近 (±0.1s) 重新搜索最强的 onset 能量
                     # 而不仅仅是依赖 librosa 的 beat_times (因为它可能漏掉或选错)
                     
                     # 定义搜索范围
                     search_start = ideal_pos - 0.1
                     search_end = ideal_pos + 0.1
                     
                     # 转换到帧索引
                     onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                     start_frame = librosa.time_to_frames(search_start, sr=sr)
                     end_frame = librosa.time_to_frames(search_end, sr=sr)
                     
                     if start_frame < end_frame and end_frame < len(onset_env):
                         local_env = onset_env[start_frame:end_frame]
                         if len(local_env) > 0:
                             # 找局部最大能量点
                             max_idx = np.argmax(local_env)
                             best_time = librosa.frames_to_time(start_frame + max_idx, sr=sr)
                             
                             # 如果这个能量点比当前的 curr 明显更早，且更接近理想位置
                             if abs(best_time - ideal_pos) < abs(curr - ideal_pos) and curr - best_time > 0.03:
                                 curr = best_time
                
                strict_grid.append(curr)
            grid = np.array(strict_grid)

    grid = grid[(grid >= 0.0) & (grid <= total_duration)]
    grid = np.sort(np.unique(grid))

    # 4.5) 强制修正：如果某个点与它后面的点间隔过短（< 0.8 * period），且该点未被同步修正，
    # 往往意味着这个点是漂移后的错误点，而后面的点才是正确的下一拍。
    
    # NEW: 专门针对“尾奏起始点”（End Tail Start）的优化
    # 用户反馈尾奏切片（最后一个切片）的开始点容易偏后。
    # 这通常是因为最后一拍的瞬态判定被吸附到了后续的余音上。
    # 我们找到有效范围内 (<= active_end) 的最后两个 grid 点
    valid_indices = np.where((grid >= active_start) & (grid <= active_end))[0]
    if len(valid_indices) >= 2:
        last_idx = valid_indices[-1]
        last_g = grid[last_idx]
        prev_g = grid[valid_indices[-2]]
        
        # 定义搜索窗口：在最后一个网格点的前面寻找更早的起跳点
        # 窗口大小：0.35 * period (约 0.15s - 0.2s)
        win_size = 0.35 * period if period else 0.3
        # 搜索范围：从 (last_g - win_size) 到 (last_g - 0.02)
        # 注意不要退到上一个 beat 之后太近的地方
        search_start = max(prev_g + 0.1, last_g - win_size)
        search_end = last_g - 0.02 
        
        if search_start < search_end:
            # 使用 RMS 能量检测上升沿
            s_samp = int(search_start * sr)
            e_samp = int(search_end * sr)
            if 0 <= s_samp < e_samp < len(y):
                chunk = y[s_samp:e_samp]
                # 计算 RMS (Short-term energy)
                rms = librosa.feature.rms(y=chunk, frame_length=512, hop_length=256)[0]
                
                if len(rms) > 1:
                    # 找变化率最大的点 (Onset Attack)
                    rms_diff = np.diff(rms)
                    if len(rms_diff) > 0:
                        max_diff_idx = np.argmax(rms_diff)
                        max_val = rms_diff[max_diff_idx]
                        
                        # 阈值判定：必须有足够的能量突变 (防止噪音触发)
                        if max_val > 0.002: 
                             onset_time = search_start + librosa.frames_to_time(max_diff_idx, sr=sr, hop_length=256)
                             
                             # 只有当这个新点比原点显著早（>0.03s）时才替换
                             if last_g - onset_time > 0.03:
                                 grid[last_idx] = onset_time
    
    # 尝试检查间隔异常 (漏拍补救)：
    if period and period > 0:
        valid_grid = []
        if len(grid) > 0:
            valid_grid.append(grid[0])
            for k in range(1, len(grid)):
                prev = valid_grid[-1]
                curr = grid[k]
                diff = curr - prev
                
                # 如果间隔明显大于 1.5 倍周期（漏拍），尝试在中间补一个检测到的 beat
                if diff > 1.5 * period:
                    # 找中间有没有漏掉的 beat
                    mid_target = prev + period
                    # 在 mid_target 附近找真实 beat
                    mid_idx = (np.abs(bt - mid_target)).argmin() if len(bt) > 0 else -1
                    if mid_idx != -1:
                        mid_beat = bt[mid_idx]
                        # 如果这个 beat 确实在中间
                        if prev < mid_beat < curr:
                            valid_grid.append(mid_beat)
                
                # NEW: 针对 Slice 11 稍微偏长的情况（1.1 - 1.4 倍周期）
                # 这种情况通常不是完全漏拍，而是吸附到了错误的（较晚的）瞬态
                elif 1.05 * period < diff < 1.45 * period:
                     ideal_next = prev + period
                     # 我们要找的是在 ideal_next 左侧（更早）的一个潜藏 beat
                     # 如果当前的 curr 明显比 ideal_next 晚 (> 0.05s)
                     
                     # 1. 首先尝试在 beat_times 里找
                     better_idx = (np.abs(bt - ideal_next)).argmin() if len(bt) > 0 else -1
                     
                     found_better = False
                     if better_idx != -1:
                         better_beat = bt[better_idx]
                         # 如果 better_beat 在 ideal_next 附近 (±0.08s) 且比 curr 明显早
                         if abs(better_beat - ideal_next) < 0.08 * period and curr - better_beat > 0.04:
                             curr = better_beat
                             found_better = True
                     
                     # 2. 如果 beat_times 里没找到，尝试用能量包络找 (Hard Search)
                     # 很多时候弱拍会被 librosa.beat 忽略，但 onset 能量还在
                     if not found_better:
                         # 搜索窗口：理想位置的前后 10% 周期
                         # 重心稍微前移，因为我们怀疑现在的点晚了
                         search_start = ideal_next - 0.15 * period
                         search_end = ideal_next + 0.05 * period
                         
                         # 计算局部 onset strength
                         s_frame = librosa.time_to_frames(search_start, sr=sr)
                         e_frame = librosa.time_to_frames(search_end, sr=sr)
                         
                         # 还需要整个 envelope，避免重复计算，这里局部算一下或者传入
                         # 为了性能，这里只算局部的 rms 突变可能不够，还是调用 librosa
                         # 注意：这步比较耗时，但只针对异常 slice 运行
                         
                         # 简单方案：找振幅包络的上升沿 (Amplitude Rise)
                         # 提取波形片段
                         s_samp = int(search_start * sr)
                         e_samp = int(search_end * sr)
                         if 0 <= s_samp < e_samp < len(y):
                             chunk = y[s_samp:e_samp]
                             # 计算 RMS 能量曲线
                             hop = 256
                             rms = librosa.feature.rms(y=chunk, frame_length=512, hop_length=hop)[0]
                             # 找 RMS 增量最大的地方 (Onset)
                             if len(rms) > 1:
                                 rms_diff = np.diff(rms)
                                 # 简单的峰值检测
                                 max_d_idx = np.argmax(rms_diff)
                                 if rms_diff[max_d_idx] > 0.005: # 阈值：有一定能量突变
                                     # 换算回时间
                                     onset_time = search_start + librosa.frames_to_time(max_d_idx, sr=sr, hop_length=hop)
                                     
                                     # 如果这个新发现的 onset 确实比 curr 早
                                     if curr - onset_time > 0.04:
                                         curr = onset_time

                valid_grid.append(curr)
            grid = np.array(valid_grid)
    
    # 5) 构造切点：默认从“第一个完整拍”开始（减少很短 slice1）
    cuts = []
    # 找到第一个 >= active_start 的 beat 切点
    grid_in = grid[(grid >= active_start) & (grid <= active_end)]
    if grid_in.size == 0:
        # fallback：至少输出一个整体切片
        return [{
            "id": 1,
            "start": round(active_start, 3),
            "end": round(active_end, 3),
            "duration": round(active_end - active_start, 3),
            "label": "Slice 1"
        }]

    first_cut = float(grid_in[0])
    # 如果 active_start 到 first_cut 太短（弱起/噪声），直接从 first_cut 开始
    if (first_cut - active_start) < 0.5 * period:
        cuts.append(first_cut)
    else:
        cuts.append(active_start)
        cuts.append(first_cut)

    # 中间切点
    for t in grid_in[1:]:
        cuts.append(float(t))

    # 末尾：如果最后残余太短，就合并到上一拍（避免很短的最后一个 slice）
    if len(cuts) >= 2:
        rem = active_end - cuts[-1]
        if rem < 0.25 * period:
            # 合并：移除最后一个 beat 切点，让最后一段更长更自然
            if len(cuts) >= 3:
                cuts.pop()
    cuts.append(active_end)

    # 6) 每个切点微调到过零点（减少点击音）
    refined_cuts = []
    for t in cuts:
        refined_cuts.append(refine_time_to_zero_crossing(y, sr, t, window_ms=8))
    refined_cuts = np.array(refined_cuts, dtype=float)
    refined_cuts = np.sort(np.unique(refined_cuts))

    # 7) 生成 slices
    slices = []
    sid = 1
    for i in range(len(refined_cuts) - 1):
        s = float(refined_cuts[i])
        e = float(refined_cuts[i + 1])
        if e - s < 0.03:
            continue
        slices.append({
            "id": sid,
            "start": round(s, 3),
            "end": round(e, 3),
            "duration": round(e - s, 3),
            "label": f"Slice {sid}"
        })
        sid += 1
    return slices

def generate_fcpxml(slices, filename, sample_rate, total_duration):
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.8">
    <resources>
        <asset id="r1" name="{filename}" src="file://localhost/{filename}" start="0s" duration="{total_duration}s" hasAudio="1" hasVideo="0" />
    </resources>
    <library>
        <event name="Beat Slices">
            <project name="Beat Slices Project">
                <sequence duration="{total_duration}s" format="r1" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="{sample_rate}">
                    <spine>
"""
    for s in slices:
        start = s['start']
        dur = s['duration']
        label = s['label']
        xml_content += f"""                        <asset-clip name="{label}" ref="r1" offset="{start}s" duration="{dur}s" start="{start}s" audioRole="dialogue">
                            <marker start="{start}s" duration="0.01s" value="{label}"/>
                            <adjust-volume amount="0dB">
                                <param name="level" key="Level" value="0dB">
                                    <keyframe time="0s" value="0dB"/>
                                </param>
                            </adjust-volume>
                        </asset-clip>\n"""

    xml_content += """                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""
    return xml_content

def plot_interactive_waveform(y, sr, beat_times):
    # Downsample for performance (target ~10k points max)
    step = max(1, len(y) // 10000)
    y_subs = y[::step]
    x_subs = np.arange(len(y_subs)) * step / sr
    
    fig = go.Figure()
    
    # Waveform
    fig.add_trace(go.Scatter(
        x=x_subs, y=y_subs,
        mode='lines',
        name='Waveform',
        line=dict(color='#00CC96', width=1),
        hoverinfo='x+y'
    ))
    
    # Beat markers (using shapes is faster than individual traces for many lines)
    shapes = []
    # Limit visible vertical lines to avoid browser crash on very long tracks
    # We display lines for the first 500 beats max, or maybe all if not huge
    display_beats = beat_times if len(beat_times) < 1000 else beat_times[::2]
    
    for t in display_beats:
        shapes.append(dict(
            type="line",
            x0=t, x1=t,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="#EF553B", width=1, dash="dot"),
            opacity=0.5
        ))

    # Add highlight for the last slice (Outro/Tail)
    total_duration = len(y) / sr
    if len(beat_times) > 0:
        last_beat = beat_times[-1]
        if last_beat < total_duration:
            # Add a subtle red background for the last segment
            shapes.append(dict(
                type="rect",
                x0=last_beat, 
                x1=total_duration,
                y0=0, y1=1,
                yref="paper",
                fillcolor="#EF553B", 
                opacity=0.1, 
                line_width=0,
            ))
            # Add a text label
            fig.add_annotation(
                x=(last_beat + total_duration) / 2,
                y=0.95,
                yref="paper",
                text="End Tail",
                showarrow=False,
                font=dict(color="#EF553B", size=10)
            )

    fig.update_layout(
        title="Interactive Waveform (Zoom/Pan enabled)",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        dragmode='pan',
        shapes=shapes,
        xaxis=dict(
            rangeslider=dict(visible=True),
            range=[0, min(len(y)/sr, 20)] # Start zoomed in on first 20s
        )
    )
    return fig

def plot_mini_waveform(y):
    """生成微型波形图"""
    # 显式关闭之前的图，防止内存泄漏（虽然 st.pyplot 会处理，但为了安全）
    plt.close('all')
    
    fig, ax = plt.subplots(figsize=(4, 1))
    # 降采样以提高性能
    step = max(1, len(y) // 400)
    y_subs = y[::step]
    
    # 居中绘制
    ax.plot(y_subs, color='#00CC96', linewidth=0.8)
    
    # 锁定 Y 轴范围保持视觉一致性
    ax.set_ylim(-1.1, 1.1)
    ax.axis('off') # 隐藏坐标轴
    
    # 透明背景，适配暗色模式
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig

# --- Sidebar Upload ---
with st.sidebar:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Music File", type=["mp3", "wav"])
    
    if uploaded_file:
        if st.button("🚀 Analyze & Slice", type="primary", use_container_width=True):
            # Progress Bar Setup
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Initializing audio engine...")
                progress_bar.progress(5)
                
                suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(uploaded_file.read())
                    tpath = tfile.name
                
                # 1. Initialize Remixer (Standard Analysis)
                status_text.text("Analyzing rhythm structure (BPM)...")
                progress_bar.progress(50)
                
                remixer = AudioRemixer(tpath)
                remixer.analyze() 
                st.session_state.remixer = remixer
                
                # Generate Slices
                # NEW: Passing y and sr for refinement
                bpm_info = estimate_bpm_best(remixer.y, remixer.sr, bpm_min=75.0, bpm_max=200.0)
                st.session_state.bpm_info = bpm_info
                slices = get_beat_slices(
                    remixer.y,
                    remixer.sr,
                    remixer.beat_times,
                    remixer.duration,
                    bpm_override=bpm_info.get("bpm", None) if isinstance(bpm_info, dict) else None
                )
                st.session_state.beat_slices = slices
                
                progress_bar.progress(100)
                
                # SKIP DRUM EXTRACTION (Optimization since tab is removed)
                # status_text.text("Separating stems for drum processing...")
                # ...
                
                os.remove(tpath)
                status_text.empty()
                progress_bar.empty()
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                if 'tpath' in locals() and os.path.exists(tpath):
                    os.remove(tpath)

# --- Main Content ---
if not uploaded_file:
    st.info("👋 Upload an audio file to start.")
    
elif st.session_state.remixer:
    remixer = st.session_state.remixer
    
    # --- Info Header ---
    st.divider()
    
    # Calculate Audio Stats（避免“自洽陷阱”：BPM 不再根据 slices 反推，而是独立估计后再驱动切片）
    slice_starts = [s.get('start') for s in (st.session_state.beat_slices or []) if isinstance(s, dict)]
    bpm_from_slices = estimate_bpm_from_times(slice_starts) if len(slice_starts) > 2 else 0.0
    bpm_lib = estimate_bpm_librosa(remixer.y, remixer.sr)

    bpm_info = st.session_state.bpm_info
    if not isinstance(bpm_info, dict) or bpm_info.get("bpm", 0) <= 0:
        bpm_info = estimate_bpm_best(remixer.y, remixer.sr, bpm_min=75.0, bpm_max=200.0)
        st.session_state.bpm_info = bpm_info

    est_bpm = float(bpm_info.get("bpm", 0.0) or 0.0)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BPM（估计）", f"{est_bpm:.1f}")
    c2.metric("总时长", f"{remixer.duration:.2f}s")
    c3.metric("切片数量", f"{len(st.session_state.beat_slices) if st.session_state.beat_slices else 0}")
    c4.metric("采样率", f"{remixer.sr} Hz")

    # 手动 BPM 修正
    with st.expander("🛠️ 手动修正 BPM / 重新切片", expanded=False):
        manual_bpm = st.number_input(
            "输入 BPM 数值 (修改后将强制重算切片)", 
            value=est_bpm, 
            min_value=10.0, 
            max_value=300.0, 
            step=0.1,
            format="%.1f"
        )
        
        if st.button("🔄 按此 BPM 重新切片", use_container_width=True):
            with st.spinner(f"正在按 BPM {manual_bpm} 重新生成切片..."):
                # Update session BPM info to force the override
                st.session_state.bpm_info = {"bpm": manual_bpm, "confidence": 1.0, "base_bpm": manual_bpm, "candidates": []}
                # Re-run slicing
                slices = get_beat_slices(
                    remixer.y,
                    remixer.sr,
                    remixer.beat_times,
                    remixer.duration,
                    bpm_override=manual_bpm
                )
                st.session_state.beat_slices = slices
                st.rerun()

    st.divider()
    
    # --- BPM Slicer Visualization (No Tabs) ---
    st.subheader("📊 BPM-Based Slicing Visualization")
    
    # Interactive Plotly Waveform
    # NOTE: beat_times passed to plot should ideally be the REFINED start times from slices
    # to match what the user sees in the table.
    # Extract start times from slices for plotting consistency
    refined_starts = [s['start'] for s in st.session_state.beat_slices if s['start'] < remixer.duration]
    
    fig_interactive = plot_interactive_waveform(remixer.y, remixer.sr, refined_starts)
    st.plotly_chart(fig_interactive, use_container_width=True)
    
    # --- Slice Previews ---
    st.divider()
    st.subheader("🎵 Slice Previews (First 10)")
    st.caption("Preview individual slices with their specific waveforms.")
    
    if st.session_state.beat_slices:
        preview_slices = st.session_state.beat_slices[:10]
        
        # Display in rows of 5
        rows = [preview_slices[i:i+5] for i in range(0, len(preview_slices), 5)]
        
        for row_items in rows:
            cols = st.columns(5)
            for idx, s in enumerate(row_items):
                with cols[idx]:
                    # Card-like container visual
                    st.markdown(f"""
                    <div style="background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #363945; margin-bottom: 10px;">
                        <div style="font-weight: bold; color: #E0E2E6; margin-bottom: 4px;">{s['label']}</div>
                        <div style="font-size: 0.8em; color: #A3A8B8;">{s['start']:.2f}s - {s['end']:.2f}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                    # Extract audio chunk
                    start_samp = int(s['start'] * remixer.sr)
                    end_samp = int(s['end'] * remixer.sr)
                    end_samp = min(end_samp, len(remixer.y))
                    
                    if start_samp < end_samp:
                        chunk = remixer.y[start_samp:end_samp]
                        # Normalize for preview
                        mx = np.max(np.abs(chunk))
                        if mx > 0: chunk = chunk / mx * 0.95
                        
                        # Show waveform
                        st.pyplot(plot_mini_waveform(chunk), use_container_width=True, transparent=True)

                        # Short fades to avoid clicks
                        chunk = apply_short_fade(chunk, remixer.sr, fade_ms=5)
                        
                        buf = io.BytesIO()
                        sf.write(buf, chunk, remixer.sr, format='WAV')
                        st.audio(buf.getvalue(), format='audio/wav')
                    
            # Spacer between rows
            st.write("")
    
    st.divider()

    # Download Section
    st.markdown("### 💾 Export All Slices")
    col_d1, col_d2, col_d3 = st.columns(3)
    
    if st.session_state.beat_slices:
        slices_data = st.session_state.beat_slices
        df_slices = pd.DataFrame(slices_data)
        
        # JSON
        json_str = json.dumps(slices_data, indent=2)
        col_d1.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name="slices.json",
            mime="application/json"
        )
        
        # Excel (with Fallback)
        excel_buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_slices.to_excel(writer, index=False, sheet_name='Slices')
            col_d2.download_button(
                label="📥 Download Excel",
                data=excel_buffer.getvalue(),
                file_name="slices.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            # Fallback CSV
            csv_data = df_slices.to_csv(index=False).encode('utf-8')
            col_d2.download_button(
                label="📥 Download CSV (Excel Error)",
                data=csv_data,
                file_name="slices.csv",
                mime="text/csv"
            )
        
        # FCPXML
        fcpxml_str = generate_fcpxml(slices_data, uploaded_file.name, int(remixer.sr), remixer.duration)
        col_d3.download_button(
            label="📥 Download FCPXML",
            data=fcpxml_str,
            file_name="slices.fcpxml",
            mime="text/xml", # FCP往往识别 .fcpxml 后缀，mime 用 text/xml 兼容性较好
            help="可以直接拖入 Final Cut Pro 的时间线。"
        )
        
        with st.expander("View Slice Data Table"):
            st.dataframe(df_slices, use_container_width=True)
