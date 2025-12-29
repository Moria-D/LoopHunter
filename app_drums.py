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
from utils_bpm import calculate_global_beat_duration

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
# Fix potential state corruption from previous version where tuple was returned
if isinstance(st.session_state.beat_slices, tuple):
    st.session_state.beat_slices = st.session_state.beat_slices[0]

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
    Strict Global Quantum Grid Slicing:
    - Calculates a single global 'beat_duration' (period) using linear regression on beat times.
    - Generates slices strictly on this grid: t = offset + k * period.
    - Ensures all slices (except maybe first/last) have identical duration.
    """
    total_duration = float(total_duration)
    if total_duration <= 0:
        total_duration = float(len(y) / sr)

    # 1. Calculate Global Parameters (Period & Offset)
    period, offset = calculate_global_beat_duration(beat_times, total_duration, bpm_override)
    
    # 2. Generate Grid Points
    # We want to cover from time=0 to time=total_duration
    # Grid formula: t = offset + k * period
    # Find start_k such that offset + k * period >= 0 (or slightly before 0 if close)
    
    if period <= 0: period = 0.5 # Safety
    
    start_k = int(np.floor((0.0 - offset) / period))
    
    slices = []
    sid = 1
    
    # Grid loop
    # We maintain strict adherence to grid points for start/end
    
    current_k = start_k
    
    # Determine first slice start
    # If the first grid point is far after 0, we might need an Intro slice [0, grid_point]
    # But usually offset is chosen to align with the first beat.
    
    # Let's iterate grid points until we exceed total_duration
    while True:
        # Calculate theoretical grid points
        t_start = offset + current_k * period
        t_end = offset + (current_k + 1) * period
        
        # Adjust for file boundaries
        
        # If this "beat" is entirely before 0, skip
        if t_end <= 0.001:
            current_k += 1
            continue
            
        # If start is before 0, clamp to 0 (First slice might be shorter if offset is negative)
        # OR if offset > 0, we might have a gap [0, offset].
        
        # Strategy:
        # If t_start < 0, we clamp start to 0. This slice will be shorter.
        # If t_start > 0 and this is the very first processed slice, we might check if there is a gap [0, t_start]
        # But 'start_k' logic tries to include the point before 0.
        
        # Refined Logic:
        # Let's start from 0.0 explicitly.
        # Find the next grid point > 0.
        # That defines the first segment.
        pass
        break 
    
    # Re-implementation of loop for clarity
    current_time = 0.0
    
    # Find first positive grid boundary
    # k such that offset + k * period > 0
    # If offset=0.1, period=1.0. k=0 -> 0.1. First boundary at 0.1.
    # If offset=-0.1, period=1.0. k=0 -> -0.1. k=1 -> 0.9. First boundary at 0.9? 
    # Wait, if offset=-0.1, the beat started at -0.1. The slice should go -0.1 to 0.9.
    # Clipped to 0.0 to 0.9.
    
    first_boundary_k = int(np.floor((0.001 - offset) / period)) + 1
    next_grid_time = offset + first_boundary_k * period
    
    # If there is a significant gap before the first grid alignment (intro)
    # user request: No explicit Intro slice. Just start first slice at offset if offset > 0.
    # The grid generation logic below handles this if we set current_k correctly.
    
    # If offset > 0, the first grid point at k=0 is at offset.
    # If offset < 0, the first grid point > 0 is at some k.
    
    # Let's ensure strict alignment.
    # We want slices: 
    # Slice 1: [offset, offset + period]
    # Slice 2: [offset + period, offset + 2*period]
    # ...
    # But only if offset >= 0.
    # If offset < 0 (meaning the theoretical grid start is before 0), we want:
    # Slice 1: [0, next_grid_point] ?? No, user said "start from actual sound start".
    # detect_first_transient ensures offset is the actual sound start (>=0).
    # So offset should be >= 0 usually.
    
    # If offset is very small (< 0.01), treat as 0.
    
    current_k = 0
    
    # If offset was calculated via regression (not transient) it might be negative.
    # But our new logic in utils_bpm uses detect_first_transient, so offset ~ start of audio.
    # If beat_times logic prevailed (no audio provided?), offset could be anything.
    
    # Ensure start is not negative
    if offset < 0:
        # Shift offset by periods until >= 0
        shift_k = int(np.ceil((0.0 - offset) / period))
        offset += shift_k * period
    
    # Generate Full Slices
    while True:
        t_start_grid = offset + current_k * period
        t_end_grid = offset + (current_k + 1) * period
        
        # If we are starting effectively at 0 (handled above or first loop)
        real_start = max(0.0, t_start_grid)
        real_end = min(total_duration, t_end_grid)
        
        if real_start >= total_duration - 0.005:
            break
            
        dur = real_end - real_start
        
        if dur > 0.005:
            slices.append({
                "id": sid,
                "start": round(real_start, 3),
                "end": round(real_end, 3),
                "duration": round(dur, 3),
                "label": f"Slice {sid}"
            })
            sid += 1
            
        current_k += 1
        
        if real_end >= total_duration:
            break
            
    return slices, period

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
                slices, period = get_beat_slices(
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
    
    # Try to find the used beat_duration and offset
    # offset is essentially the start of the first beat-aligned slice
    beat_dur_display = 0.0
    offset_display = 0.0
    
    if st.session_state.beat_slices and len(st.session_state.beat_slices) > 0:
         # Find the first slice that isn't labeled "Intro" if possible, 
         # but users just want to know the physical offset of the content.
         # If Slice 1 is 0.2s to 0.7s, offset is 0.2s.
         first_slice = st.session_state.beat_slices[0]
         offset_display = first_slice.get('start', 0.0)
         
         # Take median of first 5 slices duration
         if len(st.session_state.beat_slices) > 1:
            durs = [s['duration'] for s in st.session_state.beat_slices[:5] if 'duration' in s]
            if durs:
                beat_dur_display = np.median(durs)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BPM（估计）", f"{int(round(est_bpm))}")
    c2.metric("Beat Duration", f"{beat_dur_display:.3f}s")
    c3.metric("Offset (Start)", f"{offset_display:.3f}s")
    c4.metric("切片数量", f"{len(st.session_state.beat_slices) if st.session_state.beat_slices else 0}")
    c5.metric("采样率", f"{remixer.sr} Hz")

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
                slices, period = get_beat_slices(
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
    refined_starts = []
    if st.session_state.beat_slices:
        for s in st.session_state.beat_slices:
            if isinstance(s, dict) and 'start' in s:
                if s['start'] < remixer.duration:
                    refined_starts.append(s['start'])
    
    fig_interactive = plot_interactive_waveform(remixer.y, remixer.sr, refined_starts)
    st.plotly_chart(fig_interactive, use_container_width=True)
    
    # --- Slice Previews ---
    st.divider()
    
    total_slices = len(st.session_state.beat_slices) if st.session_state.beat_slices else 0
    display_all = total_slices < 20
    
    if display_all:
        st.subheader(f"🎵 Slice Previews (All {total_slices})")
        st.caption("Preview all slices.")
    else:
        st.subheader("🎵 Slice Previews (First 10)")
        st.caption("Preview individual slices with their specific waveforms.")
    
    if st.session_state.beat_slices:
        if display_all:
            preview_slices = st.session_state.beat_slices
        else:
            preview_slices = st.session_state.beat_slices[:10]
        
        # Display in rows of 5
        rows = [preview_slices[i:i+5] for i in range(0, len(preview_slices), 5)]
        
        for row_items in rows:
            cols = st.columns(5)
            for idx, s in enumerate(row_items):
                with cols[idx]:
                    # Ensure s is a dictionary
                    if not isinstance(s, dict):
                        continue
                        
                    label = s.get('label', f'Slice {idx}')
                    start_t = s.get('start', 0.0)
                    end_t = s.get('end', 0.0)
                    
                    # Card-like container visual
                    st.markdown(f"""
                    <div style="background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #363945; margin-bottom: 10px;">
                        <div style="font-weight: bold; color: #E0E2E6; margin-bottom: 4px;">{label}</div>
                        <div style="font-size: 0.8em; color: #A3A8B8;">{start_t:.2f}s - {end_t:.2f}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                    # Extract audio chunk
                    start_samp = int(start_t * remixer.sr)
                    end_samp = int(end_t * remixer.sr)
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
