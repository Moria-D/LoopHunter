import os
# 解决 OpenMP 库冲突问题（Windows 环境）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from audio_remixer import AudioRemixer
import io, json, tempfile, soundfile as sf

st.set_page_config(layout="wide", page_title="LoopHunter DAW")

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
    .audio-player-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    div[data-testid="stNumberInput"] input { text-align: center; }
    [data-testid="stFileUploader"] > div > div { display: flex; flex-direction: column-reverse; }
    [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] { margin-top: 10px; }
    .stDownloadButton button { height: 3rem; padding-top: 0.4rem; padding-bottom: 0.4rem; }
    h1, h2, h3, p, label { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

st.title("🎛️ Audio Loop & Remix Studio")

if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'timeline' not in st.session_state: st.session_state.timeline = None
if 'final_audio' not in st.session_state: st.session_state.final_audio = None
if 'final_dur' not in st.session_state: st.session_state.final_dur = 0.0
if 'loop_page' not in st.session_state: st.session_state.loop_page = 0
if 'analysis_report' not in st.session_state: st.session_state.analysis_report = None
if 'uploaded_file_data' not in st.session_state: st.session_state.uploaded_file_data = None
if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None
if 'temp_file_path' not in st.session_state: st.session_state.temp_file_path = None

def plot_mini_waveform_with_highlight(y, sr, loop_start, loop_end):
    fig, ax = plt.subplots(figsize=(10, 1.2)) 
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    step = 100
    y_subs = y[::step]
    sr_subs = sr / step
    librosa.display.waveshow(y_subs, sr=sr_subs, ax=ax, color='#444', alpha=0.4)
    s_idx = int(loop_start * sr / step)
    e_idx = int(loop_end * sr / step)
    if e_idx > s_idx:
        times = np.arange(len(y_subs)) / sr_subs
        loop_times = times[s_idx:e_idx]
        loop_chunk = y_subs[s_idx:e_idx]
        ax.plot(loop_times, loop_chunk, color='#3b82f6', linewidth=0.8, alpha=0.9) 
        rect = patches.Rectangle((loop_start, -1), loop_end - loop_start, 2, facecolor='#1f6feb', alpha=0.15, edgecolor=None)
        ax.add_patch(rect)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlim(0, len(y_subs)/sr_subs)
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout(pad=0)
    return fig

def plot_remix_waveform(remix_y, sr, timeline, total_remix_dur):
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.set_xlim(0, total_remix_dur)
    ax.set_xlabel("Time (s)", color='#8b949e')
    ax.tick_params(axis='x', colors='#8b949e')
    ax.set_yticks([])
    
    step = 100
    
    for i, seg in enumerate(timeline):
        start_time = seg['remix_start']
        duration = seg['duration']
        end_time = start_time + duration
        
        label = seg['type']
        color = '#238636' # Default Green (Linear)
        if label == 'Loop Extension': color = '#1f6feb' # Blue
        if seg.get('is_jump'): color = '#d2a8ff' # Purple (Tail/Jump)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        if end_sample > len(remix_y): end_sample = len(remix_y)
        if start_sample >= end_sample: continue
        
        segment_y = remix_y[start_sample:end_sample:step]
        segment_times = np.linspace(start_time, end_time, num=len(segment_y))
        
        ax.plot(segment_times, segment_y, color=color, linewidth=0.8, alpha=0.9)
        ax.axvline(x=start_time, color='white', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.text(start_time, 1.05, f"{start_time:.1f}s", color='white', rotation=0, ha='left', va='bottom', fontsize=8)
        
        if seg.get('xfade', 0) > 0 or seg.get('is_jump'):
             ax.scatter([start_time], [0], color='white', s=15, zorder=10)

        if duration > 2.0:
            lbl_text = "Loop" if "Loop" in label else label
            mid_point = start_time + duration / 2
            ax.text(mid_point, -0.8, lbl_text, color='white', ha='center', va='center', fontsize=8, alpha=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#30363d')
    plt.tight_layout()
    return fig

def plot_daw_style_timeline(remixer, metrics):
    """仿 DAW 风格的多轨对齐可视化"""
    loops_to_show = remixer.loops[:6] 
    fig, ax = plt.subplots(figsize=(12, 1 + len(loops_to_show) * 0.8))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    beat_dur = float(metrics['beat_duration'])  # 确保是 Python float
    # 限制最大显示长度，防止 LCM 过大导致图片拉伸
    max_time = min(float(metrics['cycle_duration']), float(remixer.duration) * 2) 
    
    # 绘制背景网格线
    grid_unit = float(metrics['gcd_beats']) * beat_dur
    for t in np.arange(0, max_time + grid_unit, grid_unit):
        ax.axvline(x=t, color='#30363d', linewidth=0.8, zorder=1, alpha=0.5)

    colors = ['#1f6feb', '#238636', '#d2a8ff', '#e3b341', '#f85149', '#58a6ff']
    for i, loop in enumerate(loops_to_show):
        y_pos = i
        dur = float(loop['duration'])
        loop_start = float(loop['start'])
        # 绘制 Loop 块
        rect = patches.Rectangle((loop_start, y_pos - 0.3), dur, 0.6, 
                                 edgecolor=colors[i % len(colors)], 
                                 facecolor=colors[i % len(colors)], 
                                 alpha=0.7, linewidth=1, zorder=3)
        ax.add_patch(rect)
        beats = int(round(dur / beat_dur))
        ax.text(loop_start + 0.1, y_pos, f"{beats} Beats", color='white', va='center', fontsize=8)

    ax.set_ylim(-0.5, len(loops_to_show))
    ax.set_xlim(0, max_time)
    ax.set_yticks(range(len(loops_to_show)))
    ax.set_yticklabels([f"Loop #{i+1}" for i in range(len(loops_to_show))], color='#8b949e')
    plt.tight_layout()
    return fig

def plot_enhanced_daw_view(remixer, metrics):
    """增强版 DAW 视图：按类型组织轨道，显示波形"""
    # 1. 按类型组织轨道
    tracks = {}
    for l in remixer.loops[:8]:
        t_type = l['type']
        if t_type not in tracks: tracks[t_type] = []
        tracks[t_type].append(l)
    
    track_types = list(tracks.keys())
    fig, ax = plt.subplots(figsize=(14, 1 + len(track_types) * 1.2))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    beat_dur = float(metrics['beat_duration'])
    max_time = min(float(metrics['cycle_duration']), float(remixer.duration) * 1.2)
    
    # 2. 绘制网格
    # 细线：拍 (Beats)
    for b in np.arange(0, max_time, beat_dur):
        ax.axvline(x=b, color='#30363d', linewidth=0.5, alpha=0.3, zorder=0)
    # 粗线：小节/GCD (Bars)
    grid_unit = float(metrics['gcd_beats']) * beat_dur
    for g in np.arange(0, max_time + grid_unit, grid_unit):
        ax.axvline(x=g, color='#484f58', linewidth=1.2, alpha=0.6, zorder=1)
    
    # 3. 逐轨道绘制
    for idx, t_type in enumerate(track_types):
        y_base = idx
        track_info = metrics['type_map'].get(t_type, {"name": "Other", "color": "#8b949e"})
        
        # 绘制轨道背景
        ax.add_patch(patches.Rectangle((0, y_base - 0.4), max_time, 0.8, color='#21262d', alpha=0.15))
        
        for l in tracks[t_type]:
            start, dur = float(l['start']), float(l['duration'])
            if start > max_time: continue
            
            # 绘制 Clip 外框
            color = track_info['color']
            rect = patches.Rectangle((start, y_base - 0.35), dur, 0.7, 
                                     facecolor=color, alpha=0.6, edgecolor=color, linewidth=1.5, zorder=3)
            ax.add_patch(rect)
            
            # --- 核心优化：在 Clip 内部绘制迷你波形 ---
            s_samp, e_samp = int(start * remixer.sr), int((start + dur) * remixer.sr)
            chunk = remixer.y[s_samp:e_samp:100]  # 下采样提高性能
            if len(chunk) > 0:
                t_chunk = np.linspace(start, start + dur, len(chunk))
                # 将波形缩放到轨道高度范围内
                chunk_norm = (chunk / (np.max(np.abs(chunk)) + 1e-6)) * 0.3
                ax.plot(t_chunk, y_base + chunk_norm, color='white', alpha=0.3, linewidth=0.5, zorder=4)
            
            # 标签
            beats = int(round(dur / beat_dur))
            ax.text(start + 0.05, y_base + 0.2, f"{beats}B", color='white', fontsize=7, fontweight='bold', zorder=5)
    
    # 4. 样式美化
    ax.set_xlim(0, max_time)
    ax.set_ylim(-0.6, len(track_types) - 0.4)
    ax.set_yticks(range(len(track_types)))
    ax.set_yticklabels([metrics['type_map'].get(t, {"name":t})['name'] for t in track_types], 
                       color='#c9d1d9', fontsize=10, fontweight='bold')
    
    # 顶部时间轴刻度（按小节显示）
    ax.xaxis.set_major_locator(ticker.MultipleLocator(grid_unit))
    ax.set_xticklabels([f"Bar {int(x/grid_unit)+1}" for x in ax.get_xticks()], color='#8b949e', fontsize=8)
    
    ax.invert_yaxis()
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout()
    return fig

def plot_multitrack_daw(remixer, metrics):
    """仿 FL Studio 多轨对齐视图"""
    # 按类型组织轨道
    tracks = {}
    for l in metrics['processed_loops']:
        # 兼容常规分析和分轨分析：优先使用 stem_type，否则使用 type
        t_type = l.get('stem_type') or l.get('type', 'other')
        if t_type not in tracks: tracks[t_type] = []
        tracks[t_type].append(l)
    
    track_types = list(tracks.keys())
    fig, ax = plt.subplots(figsize=(14, 1 + len(track_types) * 1.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    beat_dur = metrics['beat_duration']
    # 显示范围限制：1个周期或1.2倍时长
    display_limit = min(metrics['cycle_duration'], remixer.duration * 1.2)
    
    # 绘制 GCD 网格线
    grid_unit = metrics['gcd_beats'] * beat_dur
    for t in np.arange(0, display_limit + grid_unit, grid_unit):
        is_bar = int(round(t/grid_unit)) % 4 == 0
        ax.axvline(x=t, color='#484f58' if is_bar else '#30363d', 
                   linewidth=1.2 if is_bar else 0.6, alpha=0.5, zorder=1)
    # 逐轨绘制
    for idx, t_type in enumerate(track_types):
        y_base = idx
        t_info = metrics['type_map'].get(t_type, {"name": t_type, "color": "#58a6ff"})
        ax.add_patch(patches.Rectangle((0, y_base - 0.45), display_limit, 0.9, color='#21262d', alpha=0.2))
        
        for l in tracks[t_type]:
            start, dur = float(l['start']), float(l['duration'])
            if start > display_limit: continue
            
            # 绘制 Clip 外框
            rect = patches.Rectangle((start, y_base - 0.35), dur, 0.7, 
                                     facecolor=t_info['color'], alpha=0.7, edgecolor='white', linewidth=0.5, zorder=3)
            ax.add_patch(rect)
            
            # 绘制迷你波形预览
            s_s, e_s = int(start * remixer.sr), int((start + dur) * remixer.sr)
            chunk = remixer.y[s_s:e_s:100]
            if len(chunk) > 0:
                t_c = np.linspace(start, start + dur, len(chunk))
                ax.plot(t_c, y_base + (chunk/np.max(np.abs(chunk)+1e-6))*0.3, color='white', alpha=0.3, lw=0.5, zorder=4)
            
            ax.text(start + 0.05, y_base + 0.25, f"{l['q_beats']}B", color='white', fontsize=7, fontweight='bold', zorder=5)
    ax.set_xlim(0, display_limit)
    ax.set_yticks(range(len(track_types)))
    ax.set_yticklabels([metrics['type_map'].get(t, {"name":t})['name'].upper() for t in track_types], color='#c9d1d9')
    ax.invert_yaxis()
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout()
    return fig

def plot_multitrack_alignment(remixer, metrics):
    """仿 FL Studio 轨道对齐可视化"""
    # 按 Stem 类型分组
    tracks = {}
    for l in remixer.loops:
        t_type = l.get('stem_type', 'other')
        if t_type not in tracks: tracks[t_type] = []
        tracks[t_type].append(l)
    
    track_types = ['drums', 'bass', 'other', 'vocals']
    fig, ax = plt.subplots(figsize=(14, len(track_types) * 1.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    beat_dur = metrics['beat_duration']
    display_limit = min(metrics['cycle_duration'], remixer.duration * 1.5)
    
    # 绘制 GCD 网格 (小节线)
    grid_unit = metrics['gcd_beats'] * beat_dur
    for t in np.arange(0, display_limit + grid_unit, grid_unit):
        ax.axvline(x=t, color='#30363d', linewidth=0.8, alpha=0.5, zorder=1)
    # 逐轨绘制
    for idx, t_type in enumerate(track_types):
        y_pos = idx
        t_info = metrics['type_map'].get(t_type, {"name": t_type, "color": "#58a6ff"})
        ax.add_patch(patches.Rectangle((0, y_pos - 0.4), display_limit, 0.8, color='#21262d', alpha=0.2))
        
        if t_type in tracks:
            for l in tracks[t_type]:
                start, dur = float(l['start']), float(l['duration'])
                if start > display_limit: continue
                
                # 绘制 Clip 块
                rect = patches.Rectangle((start, y_pos - 0.35), dur, 0.7, 
                                         facecolor=t_info['color'], alpha=0.7, 
                                         edgecolor='white', linewidth=0.5, zorder=3)
                ax.add_patch(rect)
                ax.text(start + 0.1, y_pos, f"{l['q_beats']}B", color='white', fontsize=8, va='center')
    ax.set_xlim(0, display_limit)
    ax.set_yticks(range(len(track_types)))
    ax.set_yticklabels([metrics['type_map'][t]['name'] for t in track_types], color='#8b949e')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

with st.sidebar:
    st.header("1. Upload")
    uploaded_file = st.file_uploader("Audio", type=["mp3", "wav"])
    
    if uploaded_file:
        st.write("")
        if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("Scanning structure & loops..."):
                # 清理旧的临时文件
                if st.session_state.temp_file_path and os.path.exists(st.session_state.temp_file_path):
                    try:
                        os.remove(st.session_state.temp_file_path)
                    except:
                        pass
                
                suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".wav"
                # 保存文件数据以便后续分轨使用
                file_data = uploaded_file.read()
                uploaded_file.seek(0)  # 重置文件指针
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(file_data)
                    tpath = tfile.name
                
                remixer = AudioRemixer(tpath)
                remixer.analyze()
                st.session_state.remixer = remixer
                st.session_state.timeline = None
                st.session_state.final_audio = None
                st.session_state.loop_page = 0 
                st.session_state.uploaded_file_data = file_data
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.temp_file_path = tpath
                
                json_data, text_data = remixer.export_analysis_data(uploaded_file.name)
                st.session_state.analysis_report = {"json": json_data, "text": text_data}
                # 不删除临时文件，保留用于分轨功能
                count = len(remixer.loops)
                if count > 0: st.success(f"Found {count} loops!")
                else: st.warning("No loops found.")
        
        st.write("")
        if st.button("🚀 Separate & Analyze Stems", type="secondary", use_container_width=True):
            if 'remixer' not in st.session_state or st.session_state.remixer is None:
                st.warning("Please analyze audio first!")
            else:
                with st.spinner("Demucs is separating stems (this may take a minute)..."):
                    try:
                        remixer = st.session_state.remixer
                        # 1. 确保临时文件存在（如果不存在，重新创建）
                        if not os.path.exists(remixer.path):
                            if st.session_state.uploaded_file_data is None:
                                st.error("Original file data not found. Please re-upload and analyze the file.")
                                st.stop()
                            
                            # 重新创建临时文件
                            suffix = ".mp3" if st.session_state.uploaded_file_name.endswith(".mp3") else ".wav"
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                                tfile.write(st.session_state.uploaded_file_data)
                                tpath = tfile.name
                            
                            # 更新 remixer 的路径
                            remixer.path = tpath
                            st.session_state.temp_file_path = tpath
                            st.session_state.remixer = remixer
                        
                        # 2. 调用 remixer.separate_stems()
                        try:
                            stems = remixer.separate_stems()
                            st.write(f"✅ Stem separation completed! Found {len(stems)} stems: {list(stems.keys())}")
                        except Exception as e:
                            st.error(f"Error during stem separation: {str(e)}")
                            raise
                        
                        # 3. 调用 remixer.analyze_with_stems(stems)
                        try:
                            remixer.analyze_with_stems(stems)
                            st.write("✅ Analysis completed!")
                        except Exception as e:
                            st.error(f"Error during stem analysis: {str(e)}")
                            raise
                        
                        # 更新 session state
                        st.session_state.remixer = remixer
                        st.session_state.timeline = None
                        st.session_state.final_audio = None
                        
                        metrics = remixer.get_beat_metrics()
                        if metrics:
                            st.success("🎉 Stem separation and analysis completed!")
                        else:
                            st.warning("Analysis completed but no metrics found.")
                    except ImportError as e:
                        error_msg = str(e)
                        if "torchcodec" in error_msg.lower():
                            st.error(f"⚠️ TorchCodec is required but not installed. Please install it with: pip install torchcodec")
                            st.info("💡 Tip: After installing torchcodec, you may need to restart the Streamlit app.")
                        elif "demucs" in error_msg.lower() or ("torch" in error_msg.lower() and "codec" not in error_msg.lower()):
                            st.error(f"⚠️ Demucs is not installed. Please install it with: pip install demucs")
                        else:
                            st.error(f"Import error: {error_msg}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ Error during stem separation: {error_msg}")
                        import traceback
                        with st.expander("🔍 Error Details"):
                            st.code(traceback.format_exc())

    st.divider()
    
    if st.session_state.remixer and st.session_state.analysis_report:
        st.header("2. Analysis Data")
        json_str = json.dumps(st.session_state.analysis_report["json"], indent=4)
        text_str = st.session_state.analysis_report["text"]
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("JSON File", data=json_str, file_name="analysis_data.json", mime="application/json", use_container_width=True)
        with c2:
            st.download_button("User Report", data=text_str, file_name="analysis_report.txt", mime="text/plain", use_container_width=True)
        st.divider()

    if st.session_state.remixer:
        st.header("3. Remix Settings")
        duration = st.session_state.remixer.duration
        target_dur = st.slider("Target Duration (s)", min_value=10, max_value=int(duration*3), value=int(duration), step=1)
        st.write("")
        if st.button("✨ Generate Remix", type="primary", use_container_width=True):
            tl, actual_dur = st.session_state.remixer.plan_multi_loop_remix(target_dur)
            st.session_state.timeline = tl
            with st.spinner("Rendering Remix..."):
                audio = st.session_state.remixer.render_remix(tl)
                if len(audio) > 0:
                    mx = np.max(np.abs(audio))
                    if mx > 0: audio = audio / mx * 0.95
                st.session_state.final_audio = audio
                st.session_state.final_dur = actual_dur

if st.session_state.remixer:
    remixer = st.session_state.remixer
    if remixer.loops:
        # 新增：展示音乐结构分析
        metrics = remixer.get_beat_metrics()
        if metrics:
            # 检查是否有分轨分析结果
            has_stems = any(l.get('stem_type') for l in remixer.loops)
            
            if has_stems:
                # 如果有分轨分析，只显示分轨分析结果
                with st.expander("🎛️ Quantized Musical Structure (Stem-based)", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("BPM", metrics['bpm'])
                    col2.metric("GCD (Atomic Unit)", f"{metrics['gcd_beats']} Beats")
                    col3.metric("LCM (Master Cycle)", f"{metrics['lcm_beats']} Beats")
                    col4.metric("Cycle Time", f"{metrics['cycle_duration']:.1f}s")
                    
                st.caption("📊 Stem-separated tracks with quantized beat alignment")
                st.pyplot(plot_multitrack_alignment(remixer, metrics))
                
                # 新增：单轨 Loop 深度探索 (Top 5 per Track)
                st.write("")
                st.markdown("#### 🎧 Explore Loops per Track (Top 5)")
                
                track_types = ['drums', 'bass', 'other', 'vocals']
                tabs = st.tabs([metrics['type_map'].get(t, {"name":t})['name'] for t in track_types])
                
                for idx, t_type in enumerate(track_types):
                    with tabs[idx]:
                        # 筛选该类型的 Loops 并按分数排序
                        stem_loops = [l for l in remixer.loops if l.get('stem_type') == t_type]
                        stem_loops.sort(key=lambda x: x['score'], reverse=True)
                        top_loops = stem_loops[:5]
                        
                        if not top_loops:
                            st.info(f"No loops found for {t_type}.")
                            continue
                            
                        # 检查音频文件是否存在
                        if not hasattr(remixer, 'stem_paths') or t_type not in remixer.stem_paths:
                            st.error(f"Audio file for {t_type} not found.")
                            continue
                            
                        try:
                            # 加载整个轨道音频（只需加载一次）
                            y_stem, _ = librosa.load(remixer.stem_paths[t_type], sr=remixer.sr)
                            
                            for i, l in enumerate(top_loops):
                                c1, c2, c3 = st.columns([1, 4, 2])
                                with c1:
                                    st.write(f"**#{i+1}**")
                                    if st.button("▶", key=f"play_{t_type}_{i}"):
                                        s_idx = int(l['start'] * remixer.sr)
                                        e_idx = int(l['end'] * remixer.sr)
                                        loop_audio = y_stem[s_idx:e_idx]
                                        # 重复4次以预览循环效果
                                        preview = np.tile(loop_audio, 4)
                                        # 归一化
                                        mx = np.max(np.abs(preview))
                                        if mx > 0: preview = preview / mx * 0.9
                                        
                                        buf = io.BytesIO()
                                        sf.write(buf, preview, remixer.sr, format='WAV')
                                        st.session_state[f'audio_{t_type}_{i}'] = buf.getvalue()
                                
                                with c2:
                                    # 绘制迷你波形
                                    fig, ax = plt.subplots(figsize=(6, 0.8))
                                    fig.patch.set_facecolor('#161b22')
                                    ax.set_facecolor('#161b22')
                                    s_idx = int(l['start'] * remixer.sr)
                                    e_idx = int(l['end'] * remixer.sr)
                                    chunk = y_stem[s_idx:e_idx:50] # 下采样
                                    ax.plot(chunk, color='#3b82f6', linewidth=0.8, alpha=0.9)
                                    ax.axis('off')
                                    plt.tight_layout(pad=0)
                                    st.pyplot(fig)
                                    
                                    if f'audio_{t_type}_{i}' in st.session_state:
                                        st.audio(st.session_state[f'audio_{t_type}_{i}'], format='audio/wav')

                                with c3:
                                    st.markdown(f"**{l['q_beats']} Beats**")
                                    st.caption(f"Score: {l['score']:.2f}")
                                    st.caption(f"{l['start']:.1f}s - {l['end']:.1f}s")
                                st.divider()
                                
                        except Exception as e:
                            st.error(f"Error processing {t_type}: {str(e)}")

                # LCM 混音导出
                st.write("")
                st.markdown("#### 🎹 Generate Infinite Mix")
                c_cycles = st.number_input("Number of Cycles", min_value=2, max_value=16, value=4, step=1)
                
                if st.button("✨ Render LCM Remix", type="primary"):
                    with st.spinner("Rendering multi-track remix..."):
                        remix_audio = remixer.render_lcm_remix(metrics, n_cycles=c_cycles)
                        if remix_audio is not None:
                            buf = io.BytesIO()
                            sf.write(buf, remix_audio, remixer.sr, format='WAV')
                            st.audio(buf.getvalue(), format='audio/wav')
                            st.download_button("Download LCM Remix", buf, "lcm_remix.wav", "audio/wav")
                        else:
                            st.error("Could not render remix. Make sure stems are available.")
            else:
                # 如果没有分轨分析，显示常规分析结果
                with st.expander("📊 Musical Structure & Grid Alignment", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("BPM", metrics['bpm'])
                    col2.metric("GCD (Atomic Unit)", f"{metrics['gcd_beats']} Beats") 
                    col3.metric("LCM (Master Cycle)", f"{metrics['lcm_beats']} Beats")
                    col4.metric("Cycle Time", f"{metrics['cycle_duration']:.1f}s")
                    
                    st.caption("📊 Detected loops organized by type (beats, melody, climax, atmosphere)")
                    st.pyplot(plot_multitrack_daw(remixer, metrics))
        
        total_loops = len(remixer.loops)
        items_per_page = 5
        current_page = st.session_state.loop_page
        start_idx = current_page * items_per_page
        end_idx = min(start_idx + items_per_page, total_loops)
        total_pages = (total_loops + items_per_page - 1) // items_per_page
        
        st.subheader(f"2. Detected Loops ({start_idx+1}-{end_idx} of {total_loops})")
        st.caption("Click play to hear a seamless loop preview (repeated 4x).")
        for i in range(start_idx, end_idx):
            loop = remixer.loops[i]
            with st.container():
                c1, c2, c3 = st.columns([1, 6, 2], gap="small")
                with c1:
                    st.write(""); st.write("") 
                    if st.button(f"▶", key=f"play_{i}"):
                        loop_audio = remixer.generate_loop_preview(loop, repetitions=4)
                        buf = io.BytesIO()
                        sf.write(buf, loop_audio, remixer.sr, format='WAV')
                        st.session_state[f'audio_{i}'] = buf.getvalue()
                with c2:
                    fig = plot_mini_waveform_with_highlight(remixer.y, remixer.sr, float(loop['start']), float(loop['end']))
                    st.pyplot(fig)
                    if f'audio_{i}' in st.session_state:
                        st.audio(st.session_state[f'audio_{i}'], format='audio/wav')
                with c3:
                    st.write("")
                    st.markdown(f"**{loop['duration']:.1f}s**")
                    st.caption(f"{loop['start']:.1f}s - {loop['end']:.1f}s")
                st.markdown("<hr style='margin: 5px 0; opacity: 0.2;'>", unsafe_allow_html=True)
        col_prev, col_input, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("◀ Previous", disabled=(current_page == 0), use_container_width=True):
                st.session_state.loop_page -= 1
                st.rerun()
        with col_input:
            def update_page_number():
                new_page = st.session_state.page_input - 1
                if 0 <= new_page < total_pages: st.session_state.loop_page = new_page
            st.number_input("Jump to Page", min_value=1, max_value=total_pages, value=current_page + 1, step=1, key="page_input", on_change=update_page_number, label_visibility="collapsed")
            st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.8em;'>of {total_pages} pages</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("Next ▶", disabled=(end_idx == total_loops), use_container_width=True):
                st.session_state.loop_page += 1
                st.rerun()
    else:
        st.info("No loops detected.")

    if st.session_state.timeline:
        st.divider()
        st.subheader("4. Remix Result")
        if st.session_state.final_audio is not None:
            fig_wave = plot_remix_waveform(st.session_state.final_audio, st.session_state.remixer.sr, st.session_state.timeline, st.session_state.final_dur)
            st.pyplot(fig_wave)
        else:
            st.warning("Please generate remix first.")
        c_a, c_b = st.columns(2)
        c_a.info(f"Target: {target_dur}s")
        c_b.success(f"Actual: {st.session_state.final_dur:.1f}s")
        if st.session_state.final_audio is not None:
            buf = io.BytesIO()
            sf.write(buf, st.session_state.final_audio, st.session_state.remixer.sr, format='WAV')
            st.audio(buf.getvalue(), format='audio/wav')
            st.download_button("Download Remix WAV", buf, "remix.wav", type="primary", use_container_width=True)

elif not uploaded_file:
    st.info("👋 Upload an audio file to start.")