import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
import io
import soundfile as sf
import base64
import os
import tempfile
from division import LoopHunter  # 引用优化后的核心逻辑

st.set_page_config(layout="wide", page_title="LoopHunter Pro")

st.title("🎵 LoopHunter Pro - Structure-Aware Remixing")
st.markdown("""
> **Logic Upgrade**: Now featuring **Downbeat Detection**, **Energy-Based Structure Planning**, and **Equal-Power Crossfading**.
""")

# --- Sidebar ---
st.sidebar.header("1. Source")
uploaded_file = st.sidebar.file_uploader("Upload Audio (MP3/WAV)", type=["mp3", "wav"])

st.sidebar.header("2. Remix Settings")
target_duration = st.sidebar.slider("Target Duration (s)", 30, 300, 60, step=10)
# xfade_ms = st.sidebar.slider("Crossfade Length (ms)", 10, 200, 50, step=10) # Removed manual slider
generate_btn = st.sidebar.button("Generate Remixes")

# --- State Management ---
if 'hunter' not in st.session_state: st.session_state.hunter = None
if 'loops' not in st.session_state: st.session_state.loops = None
if 'remixes' not in st.session_state: st.session_state.remixes = None

# --- Audio Processing Logic ---
def equal_power_crossfade(seg1, seg2, sr, fade_ms):
    """
    等功率交叉淡化 (Equal Power Crossfade)
    保证连接处音量不塌陷，不断层。
    """
    fade_len = int((fade_ms / 1000.0) * sr)
    
    # 如果片段太短，无法淡化，直接拼接
    if len(seg1) < fade_len or len(seg2) < fade_len:
        return np.concatenate((seg1, seg2))
    
    # 生成正弦/余弦曲线 (平方和为1，功率恒定)
    t = np.linspace(0, np.pi/2, fade_len)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    
    # seg1 的尾部淡出
    seg1_main = seg1[:-fade_len]
    seg1_tail = seg1[-fade_len:] * fade_out
    
    # seg2 的头部淡入
    seg2_head = seg2[:fade_len] * fade_in
    seg2_main = seg2[fade_len:]
    
    # 叠加重叠部分
    overlap = seg1_tail + seg2_head
    
    return np.concatenate((seg1_main, overlap, seg2_main))

def stitch_remix(y, sr, timeline):
    """
    根据 Timeline 组装音频，应用 Crossfade
    """
    if not timeline: return np.array([])
    
    # 处理第一个片段
    first = timeline[0]
    s, e = int(first['source_start'] * sr), int(first['source_end'] * sr)
    current_audio = y[s:e]
    
    # 对第一个片段开头做极短的淡入，防止 Click
    tiny_fade = int(0.01 * sr)
    if len(current_audio) > tiny_fade:
        current_audio[:tiny_fade] *= np.linspace(0, 1, tiny_fade)
        
    final_parts = [current_audio]
    
    for i in range(1, len(timeline)):
        seg = timeline[i]
        s, e = int(seg['source_start'] * sr), int(seg['source_end'] * sr)
        next_audio = y[s:e]
        
        # 取出上一段音频（这里为了性能，我们只在最后合并，
        # 但为了做Crossfade，我们需要操作上一段的尾部）
        prev_audio = final_parts.pop()
        
        # 获取该段落的推荐 Crossfade (默认 30ms)
        fade_ms = seg.get('xfade_ms', 30)
        
        # 执行交叉淡化
        merged = equal_power_crossfade(prev_audio, next_audio, sr, fade_ms)
        final_parts.append(merged)
        
    # 合并所有
    full_audio = np.concatenate(final_parts)
    
    # 结尾淡出
    if len(full_audio) > tiny_fade:
        full_audio[-tiny_fade:] *= np.linspace(1, 0, tiny_fade)
        
    # Normalize
    max_val = np.max(np.abs(full_audio))
    if max_val > 0: full_audio /= max_val
    
    return full_audio

def process_file(file_obj):
    with st.spinner("Analyzing Audio Structure & Downbeats..."):
        # 保存临时文件供 librosa 读取
        suffix = ".mp3" if file_obj.name.endswith(".mp3") else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(file_obj.read())
            tpath = tfile.name
        
        try:
            hunter = LoopHunter(tpath)
            json_str = hunter.export_json()
            
            st.session_state.hunter = hunter
            st.session_state.loops = json.loads(json_str)
            st.session_state.remixes = None # Reset old remixes
        finally:
            os.remove(tpath)

# --- UI Layout ---

if uploaded_file:
    if st.sidebar.button("Analyze Audio"):
        process_file(uploaded_file)

if generate_btn and st.session_state.hunter:
    with st.spinner("Planning & Stitching Remixes..."):
        remixes = st.session_state.hunter.generate_remixes(target_duration, top_n=3)
        st.session_state.remixes = remixes

if st.session_state.hunter and st.session_state.loops:
    data = st.session_state.loops
    y = st.session_state.hunter.y
    sr = st.session_state.hunter.sr
    
    tab1, tab2 = st.tabs(["🔍 Analysis & Loops", "🎹 Remix Studio"])
    
    with tab1:
        st.subheader("Detected Musical Loops")
        st.caption("Loops are automatically aligned to the nearest Downbeat (Bar 1).")
        
        # Waveform Plot
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, alpha=0.3, ax=ax, color='gray')
        colors = {"chorus": "red", "verse": "green", "intro": "cyan", "outro": "magenta", "breakdown": "purple", "melody": "blue"}
        
        for loop in data['looping_points']:
            c = colors.get(loop['type'], 'orange')
            ax.axvspan(loop['start_position'], loop['start_position']+loop['duration'], color=c, alpha=0.2)
        st.pyplot(fig)
        
        # List
        for i, loop in enumerate(data['looping_points']):
            with st.expander(f"#{i+1} - {loop['type'].upper()} ({loop['bars']} Bars) - Score: {loop['score']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Time:** {loop['start_position']}s - {loop['start_position']+loop['duration']:.2f}s")
                    st.progress(min(loop['score'], 1.0))
                with col2:
                    # Preview Audio
                    s, e = int(loop['start_position']*sr), int((loop['start_position']+loop['duration'])*sr)
                    buf = io.BytesIO()
                    sf.write(buf, y[s:e], sr, format='WAV')
                    st.audio(buf.getvalue(), format='audio/wav')

    with tab2:
        st.subheader("Auto-Generated Remixes")
        
        if st.session_state.remixes:
            for remix in st.session_state.remixes:
                st.markdown("---")
                c1, c2 = st.columns([1, 4])
                
                with c1:
                    st.markdown(f"### Option {remix['rank']}")
                    st.markdown(f"⏱ **{remix['actual_duration']}s**")
                    
                with c2:
                    st.markdown("**Arrangement Structure:**")
                    # Visual Timeline
                    html = "<div style='display:flex; flex-wrap:nowrap; overflow-x:auto; gap:4px; padding-bottom:10px;'>"
                    for i, seg in enumerate(remix['timeline']):
                        c = colors.get(seg['type'], "#555")
                        # 动态计算宽度，但限制最小和最大值
                        width = max(50, min(150, int(seg['duration'] * 15)))
                        
                        # 添加起止时间显示
                        start_t = seg['remix_start']
                        end_t = seg['remix_end']
                        
                        html += f"""
                        <div style='background:{c}; color:white; min-width:{width}px; height:70px; 
                             border-radius:4px; font-size:11px; padding:4px; flex-shrink:0; 
                             display:flex; flex-direction:column; justify-content:center; align-items:center;
                             box-shadow: 0 2px 4px rgba(0,0,0,0.2); position: relative;'>
                             <b style='font-size:12px; margin-bottom:2px;'>{seg['type'].upper()}</b>
                             <span>{seg['duration']}s</span>
                             <div style='font-size:9px; opacity:0.8; margin-top:4px; border-top:1px solid rgba(255,255,255,0.3); width:90%; text-align:center;'>
                               {start_t:.1f}s - {end_t:.1f}s
                             </div>
                             <div style='position:absolute; top:2px; left:4px; font-size:9px; opacity:0.6;'>#{i+1}</div>
                        </div>
                        """
                        # 在非最后一个元素后添加小箭头视觉
                        if i < len(remix['timeline']) - 1:
                            html += "<div style='display:flex; align-items:center; color:#888; font-size:10px;'>▶</div>"
                            
                    html += "</div>"
                    st.markdown(html, unsafe_allow_html=True)
                    
                    # Generate Audio on Demand
                    audio_arr = stitch_remix(y, sr, remix['timeline'])
                    buf = io.BytesIO()
                    sf.write(buf, audio_arr, sr, format='WAV')
                    st.audio(buf.getvalue(), format='audio/wav')
        else:
            st.info("Adjust settings in the sidebar and click 'Generate Remixes'.")

else:
    st.info("👋 Upload a music file to begin.")