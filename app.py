import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import io
import os
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from division import AudioRemixer

st.set_page_config(layout="wide", page_title="LoopHunter - Audjust Style")

# 注入 CSS 以模仿 Audjust 的深色/霓虹风格
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stButton>button { 
        width: 100%; 
        border-radius: 6px; 
        font-weight: 600; 
        background-color: #21262d; 
        color: white; 
        border: 1px solid #30363d;
    }
    .stButton>button:hover {
        border-color: #8b949e;
        color: #58a6ff;
    }
    .loop-row {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
    }
    h1, h2, h3, p { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

st.title("🎛️ Infinite Loop Finder")
st.caption("Detects segments strictly usable for seamless infinite looping (A -> B -> A).")

if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'selected_loop' not in st.session_state: st.session_state.selected_loop = None
if 'page' not in st.session_state: st.session_state.page = 0

def plot_waveform_with_loop(y, sr, loop=None):
    """绘制带高亮的波形图"""
    fig, ax = plt.subplots(figsize=(12, 2))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    
    # 绘制波形
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#7ee787', alpha=0.6) # Audjust Green
    
    if loop:
        # 绘制 Loop 区域 (从 start 到 end)
        rect = patches.Rectangle((loop['start'], -1), loop['duration'], 2, 
                                 facecolor='#1f6feb', alpha=0.5, edgecolor=None) # Audjust Blue
        ax.add_patch(rect)
        
        # 标记点
        ax.axvline(x=loop['start'], color='white', linestyle='--', linewidth=1, alpha=0.8)
        ax.axvline(x=loop['end'], color='white', linestyle='--', linewidth=1, alpha=0.8)
        
        # 文字
        ax.text(loop['start'], 0.8, "Start", color='white', fontsize=8)
        ax.text(loop['end'], 0.8, "End (Loop Point)", color='white', ha='right', fontsize=8)

    ax.set_yticks([])
    ax.set_xlim(0, librosa.get_duration(y=y, sr=sr))
    ax.tick_params(axis='x', colors='#8b949e')
    # remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tight_layout()
    return fig

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("MP3 / WAV", type=["mp3", "wav"])
    
    if uploaded_file:
        if st.button("Analyze Loops"):
            with st.spinner("Scanning for beat-sync loops..."):
                suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(uploaded_file.read())
                    tpath = tfile.name
                
                remixer = AudioRemixer(tpath)
                remixer.analyze()
                st.session_state.remixer = remixer
                st.session_state.selected_loop = None
                st.session_state.page = 0
                os.remove(tpath)
                
                count = len(remixer.loops)
                if count > 0:
                    st.success(f"Unlock {count} loops!")
                else:
                    st.error("No loops found. Try a more repetitive song.")

# --- Main Area ---
if st.session_state.remixer:
    remixer = st.session_state.remixer
    loops = remixer.loops
    
    # 1. Top Waveform
    st.subheader("Waveform Visualization")
    fig = plot_waveform_with_loop(remixer.y, remixer.sr, st.session_state.selected_loop)
    st.pyplot(fig)
    
    # 2. Filter / Stats
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**Found {len(loops)} Segments usable for infinite looping**")
    
    st.divider()
    
    # 3. Loop List (Audjust Style)
    # 分页显示，防止卡顿
    page_size = 10
    
    start_idx = st.session_state.page * page_size
    end_idx = start_idx + page_size
    current_loops = loops[start_idx:end_idx]
    
    # Grid Header
    h1, h2, h3, h4 = st.columns([1, 4, 2, 2])
    h1.markdown("#")
    h2.markdown("Loop Segment")
    h3.markdown("Duration")
    h4.markdown("Action")
    
    for i, loop in enumerate(current_loops):
        idx = start_idx + i
        
        # Container styling simulation
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 4, 2, 2])
            
            # Index
            col1.write(f"{idx+1}")
            
            # Mini-Visual (Text representation of bar)
            score_bar = "█" * int(loop['score'] * 10)
            col2.caption(f"Confidence: {score_bar} ({loop['score']:.2f})")
            col2.text(f"Region: {loop['start']:.1f}s — {loop['end']:.1f}s")
            
            # Duration
            col3.markdown(f"**{loop['duration']:.2f}s**")
            col3.caption(f"~{int(loop['beats_len']/4)} Bars")
            
            # Actions
            if col4.button("▶ Play Loop", key=f"play_{idx}"):
                st.session_state.selected_loop = loop
                # 生成拼接音频 A->B->A->B
                looped_audio = remixer.render_loop_preview(loop, repetitions=4)
                
                # Encode to play
                buf = io.BytesIO()
                sf.write(buf, looped_audio, remixer.sr, format='WAV')
                st.audio(buf.getvalue(), format='audio/wav', start_time=0)
                
            st.markdown("---")
            
    # Pagination
    p1, p2, p3 = st.columns([1, 1, 1])
    if p1.button("Previous") and st.session_state.page > 0:
        st.session_state.page -= 1
        st.rerun() # [Fixed] 使用 st.rerun() 替代 experimental_rerun
    
    if p3.button("Next") and end_idx < len(loops):
        st.session_state.page += 1
        st.rerun() # [Fixed] 使用 st.rerun() 替代 experimental_rerun

elif not uploaded_file:
    st.info("Please upload an audio file to start scanning for loops.")