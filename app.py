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

st.set_page_config(layout="wide", page_title="LoopHunter - Smart Cut")

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
    h1, h2, h3, p { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

st.title("🎛️ Audio Smart Remixer")
st.caption("Auto-Extension & Smart Shortening.")

if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'timeline' not in st.session_state: st.session_state.timeline = None
if 'final_audio' not in st.session_state: st.session_state.final_audio = None
if 'final_dur' not in st.session_state: st.session_state.final_dur = 0.0

def plot_structure(y, sr, timeline, total_remix_dur):
    source_dur = librosa.get_duration(y=y, sr=sr)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.6})
    fig.patch.set_facecolor('#0d1117')
    
    # 1. Source
    ax1.set_facecolor('#161b22')
    ax1.set_title("Source Audio", color='#8b949e', loc='left', fontsize=10)
    ax1.set_xlim(0, source_dur)
    ax1.set_yticks([])
    ax1.tick_params(axis='x', colors='#8b949e')
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#3fb950', alpha=0.5)
    
    # 2. Remix
    ax2.set_facecolor('#161b22')
    ax2.set_title("Remix Structure", color='#8b949e', loc='left', fontsize=10)
    ax2.set_xlim(0, total_remix_dur)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.tick_params(axis='x', colors='#8b949e')
    
    for i, seg in enumerate(timeline):
        start = seg['remix_start']
        dur = seg['duration']
        label = seg['type']
        
        c = '#238636' # Linear
        if label == 'Loop Extension': c = '#1f6feb' # Loop
        if seg.get('is_jump'): c = '#d2a8ff' # Tail/Skip
        
        # Source highlight
        rect_src = patches.Rectangle((seg['source_start'], -1), dur, 2, facecolor=c, alpha=0.4)
        ax1.add_patch(rect_src)
        
        # Remix block
        rect = patches.Rectangle((start, 0.2), dur, 0.6, facecolor=c, edgecolor='white', linewidth=0.5)
        ax2.add_patch(rect)
        
        # Text
        ax2.text(start, 0.9, f"{start:.1f}s", color='white', rotation=0, ha='left', va='bottom', fontsize=8)
        if dur > 5.0:
            ax2.text(start + dur/2, 0.5, label, color='white', ha='center', va='center', fontsize=9, fontweight='bold')
            
        # Jump Marker
        if seg.get('xfade', 0) > 0:
             ax2.scatter([start], [0.8], color='white', s=20, zorder=10)

    plt.tight_layout()
    return fig

# --- Sidebar ---
with st.sidebar:
    st.header("1. Upload")
    uploaded_file = st.file_uploader("Audio", type=["mp3", "wav"])
    
    if uploaded_file:
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(uploaded_file.read())
                    tpath = tfile.name
                
                remixer = AudioRemixer(tpath)
                remixer.analyze()
                st.session_state.remixer = remixer
                st.session_state.timeline = None
                st.session_state.final_audio = None
                os.remove(tpath)
                st.success("Ready.")

    st.divider()
    
    if st.session_state.remixer:
        duration = st.session_state.remixer.duration
        # 允许缩短到 10s，最大 3倍
        target_dur = st.slider("Target Duration (s)", 
                               min_value=10, 
                               max_value=int(duration*3), 
                               value=int(duration), 
                               step=1) # 精度提高到1秒
        
        if st.button("Generate Remix", type="primary"):
            tl, actual_dur = st.session_state.remixer.plan_multi_loop_remix(target_dur)
            st.session_state.timeline = tl
            
            with st.spinner("Rendering..."):
                audio = st.session_state.remixer.render_remix(tl)
                if len(audio) > 0:
                    mx = np.max(np.abs(audio))
                    if mx > 0: audio = audio / mx * 0.95
                
                st.session_state.final_audio = audio
                st.session_state.final_dur = actual_dur

# --- Main ---
if st.session_state.remixer and st.session_state.timeline:
    tl = st.session_state.timeline
    
    st.subheader("Structure")
    fig = plot_structure(
        st.session_state.remixer.y, 
        st.session_state.remixer.sr, 
        tl, 
        st.session_state.final_dur
    )
    st.pyplot(fig)
    
    st.divider()
    
    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.info(f"Target: {target_dur}s")
    c2.success(f"Actual: {st.session_state.final_dur:.1f}s")
    
    if st.session_state.final_audio is not None:
        buf = io.BytesIO()
        sf.write(buf, st.session_state.final_audio, st.session_state.remixer.sr, format='WAV')
        st.audio(buf.getvalue(), format='audio/wav')
        st.download_button("Download Remix", buf, "remix.wav")

elif not uploaded_file:
    st.info("👋 Upload audio to start.")