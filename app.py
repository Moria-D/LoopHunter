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

st.title("🎛️ LoopHunter (Auto-Fit Mode)")
st.caption("Intelligently shorten or extend music while preserving Intro & Outro.")

# CSS
st.markdown("""
<style>
    .stButton>button { width: 100%; font-weight: bold; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'timeline' not in st.session_state: st.session_state.timeline = None

def plot_connection_map(y, sr, timeline, total_remix_dur):
    source_dur = librosa.get_duration(y=y, sr=sr)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.4})
    fig.patch.set_facecolor('#0E1117')
    
    # Source
    ax1.set_facecolor('#1E1E1E')
    ax1.set_title("SOURCE: Original Waveform", color='white', loc='left')
    ax1.set_xlim(0, source_dur)
    ax1.set_yticks([])
    ax1.tick_params(axis='x', colors='white')
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='#444', alpha=0.5)
    
    # Remix
    ax2.set_facecolor('#1E1E1E')
    ax2.set_title("REMIX: Generated Timeline", color='white', loc='left')
    ax2.set_xlim(0, total_remix_dur)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.tick_params(axis='x', colors='white')
    
    colors = {'Intro': '#00bcd4', 'Head': '#00bcd4', 'Body': '#4caf50', 'Outro': '#9c27b0', 'Tail': '#9c27b0'}
    
    for i, seg in enumerate(timeline):
        src_s = seg['source_start']
        src_e = seg['source_end']
        remix_s = seg['remix_start']
        remix_e = remix_s + seg['duration']
        
        l_type = seg['type']
        color = colors.get(l_type, '#999')
        
        # Source Block
        rect_src = patches.Rectangle((src_s, -1), src_e-src_s, 2, facecolor=color, alpha=0.4)
        ax1.add_patch(rect_src)
        
        # Remix Block
        rect_remix = patches.Rectangle((remix_s, 0.2), remix_e-remix_s, 0.6, facecolor=color, edgecolor='white', linewidth=0.5)
        ax2.add_patch(rect_remix)
        
        # Time Labels
        if i == 0 or timeline[i-1]['type'] != seg['type']:
             ax2.text(remix_s, 0.9, f"{remix_s:.1f}s", color='white', fontsize=8, rotation=45)
             
        # Center Label
        ax2.text((remix_s+remix_e)/2, 0.5, l_type, color='white', ha='center', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    return fig

# Sidebar
with st.sidebar:
    st.header("1. Upload")
    uploaded_file = st.file_uploader("Audio", type=["mp3", "wav"])
    
    if uploaded_file:
        if st.button("Analyze"):
            with st.spinner("Analyzing structure..."):
                suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(uploaded_file.read())
                    tpath = tfile.name
                
                remixer = AudioRemixer(tpath)
                remixer.analyze()
                st.session_state.remixer = remixer
                st.session_state.timeline = None
                os.remove(tpath)
                st.success("Analysis Done.")

    st.divider()
    
    if st.session_state.remixer:
        orig_dur = st.session_state.remixer.duration
        target_dur = st.slider("Target Duration (s)", 15, int(orig_dur*2), int(orig_dur), step=5)
        
        if st.button("Generate Remix"):
            tl = st.session_state.remixer.generate_path(target_dur)
            st.session_state.timeline = tl

# Main
if st.session_state.remixer and st.session_state.timeline:
    tl = st.session_state.timeline
    remixer = st.session_state.remixer
    final_dur = sum(s['duration'] for s in tl)
    
    st.subheader("🔗 Remix Map")
    st.info(f"Original: {remixer.duration:.1f}s | Target: {target_dur}s | Actual: {final_dur:.1f}s")
    
    fig = plot_connection_map(remixer.y, remixer.sr, tl, final_dur)
    st.pyplot(fig)
    
    st.subheader("🎧 Final Audio")
    if st.button("▶️ Render"):
        with st.spinner("Rendering..."):
            audio = remixer.render(tl)
            if np.max(np.abs(audio)) > 0: audio = audio / np.max(np.abs(audio)) * 0.95
            
            buf = io.BytesIO()
            sf.write(buf, audio, remixer.sr, format='WAV')
            st.audio(buf.getvalue(), format='audio/wav')
            st.download_button("Download WAV", buf, "remix.wav")

elif not uploaded_file:
    st.info("Please upload a file.")