import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import io
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import our backend modules
from division import AudioRemixer
from drum_processor import DrumLoopExtractor

st.set_page_config(layout="wide", page_title="LoopHunter - Drum Kit Edition")

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
    h1, h2, h3, p, label { color: #c9d1d9; }
    .stDownloadButton button { height: 3rem; }
</style>
""", unsafe_allow_html=True)

st.title("🥁 LoopHunter - Drum Loop Extractor")
st.caption("Lightweight tool focused on separating and looping Kick, Snare, and Cymbals.")

if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'drum_loops' not in st.session_state: st.session_state.drum_loops = None
if 'full_drums_audio' not in st.session_state: st.session_state.full_drums_audio = None

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

# --- Sidebar Upload ---
with st.sidebar:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Music File", type=["mp3", "wav"])
    
    if uploaded_file:
        if st.button("🚀 Analyze Drums", type="primary", use_container_width=True):
            with st.spinner("Analyzing rhythm structure..."):
                suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(uploaded_file.read())
                    tpath = tfile.name
                
                # 1. Initialize Remixer (Standard Analysis)
                remixer = AudioRemixer(tpath)
                remixer.analyze() 
                st.session_state.remixer = remixer
                
                # 2. Get Stems (We only strictly need Drums, but analyze_stems does all)
                # We could optimize this in division.py to only return drums if requested, 
                # but currently we just run the standard pipeline.
                _, stems_audio = remixer.analyze_stems()
                
                if 'drums' in stems_audio:
                    st.session_state.full_drums_audio = stems_audio['drums']
                    
                    # 3. Run Drum Loop Extraction Immediately
                    with st.spinner("Decomposing drum kit and finding loops..."):
                        extractor = DrumLoopExtractor(sr=remixer.sr)
                        drum_loops = extractor.process(stems_audio['drums'], beat_times=remixer.beat_times)
                        st.session_state.drum_loops = drum_loops
                else:
                    st.error("Could not extract drums stem.")
                
                os.remove(tpath)
                st.success("Analysis Complete!")

# --- Main Content ---
if not uploaded_file:
    st.info("👋 Upload an audio file to start.")
    
elif st.session_state.drum_loops:
    remixer = st.session_state.remixer
    
    # 1. Full Drum Stem Preview
    st.subheader("1. Full Drum Stem")
    if st.session_state.full_drums_audio is not None:
        buf = io.BytesIO()
        sf.write(buf, st.session_state.full_drums_audio, remixer.sr, format='WAV')
        st.audio(buf.getvalue(), format='audio/wav')
    
    st.divider()
    
    # 2. Components Loop Extraction
    st.subheader("2. Component Loops (Kick / Snare / Cymbals)")
    
    for name, data in st.session_state.drum_loops.items():
        with st.container():
            st.markdown(f"### {name.capitalize()}")
            col_a, col_b = st.columns([1, 3])
            
            # Left: Full Component Audio
            audio_comp = data['audio']
            if len(audio_comp) > 0:
                mx = np.max(np.abs(audio_comp))
                if mx > 0: audio_comp_norm = audio_comp / mx * 0.95
                else: audio_comp_norm = audio_comp
                
                buf = io.BytesIO()
                sf.write(buf, audio_comp_norm, remixer.sr, format='WAV')
                col_a.write("**Full Track**")
                col_a.audio(buf.getvalue(), format='audio/wav')
            
            # Right: Loop Info & Previews
            loop = data['loop']
            if loop:
                col_b.success(f"Loop Detected: {loop['duration']:.2f}s (Conf: {loop['confidence']:.2f})")
                
                # Extract Loop Audio
                s_samp = int(loop['start'] * remixer.sr)
                e_samp = int(loop['end'] * remixer.sr)
                loop_chunk = data['audio'][s_samp:e_samp]
                
                if len(loop_chunk) > 0:
                    mx = np.max(np.abs(loop_chunk))
                    if mx > 0: loop_chunk = loop_chunk / mx * 0.95
                    
                    # 4x Preview
                    preview_4x = np.tile(loop_chunk, 4)
                    buf_loop_4x = io.BytesIO()
                    sf.write(buf_loop_4x, preview_4x, remixer.sr, format='WAV')
                    
                    # 1x Preview
                    buf_loop_1x = io.BytesIO()
                    sf.write(buf_loop_1x, loop_chunk, remixer.sr, format='WAV')
                    
                    c1, c2 = col_b.columns(2)
                    c1.write("**Loop (1x)**")
                    c1.audio(buf_loop_1x.getvalue(), format='audio/wav')
                    
                    c2.write("**Loop (4x)**")
                    c2.audio(buf_loop_4x.getvalue(), format='audio/wav')
                
                # Waveform
                fig_loop = plot_mini_waveform_with_highlight(data['audio'], remixer.sr, loop['start'], loop['end'])
                col_b.pyplot(fig_loop)
            else:
                col_b.warning("No consistent loop found.")
                
            st.markdown("<hr style='margin: 10px 0; opacity: 0.1;'>", unsafe_allow_html=True)

else:
    if uploaded_file and not st.session_state.drum_loops:
        st.info("Click '🚀 Analyze Drums' in the sidebar to begin.")

