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

st.set_page_config(layout="wide", page_title="LoopHunter - Final UI")

# --- CSS 注入：核心布局调整 ---
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    
    /* 1. 精确调整上传组件布局 */
    /* 目标：Label (Top) -> File List (Middle) -> Dropzone (Bottom) */
    
    /* 找到包含 dropzone 和 file list 的内部容器 */
    [data-testid="stFileUploader"] > div > div {
        display: flex;
        flex-direction: column-reverse;
    }
    
    /* 调整 Dropzone (Browse Button) 的样式，使其在底部时看起来协调 */
    [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
        margin-top: 10px;
    }
    
    /* 2. 优化按钮样式 */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3rem; /* 增加按钮高度 */
        margin-top: 5px;
    }
    
    /* 播放器容器样式 */
    .audio-player-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* 数字输入框居中 */
    div[data-testid="stNumberInput"] input {
        text-align: center;
    }
    
    h1, h2, h3, p, label { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

st.title("🎛️ Audio Loop & Remix Studio")

# State Management
if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'timeline' not in st.session_state: st.session_state.timeline = None
if 'final_audio' not in st.session_state: st.session_state.final_audio = None
if 'final_dur' not in st.session_state: st.session_state.final_dur = 0.0
if 'loop_page' not in st.session_state: st.session_state.loop_page = 0

# ----------------- PLOT FUNCTIONS -----------------

def plot_mini_waveform_with_highlight(y, sr, loop_start, loop_end):
    """绘制长条形波形图"""
    fig, ax = plt.subplots(figsize=(10, 1.2)) 
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    
    step = 100
    y_subs = y[::step]
    sr_subs = sr / step
    
    librosa.display.waveshow(y_subs, sr=sr_subs, ax=ax, color='#444', alpha=0.4)
    
    # Highlight
    s_idx = int(loop_start * sr / step)
    e_idx = int(loop_end * sr / step)
    
    if e_idx > s_idx:
        times = np.arange(len(y_subs)) / sr_subs
        loop_times = times[s_idx:e_idx]
        loop_chunk = y_subs[s_idx:e_idx]
        ax.plot(loop_times, loop_chunk, color='#3b82f6', linewidth=0.8, alpha=0.9) 
        
        rect = patches.Rectangle((loop_start, -1), loop_end - loop_start, 2, 
                                 facecolor='#1f6feb', alpha=0.15, edgecolor=None)
        ax.add_patch(rect)
    
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, len(y_subs)/sr_subs)
    for spine in ax.spines.values(): spine.set_visible(False)
        
    plt.tight_layout(pad=0)
    return fig

def plot_remix_structure(timeline, total_remix_dur):
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    ax.set_xlim(0, total_remix_dur)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", color='#8b949e')
    ax.tick_params(axis='x', colors='#8b949e')
    
    for i, seg in enumerate(timeline):
        start = seg['remix_start']
        dur = seg['duration']
        label = seg['type']
        
        c = '#238636' # Linear
        if label == 'Loop Extension': c = '#1f6feb' # Loop
        if seg.get('is_jump'): c = '#d2a8ff' # Tail/Skip
        
        rect = patches.Rectangle((start, 0.2), dur, 0.6, facecolor=c, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)
        
        # Timestamp
        ax2_text = f"{start:.1f}s"
        ax.text(start, 0.9, ax2_text, color='white', rotation=45, ha='left', va='bottom', fontsize=8)

        if dur > 2.0:
            lbl = label if "Loop" not in label else "Loop"
            ax.text(start + dur/2, 0.5, lbl, color='white', ha='center', va='center', fontsize=9, fontweight='bold')
            
        if seg.get('xfade', 0) > 0:
             ax.scatter([start], [0.8], color='white', s=20, zorder=10)

    plt.tight_layout()
    return fig

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("1. Upload")
    
    # 上传组件
    uploaded_file = st.file_uploader("Audio", type=["mp3", "wav"])
    
    # 按钮逻辑优化
    if uploaded_file:
        st.write("") # Spacer
        
        # Analyze 按钮
        if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("Scanning structure & loops..."):
                suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(uploaded_file.read())
                    tpath = tfile.name
                
                remixer = AudioRemixer(tpath)
                remixer.analyze()
                st.session_state.remixer = remixer
                st.session_state.timeline = None
                st.session_state.final_audio = None
                st.session_state.loop_page = 0 
                os.remove(tpath)
                
                count = len(remixer.loops)
                if count > 0:
                    st.success(f"Found {count} loops!")
                else:
                    st.warning("No loops found.")

    st.divider()
    
    if st.session_state.remixer:
        st.header("3. Remix Settings")
        duration = st.session_state.remixer.duration
        target_dur = st.slider("Target Duration (s)", 
                               min_value=10, 
                               max_value=int(duration*3), 
                               value=int(duration), 
                               step=1)
        
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

# ----------------- MAIN AREA -----------------
if st.session_state.remixer:
    remixer = st.session_state.remixer
    
    # --- 2. Detected Loops Area ---
    if remixer.loops:
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
                    st.write("") 
                    st.write("") 
                    if st.button(f"▶", key=f"play_{i}"):
                        loop_audio = remixer.generate_loop_preview(loop, repetitions=4)
                        buf = io.BytesIO()
                        sf.write(buf, loop_audio, remixer.sr, format='WAV')
                        st.session_state[f'audio_{i}'] = buf.getvalue()
                
                with c2:
                    fig = plot_mini_waveform_with_highlight(remixer.y, remixer.sr, loop['start'], loop['end'])
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
                if 0 <= new_page < total_pages:
                    st.session_state.loop_page = new_page

            st.number_input(
                "Jump to Page", 
                min_value=1, 
                max_value=total_pages, 
                value=current_page + 1,
                step=1,
                key="page_input",
                on_change=update_page_number,
                label_visibility="collapsed"
            )
            st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.8em;'>of {total_pages} pages</div>", unsafe_allow_html=True)
            
        with col_next:
            if st.button("Next ▶", disabled=(end_idx == total_loops), use_container_width=True):
                st.session_state.loop_page += 1
                st.rerun()

    else:
        st.info("No loops detected.")

    # --- 4. Remix Result Area ---
    if st.session_state.timeline:
        st.divider()
        st.subheader("4. Remix Result")
        
        fig_struct = plot_remix_structure(st.session_state.timeline, st.session_state.final_dur)
        st.pyplot(fig_struct)
        
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