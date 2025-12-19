import streamlit as st
import numpy as np
import librosa, librosa.display, soundfile as sf
import io, os, tempfile, json
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from division import AudioRemixer
from drum_processor import DrumLoopExtractor

# --- 原有绘图与分析辅助函数 (完全保留) ---
def analyze_rhythm_structure(events, duration):
    stats = {}
    for instr, ev_list in events.items():
        if not ev_list:
            stats[instr] = {"count": 0, "density_ppm": 0, "active_ratio_pct": 0}
            continue
        total_act = sum(e['duration'] for e in ev_list)
        stats[instr] = {"count": len(ev_list), "density_ppm": round(len(ev_list)/duration*60, 1), 
                        "active_ratio_pct": round(total_act/duration*100, 1)}
    return stats

def plot_mini_waveform_with_highlight(y, sr, loop_start, loop_end):
    fig, ax = plt.subplots(figsize=(10, 1.2))
    fig.patch.set_facecolor('#161b22'); ax.set_facecolor('#161b22')
    step = 100; y_subs = y[::step]; sr_subs = sr / step
    librosa.display.waveshow(y_subs, sr=sr_subs, ax=ax, color='#444', alpha=0.4)
    if loop_end > loop_start:
        s_idx, e_idx = int(loop_start * sr / step), int(loop_end * sr / step)
        ax.plot(np.arange(len(y_subs))[s_idx:e_idx]/sr_subs, y_subs[s_idx:e_idx], color='#3b82f6', linewidth=0.8)
        ax.add_patch(patches.Rectangle((loop_start, -1), loop_end - loop_start, 2, facecolor='#1f6feb', alpha=0.15))
    ax.set_yticks([]); ax.set_xticks([]); ax.set_xlim(0, len(y_subs)/sr_subs)
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout(pad=0); return fig

def plot_single_stem_waveform(y, sr, color):
    step = max(1, len(y) // 3000); y_d = y[::step]; x_d = np.arange(0, len(y), step) / sr
    fig = go.Figure(data=go.Scatter(x=x_d, y=y_d, mode='lines', line=dict(color=color, width=1), fill='tozeroy'))
    fig.update_layout(template="plotly_dark", height=120, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, 
                      xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_remix_waveform(remix_y, sr, timeline, total_remix_dur):
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
    ax.set_xlim(0, total_remix_dur); ax.tick_params(axis='x', colors='#8b949e'); ax.set_yticks([])
    step = 100
    for seg in timeline:
        s_time, dur = seg['remix_start'], seg['duration']
        color = '#1f6feb' if seg['type'] == 'Loop Extension' else ('#d2a8ff' if seg.get('is_jump') else '#238636')
        s_samp, e_samp = int(s_time * sr), int((s_time + dur) * sr)
        if e_samp > len(remix_y): e_samp = len(remix_y)
        ax.plot(np.linspace(s_time, s_time+dur, (e_samp-s_samp)//step), remix_y[s_samp:e_samp:step], color=color, linewidth=0.8)
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout(); return fig

# --- App Setup (整合功能) ---
st.set_page_config(layout="wide", page_title="LoopHunter - Remix Studio")
st.title("🎛️ Audio Loop & Remix Studio")

if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'stem_audio' not in st.session_state: st.session_state.stem_audio = None
if 'drum_loops' not in st.session_state: st.session_state.drum_loops = None

with st.sidebar:
    st.header("1. Upload & Analyze")
    up = st.file_uploader("Music File", type=["mp3", "wav"])
    if up and st.button("🚀 Full Analysis", type="primary"):
        with st.spinner("Analyzing rhythm & stems..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as t:
                t.write(up.read()); tp = t.name
            rx = AudioRemixer(tp); rx.analyze(); st.session_state.remixer = rx
            ev, sa = rx.analyze_stems(); st.session_state.stem_audio = sa
            os.remove(tp); st.success("Analysis Complete!")

    if st.session_state.remixer:
        st.divider(); st.header("2. Remix Logic")
        target = st.slider("Target Duration (s)", 10, int(st.session_state.remixer.duration*3), int(st.session_state.remixer.duration))
        if st.button("✨ Generate Remix", type="primary"):
            tl, ad = st.session_state.remixer.plan_multi_loop_remix(target)
            st.session_state.timeline, st.session_state.final_dur = tl, ad
            audio = st.session_state.remixer.render_remix(tl)
            st.session_state.final_audio = audio / np.max(np.abs(audio)) * 0.95 if np.any(audio) else audio

# --- Main Dashboard (整合所有原有功能) ---
if st.session_state.remixer:
    remixer = st.session_state.remixer
    
    # 2. 全局 Loop 扫描 (完全保留)
    if remixer.loops:
        st.subheader("Detected Global Loops")
        for i, loop in enumerate(remixer.loops[:3]):
            st.pyplot(plot_mini_waveform_with_highlight(remixer.y, remixer.sr, loop['start'], loop['end']))

    # 3. 分轨试听与结构分析 (完全保留并优化)
    if st.session_state.stem_audio:
        st.divider(); st.subheader("Instrument Rhythm Breakdown & Stem Players")
        if st.button("🥁 Deep Drum Structure Analysis", type="primary"):
            extractor = DrumLoopExtractor(sr=remixer.sr)
            st.session_state.drum_loops = extractor.process(st.session_state.stem_audio['drums'], remixer.beat_times)

        detailed_rows = []
        for instr, sa in st.session_state.stem_audio.items():
            with st.container():
                st.write(f"**{instr.upper()}**")
                st.plotly_chart(plot_single_stem_waveform(sa, remixer.sr, "#3b82f6"), use_container_width=True)
                st.audio(io.BytesIO(sf.write(io.BytesIO(), sa/np.max(np.abs(sa)), remixer.sr, format='WAV')))
                
                # 修复 KeyError: 'loop' 并显示多重循环表格
                if st.session_state.drum_loops and instr in st.session_state.drum_loops:
                    markers = st.session_state.drum_loops[instr]['markers']
                    if markers:
                        with st.expander(f"♾️ Structure Table & Loop Preview ({instr.capitalize()})", expanded=True):
                            st.table(pd.DataFrame(markers))
                            for m in markers: detailed_rows.append({**m, "Track": instr})
                            # 预览首个 Loop 段落
                            loop_seg = next((m for m in markers if "loop" in m["Type"]), None)
                            if loop_seg:
                                st.pyplot(plot_mini_waveform_with_highlight(sa, remixer.sr, loop_seg['Start Time (s)'], loop_seg['End Time (s)']))

        # CSV 导出 (第一列为时间)
        if detailed_rows:
            df_loops = pd.DataFrame(detailed_rows)[["Start Time (s)", "End Time (s)", "Duration (s)", "Track", "Type"]]
            st.download_button("Download track_structure.csv", df_loops.to_csv(index=False).encode('utf-8'), "track_structure.csv", "text/csv", use_container_width=True)

    # 4. Remix 混剪结果 (完全保留)
    if 'final_audio' in st.session_state:
        st.divider(); st.subheader("Generated Remix Result")
        st.pyplot(plot_remix_waveform(st.session_state.final_audio, remixer.sr, st.session_state.timeline, st.session_state.final_dur))
        st.audio(io.BytesIO(sf.write(io.BytesIO(), st.session_state.final_audio, remixer.sr, format='WAV')))