import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import io
import os
import tempfile
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from division import AudioRemixer
from drum_processor import DrumLoopExtractor

# --- 1. Analysis Helper Functions ---
def analyze_rhythm_structure(events, duration):
    """生成统计洞察数据"""
    stats = {}
    for instr, ev_list in events.items():
        if not ev_list:
            stats[instr] = {"count": 0, "density_ppm": 0, "avg_dur_ms": 0, "active_ratio_pct": 0, "avg_gap_ms": 0}
            continue
        count = len(ev_list)
        total_active_time = sum(e['duration'] for e in ev_list)
        avg_dur = total_active_time / count if count > 0 else 0
        density = count / duration * 60  
        active_ratio = total_active_time / duration * 100
        intervals = []
        for i in range(len(ev_list) - 1):
            gap = ev_list[i+1]['start'] - (ev_list[i]['start'] + ev_list[i]['duration'])
            if gap > 0: intervals.append(gap)
        avg_interval = np.mean(intervals) if intervals else 0
        stats[instr] = {
            "count": count,
            "density_ppm": round(density, 1),
            "avg_dur_ms": round(avg_dur * 1000, 1),
            "active_ratio_pct": round(active_ratio, 1),
            "avg_gap_ms": round(avg_interval * 1000, 1)
        }
    return stats

def generate_text_report(stats):
    """生成自然语言摘要总结报告"""
    report = []
    if not stats: return "No analysis data available."
    densities = {k: v['density_ppm'] for k, v in stats.items()}
    if densities:
        dominant = max(densities, key=densities.get)
        report.append(f"**Dominant Element:** The **{dominant.capitalize()}** leads the texture ({densities[dominant]} events/min).")
    sorted_instrs = sorted(stats.items(), key=lambda x: x[1]['active_ratio_pct'], reverse=True)
    for instr, data in sorted_instrs:
        desc = f"{instr.capitalize()}: {data['active_ratio_pct']}% active, {data['avg_dur_ms']}ms avg duration."
        report.append(f"- {desc}")
    return "\n\n".join(report)

# --- 2. Plotting Helpers ---
def plot_mini_waveform_with_highlight(y, sr, loop_start, loop_end):
    """带高亮区域的小型波形预览"""
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
        ax.plot(times[s_idx:e_idx], y_subs[s_idx:e_idx], color='#3b82f6', linewidth=0.8, alpha=0.9) 
        ax.add_patch(patches.Rectangle((loop_start, -1), loop_end - loop_start, 2, facecolor='#1f6feb', alpha=0.15))
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlim(0, len(y_subs)/sr_subs)
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout(pad=0)
    return fig

def plot_single_stem_waveform(y, sr, color):
    """绘制分轨交互式 Plotly 波形图"""
    step = max(1, len(y) // 3000)
    y_down = y[::step]
    x_down = np.arange(0, len(y), step) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_down, y=y_down, mode='lines', line=dict(color=color, width=1), fill='tozeroy'))
    fig.update_layout(template="plotly_dark", height=120, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      xaxis=dict(visible=False, fixedrange=True), yaxis=dict(visible=False, fixedrange=True),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_remix_waveform(remix_y, sr, timeline, total_remix_dur):
    """绘制 Remix 渲染结果波形图"""
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
    ax.set_xlim(0, total_remix_dur); ax.set_xlabel("Time (s)", color='#8b949e')
    ax.tick_params(axis='x', colors='#8b949e'); ax.set_yticks([])
    step = 100
    for seg in timeline:
        s_time, dur = seg['remix_start'], seg['duration']
        color = '#1f6feb' if seg['type'] == 'Loop Extension' else ('#d2a8ff' if seg.get('is_jump') else '#238636')
        s_samp, e_samp = int(s_time * sr), int((s_time + dur) * sr)
        if e_samp > len(remix_y): e_samp = len(remix_y)
        if s_samp >= e_samp: continue
        seg_y = remix_y[s_samp:e_samp:step]
        ax.plot(np.linspace(s_time, s_time + dur, len(seg_y)), seg_y, color=color, linewidth=0.8, alpha=0.9)
        ax.axvline(x=s_time, color='white', linestyle=':', linewidth=0.8, alpha=0.6)
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    return fig

# --- 3. App Setup ---
st.set_page_config(layout="wide", page_title="LoopHunter - Audio Loop & Remix Studio")

st.title("🎛️ Audio Loop & Remix Studio")
st.caption("Advanced tool for scanning loops, analyzing instrument rhythm, and generating custom-length remixes.")

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stButton>button { width: 100%; border-radius: 6px; font-weight: 600; }
    .audio-player-box { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 10px; }
    h1, h2, h3, p, label { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'stem_events' not in st.session_state: st.session_state.stem_events = None
if 'stem_audio' not in st.session_state: st.session_state.stem_audio = None
if 'timeline' not in st.session_state: st.session_state.timeline = None
if 'final_audio' not in st.session_state: st.session_state.final_audio = None
if 'drum_loops' not in st.session_state: st.session_state.drum_loops = None

# --- 4. Sidebar Controller ---
with st.sidebar:
    st.header("1. Upload Audio")
    uploaded_file = st.file_uploader("Choose music file", type=["mp3", "wav"])
    if uploaded_file and st.button("🚀 Run Full Analysis", type="primary"):
        with st.spinner("Analyzing rhythm & stems..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
                tfile.write(uploaded_file.read())
                tpath = tfile.name
            remixer = AudioRemixer(tpath); remixer.analyze()
            st.session_state.remixer = remixer
            events, stems_audio = remixer.analyze_stems()
            st.session_state.stem_events, st.session_state.stem_audio = events, stems_audio
            st.session_state.timeline = None; st.session_state.final_audio = None; st.session_state.drum_loops = None
            os.remove(tpath); st.success("Analysis Complete!")

    if st.session_state.remixer:
        st.divider(); st.header("2. Remix Logic")
        dur = st.session_state.remixer.duration
        target_dur = st.slider("Target Duration (s)", 10, int(dur*3), int(dur))
        if st.button("✨ Generate Remix", type="primary"):
            tl, actual_dur = st.session_state.remixer.plan_multi_loop_remix(target_dur)
            st.session_state.timeline, st.session_state.final_dur = tl, actual_dur
            with st.spinner("Rendering..."):
                audio = st.session_state.remixer.render_remix(tl)
                if len(audio) > 0: audio = (audio / np.max(np.abs(audio)) * 0.95) if np.max(np.abs(audio)) > 0 else audio
                st.session_state.final_audio = audio

# --- 5. Main Dashboard ---
if st.session_state.remixer:
    remixer = st.session_state.remixer
    
    # Section 2: Global Loops
    if remixer.loops:
        st.subheader("2. Detected Global Loops")
        for i, loop in enumerate(remixer.loops[:5]):
            with st.container():
                c1, c2, c3 = st.columns([1, 6, 2])
                with c1: 
                    if st.button(f"▶", key=f"lp_{i}"):
                        st.session_state[f'aud_{i}'] = remixer.generate_loop_preview(loop, 4)
                with c2:
                    st.pyplot(plot_mini_waveform_with_highlight(remixer.y, remixer.sr, loop['start'], loop['end']))
                    if f'aud_{i}' in st.session_state:
                        buf = io.BytesIO(); sf.write(buf, st.session_state[f'aud_{i}'], remixer.sr, format='WAV')
                        st.audio(buf.getvalue(), format='audio/wav')
                with c3: st.markdown(f"**{loop['duration']:.1f}s**"); st.caption(f"Score: {loop['score']:.2f}")

    # Section 3: Rhythm Breakdown & Stem Players (融合 Drum Decomposition)
    if st.session_state.stem_events:
        st.divider(); st.subheader("3. Instrument Rhythm Breakdown & Stem Players")
        
        # 鼓组分解触发按钮放在此处
        if st.session_state.stem_audio and 'drums' in st.session_state.stem_audio:
            if st.button("🥁 Analyze Drum Components (Kick/Snare/Cymbals)", type="primary"):
                with st.spinner("Decomposing Drum Kit..."):
                    extractor = DrumLoopExtractor(sr=remixer.sr)
                    st.session_state.drum_loops = extractor.process(st.session_state.stem_audio['drums'], beat_times=remixer.beat_times)
        
        stats = analyze_rhythm_structure(st.session_state.stem_events, remixer.duration)
        colors_list = px.colors.qualitative.Plotly
        
        for i, (instr, data) in enumerate(stats.items()):
            with st.container():
                col_info, col_wave = st.columns([1, 5])
                with col_info:
                    st.markdown(f"**{instr.upper()}**")
                    st.caption(f"Density: {data['density_ppm']} ppm\nActive: {data['active_ratio_pct']}%")
                
                with col_wave:
                    if instr in st.session_state.stem_audio:
                        # 交互式 Plotly 波形图
                        st.plotly_chart(plot_single_stem_waveform(st.session_state.stem_audio[instr], remixer.sr, colors_list[i % len(colors_list)]), 
                                        use_container_width=True, config={'displayModeBar': False})
                        
                        # 分轨音频播放器 (保持长进度条)
                        stem_y = st.session_state.stem_audio[instr]
                        if np.any(stem_y):
                            mx = np.max(np.abs(stem_y))
                            norm_y = stem_y / mx * 0.95 if mx > 0 else stem_y
                            buf = io.BytesIO(); sf.write(buf, norm_y, remixer.sr, format='WAV')
                            st.caption(f"Full Track: {instr.capitalize()}")
                            st.audio(buf.getvalue())
                        
                        # --- 融合区域: 如果是鼓组子组件且已分析，则插入 Loop 预览 ---
                        if st.session_state.drum_loops and instr in st.session_state.drum_loops:
                            loop = st.session_state.drum_loops[instr]['loop']
                            if loop:
                                with st.expander(f"♾️ Precision Loop Preview ({instr.capitalize()})", expanded=True):
                                    cl1, cl2 = st.columns([2, 1])
                                    with cl1:
                                        st.success(f"Loop Detected: {loop['duration']:.2f}s")
                                        st.pyplot(plot_mini_waveform_with_highlight(st.session_state.drum_loops[instr]['audio'], remixer.sr, loop['start'], loop['end']))
                                    with cl2:
                                        s_samp, e_samp = int(loop['start']*remixer.sr), int(loop['end']*remixer.sr)
                                        loop_chunk = st.session_state.drum_loops[instr]['audio'][s_samp:e_samp]
                                        if len(loop_chunk) > 0:
                                            l_norm = loop_chunk / np.max(np.abs(loop_chunk)) * 0.95
                                            b1, b4 = io.BytesIO(), io.BytesIO()
                                            sf.write(b1, l_norm, remixer.sr, format='WAV')
                                            sf.write(b4, np.tile(l_norm, 4), remixer.sr, format='WAV')
                                            st.caption("1x Loop"); st.audio(b1.getvalue())
                                            st.caption("4x Loop Preview"); st.audio(b4.getvalue())
                        
                        # 5段随机事件试听功能
                        ev_list = st.session_state.stem_events.get(instr, [])
                        if ev_list:
                            with st.expander(f"🔍 Sample Note Audition ({instr.capitalize()})"):
                                num_events = len(ev_list)
                                start_idx = np.random.randint(0, max(1, num_events - 4))
                                for idx in range(start_idx, min(num_events, start_idx + 5)):
                                    ev = ev_list[idx]
                                    s_idx, e_idx = max(0, int((ev['start']-0.05)*remixer.sr)), min(len(stem_y), int((ev['start']+ev['duration']+0.05)*remixer.sr))
                                    clip = stem_y[s_idx:e_idx]
                                    if len(clip) > 0:
                                        buf_c = io.BytesIO(); sf.write(buf_c, clip/np.max(np.abs(clip))*0.9, remixer.sr, format='WAV')
                                        sc1, sc2 = st.columns([1, 4])
                                        sc1.caption(f"Event #{idx+1}"); sc2.audio(buf_c.getvalue())
            st.divider()

        # Section: Export Data (JSON & CSV)
        st.write("#### 📥 Export Analysis Data")
        st.info(generate_text_report(stats))
        
        all_rows = []
        for instr_type, event_list in st.session_state.stem_events.items():
            for evt in event_list:
                all_rows.append({
                    "Instrument": instr_type,
                    "Start Time (s)": round(evt['start'], 3),
                    "Duration (s)": round(evt['duration'], 3),
                    "End Time (s)": round(evt['start'] + evt['duration'], 3)
                })
        df_export = pd.DataFrame(all_rows).sort_values(by="Start Time (s)")
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            json_data = json.dumps(st.session_state.stem_events, indent=4)
            st.download_button(label="Download JSON", data=json_data, file_name="instrument_rhythm.json", mime="application/json", use_container_width=True)
        with col_dl2:
            csv_data = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download CSV", data=csv_data, file_name="instrument_rhythm.csv", mime="text/csv", use_container_width=True)

    # Section 4: Remix Result
    if st.session_state.timeline and st.session_state.final_audio is not None:
        st.divider(); st.subheader("4. Generated Remix Result")
        st.pyplot(plot_remix_waveform(st.session_state.final_audio, remixer.sr, st.session_state.timeline, st.session_state.final_dur))
        st.success(f"Remix Ready: {st.session_state.final_dur:.1f}s")
        buf = io.BytesIO(); sf.write(buf, st.session_state.final_audio, remixer.sr, format='WAV')
        st.audio(buf.getvalue()); st.download_button("📥 Download Remix WAV", buf, "remix.wav", type="primary")

else: st.info("👋 Please upload an audio file in the sidebar and run full analysis to begin.")