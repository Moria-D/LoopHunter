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

# --- Analysis Helper Functions ---
def analyze_rhythm_structure(events, duration):
    """
    Generate statistical insights from rhythm data
    """
    stats = {}
    for instr, ev_list in events.items():
        if not ev_list:
            stats[instr] = {
                "count": 0, 
                "density_ppm": 0, 
                "avg_dur_ms": 0, 
                "active_ratio_pct": 0,
                "avg_gap_ms": 0
            }
            continue
            
        count = len(ev_list)
        total_active_time = sum(e['duration'] for e in ev_list)
        avg_dur = total_active_time / count if count > 0 else 0
        density = count / duration * 60  # events per minute
        active_ratio = total_active_time / duration * 100
        
        # Calculate intervals (silence between notes)
        intervals = []
        for i in range(len(ev_list) - 1):
            gap = ev_list[i+1]['start'] - (ev_list[i]['start'] + ev_list[i]['duration'])
            if gap > 0: intervals.append(gap)
        avg_interval = np.mean(intervals) if intervals else 0
        
        stats[instr] = {
            "count": count,
            "density_ppm": round(density, 1), # parts per minute
            "avg_dur_ms": round(avg_dur * 1000, 1),
            "active_ratio_pct": round(active_ratio, 1),
            "avg_gap_ms": round(avg_interval * 1000, 1)
        }
    return stats

def generate_text_report(stats):
    """
    Generate a natural language summary of the analysis - Dynamic for any instruments
    """
    report = []
    
    if not stats:
        return "No analysis data available."

    # 1. Dominant Instrument
    densities = {k: v['density_ppm'] for k, v in stats.items()}
    if densities:
        dominant = max(densities, key=densities.get)
        report.append(f"**Dominant Element:** The **{dominant.capitalize()}** leads the texture with the highest rhythmic density ({densities[dominant]} events/min).")
    
    # 2. Instrument Roles (Dynamic)
    sorted_instrs = sorted(stats.items(), key=lambda x: x[1]['active_ratio_pct'], reverse=True)
    
    for instr, data in sorted_instrs:
        name = instr.capitalize()
        active = data['active_ratio_pct']
        avg_dur = data['avg_dur_ms']
        
        desc = ""
        if active > 50:
            desc = f"Main sustained layer ({active}% active)"
        elif active > 20:
            desc = f"Core rhythmic element ({active}% active)"
        else:
            desc = f"Sparse accents or fills ({active}% active)"
            
        if avg_dur > 500:
            desc += " with long, sustained notes."
        elif avg_dur < 100:
            desc += " with short, percussive hits."
        else:
            desc += "."
            
        report.append(f"- **{name}:** {desc}")
        
    # 3. Complexity
    total_events = sum(s['count'] for s in stats.values())
    if total_events > 300:
        report.append("**Overall Texture:** High complexity with frequent interplay between layers.")
    elif total_events < 100:
        report.append("**Overall Texture:** Minimalist and spacious.")
    else:
        report.append("**Overall Texture:** Balanced rhythmic structure.")
        
    return "\n\n".join(report)

st.set_page_config(layout="wide", page_title="LoopHunter - Final UI")

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
if 'stem_events' not in st.session_state: st.session_state.stem_events = None
if 'stem_audio' not in st.session_state: st.session_state.stem_audio = None  # [New] Store audio for stems
if 'timeline' not in st.session_state: st.session_state.timeline = None
if 'final_audio' not in st.session_state: st.session_state.final_audio = None
if 'final_dur' not in st.session_state: st.session_state.final_dur = 0.0
if 'loop_page' not in st.session_state: st.session_state.loop_page = 0
if 'analysis_report' not in st.session_state: st.session_state.analysis_report = None

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

def plot_combined_waveforms(full_y, stems_audio, sr):
    fig = go.Figure()
    step = max(1, len(full_y) // 5000) # Target ~5k points for performance
    
    # Time axis
    time_axis = np.arange(0, len(full_y), step) / sr
    
    # 1. Full Mix (Background)
    fig.add_trace(go.Scatter(
        x=time_axis, y=full_y[::step],
        name="Full Mix",
        line=dict(color='gray', width=1),
        opacity=0.5
    ))
    
    # 2. Stems
    colors = px.colors.qualitative.Plotly
    for i, (name, y) in enumerate(stems_audio.items()):
        if len(y) == 0 or np.max(np.abs(y)) < 0.01: continue
        # Downsample stem to match full mix length logic
        y_down = y[::step]
        # Ensure lengths match
        min_len = min(len(time_axis), len(y_down))
        
        fig.add_trace(go.Scatter(
            x=time_axis[:min_len], y=y_down[:min_len],
            name=name.capitalize(),
            line=dict(color=colors[i % len(colors)], width=1),
            opacity=0.8,
            visible='legendonly' # Hide by default to keep it clean
        ))
        
    fig.update_layout(
        title="Combined Waveform View (Click legend to show/hide tracks)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_single_stem_waveform(y, sr, color):
    step = max(1, len(y) // 3000)
    y_down = y[::step]
    x_down = np.arange(0, len(y), step) / sr
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_down, y=y_down,
        mode='lines',
        line=dict(color=color, width=1),
        fill='tozeroy'
    ))
    fig.update_layout(
        template="plotly_dark",
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_instrument_timeline_interactive(events, total_duration):
    """
    使用 Plotly 绘制交互式乐器节奏图 (Gantt-like chart) - 支持任意乐器键名
    """
    # 1. 转换数据为 DataFrame
    data = []
    
    # 动态生成颜色映射
    # 使用 Plotly 默认色板或自定义色板循环
    color_palette = px.colors.qualitative.Pastel
    unique_instruments = sorted(list(events.keys()))
    colors_map = {instr.capitalize(): color_palette[i % len(color_palette)] for i, instr in enumerate(unique_instruments)}
    
    # 覆盖默认颜色以保持一致性 (如果存在)
    default_colors = {'Melody': '#d2a8ff', 'Bass': '#79c0ff', 'Drums': '#ff7b72'}
    for k, v in default_colors.items():
        if k in colors_map:
            colors_map[k] = v
            
    for instr, ev_list in events.items():
        for ev in ev_list:
            data.append(dict(
                Instrument=instr.capitalize(),
                Start=ev['start'],
                End=ev['start'] + ev['duration'],
                Duration=ev['duration']
            ))
            
    if not data:
        return None
        
    df = pd.DataFrame(data)
    
    # 2. 绘制 Gantt 图 (使用 px.bar)
    fig = px.bar(
        df, 
        base="Start", 
        x="Duration", 
        y="Instrument", 
        color="Instrument",
        orientation='h',
        color_discrete_map=colors_map,
        hover_data={"Start": ":.2f", "End": ":.2f", "Duration": ":.2f", "Instrument": False},
        height=300
    )
    
    # 3. 布局美化
    fig.update_layout(
        xaxis=dict(
            title="Time (s)",
            showgrid=True,
            gridcolor='#30363d',
            zerolinecolor='#30363d',
            range=[0, total_duration]
        ),
        yaxis=dict(
            title="",
            categoryorder='total ascending' # 按活跃度排序
        ),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        margin=dict(l=10, r=10, t=30, b=30),
        showlegend=False,
        hovermode='closest'
    )
    
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

with st.sidebar:
    st.header("1. Upload")
    uploaded_file = st.file_uploader("Audio", type=["mp3", "wav"])
    
    if uploaded_file:
        st.write("")
        if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
            with st.spinner("Scanning structure & loops..."):
                suffix = ".mp3" if uploaded_file.name.endswith(".mp3") else ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(uploaded_file.read())
                    tpath = tfile.name
                remixer = AudioRemixer(tpath)
                remixer.analyze()
                st.session_state.remixer = remixer
                # [Update] Unpack events and audio
                events, stems_audio = remixer.analyze_stems()
                st.session_state.stem_events = events
                st.session_state.stem_audio = stems_audio
                
                st.session_state.timeline = None
                st.session_state.final_audio = None
                st.session_state.loop_page = 0 
                json_data, text_data = remixer.export_analysis_data(uploaded_file.name)
                st.session_state.analysis_report = {"json": json_data, "text": text_data}
                os.remove(tpath)
                count = len(remixer.loops)
                if count > 0: st.success(f"Found {count} loops!")
                else: st.warning("No loops found.")

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
                if 0 <= new_page < total_pages: st.session_state.loop_page = new_page
            st.number_input("Jump to Page", min_value=1, max_value=total_pages, value=current_page + 1, step=1, key="page_input", on_change=update_page_number, label_visibility="collapsed")
            st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.8em;'>of {total_pages} pages</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("Next ▶", disabled=(end_idx == total_loops), use_container_width=True):
                st.session_state.loop_page += 1
                st.rerun()
    else:
        st.info("No loops detected.")

    if st.session_state.stem_events:
        st.divider()
        st.subheader("3. Instrument Rhythm Breakdown")
        st.caption("Interactive visualization: Zoom, Pan, and Hover to see details.")
        
        # Plotly Chart
        fig_inst = plot_instrument_timeline_interactive(st.session_state.stem_events, remixer.duration)
        if fig_inst:
            st.plotly_chart(fig_inst, use_container_width=True)
            
        # Download Data
        st.write("#### Download Rhythm Data")
        c1, c2 = st.columns(2)
        
        # Prepare JSON
        rhythm_json = json.dumps(st.session_state.stem_events, indent=4)
        
        # Prepare Excel/CSV friendly format
        flat_data = []
        for instr, ev_list in st.session_state.stem_events.items():
            for ev in ev_list:
                flat_data.append({
                    "Instrument": instr,
                    "Start (s)": round(ev['start'], 4),
                    "Duration (s)": round(ev['duration'], 4),
                    "End (s)": round(ev['start'] + ev['duration'], 4)
                })
        df_rhythm = pd.DataFrame(flat_data)
        csv_data = df_rhythm.to_csv(index=False).encode('utf-8')
        
        with c1:
            st.download_button(
                label="📥 Download JSON",
                data=rhythm_json,
                file_name="instrument_rhythm.json",
                mime="application/json",
                use_container_width=True
            )
        with c2:
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="instrument_rhythm.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        # Analysis Report
        st.write("#### 📊 Structural Analysis Report")
        stats = analyze_rhythm_structure(st.session_state.stem_events, remixer.duration)
        
        # --- NEW: Combined Waveform ---
        if st.session_state.stem_audio:
            st.subheader("Waveform Overview")
            fig_combined = plot_combined_waveforms(remixer.y, st.session_state.stem_audio, remixer.sr)
            st.plotly_chart(fig_combined, use_container_width=True)
            
            st.write("#### Individual Stems Breakdown")
            # Generate Colors
            colors_list = px.colors.qualitative.Plotly
            
            for i, (instr, data) in enumerate(stats.items()):
                # Create a container for each stem
                with st.container():
                    col_info, col_wave = st.columns([1, 5])
                    
                    # Info Column
                    with col_info:
                        st.markdown(f"**{instr.capitalize()}**")
                        st.caption(f"Density: {data['density_ppm']} ppm")
                        st.caption(f"Active: {data['active_ratio_pct']}%")
                        
                    # Waveform & Player Column
                    with col_wave:
                        if instr in st.session_state.stem_audio:
                            color = colors_list[i % len(colors_list)]
                            fig_stem = plot_single_stem_waveform(st.session_state.stem_audio[instr], remixer.sr, color)
                            st.plotly_chart(fig_stem, use_container_width=True, config={'displayModeBar': False})
                            
                            # Player - Now in the wider column for better seek bar
                            stem_y = st.session_state.stem_audio[instr]
                            if np.any(stem_y):
                                max_val = np.max(np.abs(stem_y))
                                if max_val > 0: stem_y = stem_y / max_val * 0.95
                                buf = io.BytesIO()
                                sf.write(buf, stem_y, remixer.sr, format='WAV')
                                st.audio(buf.getvalue(), format='audio/wav')
                    
                    st.divider()
        
        # Text Report
        st.info(generate_text_report(stats))

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