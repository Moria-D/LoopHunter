import streamlit as st
import numpy as np
import soundfile as sf
import io, os, tempfile, pandas as pd, base64
import plotly.graph_objects as go
from division import AudioRemixer
from drum_processor import DrumLoopExtractor

# --- [UI 组件恢复] ---
def get_audio_player_html(audio_arr, sr):
    """HTML5 原生播放器：支持长进度条和无缝循环"""
    buf = io.BytesIO(); sf.write(buf, audio_arr, sr, format='WAV')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<div style="margin: 10px 0;"><audio controls loop style="width: 100%;"><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio></div>'

def plot_multi_track_overlay(audio_dict, sr):
    """交互式多轨叠加波形图"""
    fig = go.Figure()
    colors = {'kick': '#FF4B4B', 'snare_perc': '#00CC96', 'cymbals': '#636EFA', 'bass': '#AB63FA', 'instruments': '#FFA15A', 'sum': '#FFFFFF'}
    for name, y in audio_dict.items():
        if y is None or len(y) < 10: continue
        step = max(1, len(y) // 2500)
        fig.add_trace(go.Scatter(x=np.arange(0, len(y), step)/sr, y=y[::step], name=name.upper(), line=dict(color=colors.get(name, '#EEE'), width=1.5), opacity=0.6))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"), hovermode="x unified")
    return fig

def plot_individual_wave(y, sr, name, color):
    """单轨道波形图"""
    step = max(1, len(y) // 2500)
    fig = go.Figure(data=go.Scatter(x=np.arange(len(y[::step]))*step/sr, y=y[::step], mode='lines', line=dict(color=color, width=1.5), fill='tozeroy'))
    fig.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=30, b=30), title=f"Track: {name.upper()}", xaxis_title="Time (s)", showlegend=False)
    return fig

def to_excel_report(res):
    """高级 Excel 报告导出"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame(res.get('drum_sync', [])).to_excel(writer, sheet_name='Drum Sync', index=False)
        pd.DataFrame(res.get('global_sync', [])).to_excel(writer, sheet_name='Global Sync', index=False)
        all_m = []
        for k in ['kick','snare_perc','cymbals','bass','instruments']:
            if k in res:
                for m in res[k]['markers']: all_m.append({**m, "Track": k})
        pd.DataFrame(all_m).sort_values("Start Time (s)").to_excel(writer, sheet_name='Track Details', index=False)
    return output.getvalue()

st.set_page_config(layout="wide", page_title="LoopHunter - Master Sync")
st.title("🎛️ Audio Loop Studio (Precision Activity Sync)")

if 'remixer' not in st.session_state: st.session_state.remixer = None
if 'stem_audio' not in st.session_state: st.session_state.stem_audio = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

with st.sidebar:
    st.header("1. 音频处理")
    up = st.file_uploader("上传文件 (MP3/WAV)", type=["mp3", "wav"])
    if up and st.button("🚀 执行全功能分离分析", type="primary"):
        with st.spinner("提取轨道中..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as t:
                t.write(up.read()); tp = t.name
            rx = AudioRemixer(tp); rx.analyze(); st.session_state.remixer = rx
            st.session_state.stem_audio = rx.analyze_stems()
            os.remove(tp); st.success("提取完成!")

if st.session_state.remixer and st.session_state.stem_audio:
    st.divider()
    if st.button("🔍 开始精准划分同步片段 (剥离空白 & 6s 采样锁定)", type="primary"):
        extractor = DrumLoopExtractor(sr=st.session_state.remixer.sr)
        st.session_state.analysis_results = extractor.process_all_tracks(st.session_state.stem_audio, st.session_state.remixer.beat_times)

    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        st.download_button("📥 下载完整结构 Excel 报告", to_excel_report(res), "musical_analysis_report.xlsx", use_container_width=True)
        
        tab1, tab2, tab3 = st.tabs(["🥁 打击乐活跃同步", "🌎 全轨道活跃对齐", "📊 轨道深度详情"])
        
        with tab1:
            for idx, seg in enumerate(res.get('drum_sync', [])):
                with st.expander(f"活跃对齐段 #{idx+1} | {seg['Duration (s)']}s"):
                    s_s, e_s = int(seg['Start Time (s)']*st.session_state.remixer.sr), int(seg['End Time (s)']*st.session_state.remixer.sr)
                    seg_aud = {n: res[n]['audio'][s_s:e_s] for n in ['kick', 'snare_perc', 'cymbals'] if n in res}
                    st.plotly_chart(plot_multi_track_overlay(seg_aud, st.session_state.remixer.sr), use_container_width=True)
                    st.markdown(get_audio_player_html(np.sum(list(seg_aud.values()), axis=0), st.session_state.remixer.sr), unsafe_allow_html=True)

        with tab2:
            for idx, seg in enumerate(res.get('global_sync', [])):
                with st.expander(f"全局活跃片段 #{idx+1} | {seg['Duration (s)']}s"):
                    s_s, e_s = int(seg['Start Time (s)']*st.session_state.remixer.sr), int(seg['End Time (s)']*st.session_state.remixer.sr)
                    keys = [k for k in res if 'audio' in res[k]]
                    seg_aud = {n: res[n]['audio'][s_s:e_s] for n in keys}
                    st.plotly_chart(plot_multi_track_overlay(seg_aud, st.session_state.remixer.sr), use_container_width=True)
                    st.markdown(get_audio_player_html(np.sum(list(seg_aud.values()), axis=0), st.session_state.remixer.sr), unsafe_allow_html=True)

        with tab3:
            colors = {'kick': '#FF4B4B', 'snare_perc': '#00CC96', 'cymbals': '#636EFA', 'bass': '#AB63FA', 'instruments': '#FFA15A'}
            for t_n in [k for k in res if 'audio' in res[k]]:
                with st.container():
                    st.markdown(f"#### 轨道分析: {t_n.upper()}")
                    st.plotly_chart(plot_individual_wave(res[t_n]['audio'], st.session_state.remixer.sr, t_n, colors.get(t_n, '#FFF')), use_container_width=True)
                    st.markdown(get_audio_player_html(res[t_n]['audio'], st.session_state.remixer.sr), unsafe_allow_html=True)
                    if res[t_n].get('samples'):
                        st.write("🎹 唯一短 Loop 模式库 (活跃段采样):")
                        cols = st.columns(3)
                        for i, (l_type, l_audio) in enumerate(res[t_n]['samples'].items()):
                            with cols[i % 3]:
                                st.caption(f"Pattern ID: {l_type}")
                                fig_l = plot_individual_wave(l_audio, st.session_state.remixer.sr, l_type, colors.get(t_n, '#FFF'))
                                fig_l.update_layout(height=120, title="")
                                st.plotly_chart(fig_l, use_container_width=True)
                                st.markdown(get_audio_player_html(l_audio, st.session_state.remixer.sr), unsafe_allow_html=True)
                    st.divider()