# LoopHunter - 音频循环与混音工作室

一个基于 Streamlit 的音频循环检测和混音工具。

## 功能特性

- 🎵 自动检测音频中的循环片段
- 🎛️ 智能混音和时长调整
- 📊 可视化音频波形和循环点
- 📥 导出分析报告（JSON 和文本格式）

## 安装依赖

```bash
pip3 install -r requirements.txt
```

或者如果 `pip3` 不可用，可以尝试：

```bash
python3 -m pip install -r requirements.txt
```

## 运行应用

```bash
streamlit run main.py
```

如果 `streamlit` 命令不可用，可以尝试：

```bash
python3 -m streamlit run main.py
```

应用将在浏览器中自动打开，默认地址为 `http://localhost:8501`

## 使用说明

1. **上传音频文件**：支持 MP3 和 WAV 格式
2. **分析音频**：点击 "🚀 Analyze Audio" 按钮开始分析
3. **查看循环**：浏览检测到的循环片段
4. **生成混音**：设置目标时长，点击 "✨ Generate Remix" 生成混音
5. **下载结果**：下载混音后的音频文件和分析报告

## 系统要求

- Python 3.8+
- 足够的系统内存（处理音频文件需要）

