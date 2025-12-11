

# LoopHunter 智能音频重组解决方案说明文档

## 1\. 整体架构 (System Architecture)

LoopHunter 采用经典的 **分层架构 (Layered Architecture)**，将前端交互与后端核心逻辑分离，确保了系统的可维护性和扩展性。

### 1.1 架构图示

  * **表现层 (Frontend)**: 基于 `Streamlit` 框架构建。负责用户交互（文件上传、参数设置）、可视化渲染（波形图、结构图）以及播放控制。
  * **逻辑层 (Core Engine)**: `division.py` 中的 `AudioRemixer` 类。负责音频信号处理、特征提取、路径规划算法及音频合成。
  * **数据层 (Data)**: 内存中的 Session State 管理，以及生成的 JSON/TXT 分析报告。

### 1.2 处理流程

1.  **输入**: 用户上传音频文件 (MP3/WAV)。
2.  **预处理**: 静音切除、加载音频数据。
3.  **分析 (Analysis)**: 提取节拍、色度特征、音色特征，计算自相似矩阵，提取 Loop 候选集。
4.  **规划 (Planning)**: 根据用户设定的 `Target Duration`，计算最优的播放路径（Timeline）。
5.  **合成 (Rendering)**: 基于 Timeline 进行音频切片、微调切点、交叉淡化（Crossfade）拼接。
6.  **输出**: 生成新音频、可视化图表及分析文档。

-----

## 2\. 核心技术策略 (Technical Strategies)

### 2.1 如何提取 Loop (Loop Extraction Algorithm)

Loop 的提取并非简单的寻找重复波形，而是基于**音乐语义**的深度分析。我们采用了 **“节拍同步特征堆叠 (Beat-Synchronous Feature Shingling)”** 技术。

1.  **静音切除 (Silence Trimming)**:
      * 在分析前，自动检测并切除音频末尾低于 -60dB 的静音部分，确保时长计算准确，同时保留 0.5s 的混响尾音。
2.  **节拍网格化 (Beat Tracking)**:
      * 使用 `HPSS` (Harmonic-Percussive Source Separation) 分离打击乐成分，利用 `librosa.beat.beat_track` 提取高精度的节拍点 (Beat Frames)。所有的分析均对齐到节拍网格上，而非绝对时间。
3.  **多维特征融合**:
      * **Chroma (和声)**: 捕捉旋律和和弦走向。
      * **MFCC (音色)**: 捕捉乐器织体和鼓点特征。
      * 两者归一化后堆叠，形成每一拍的综合声学指纹。
4.  **特征堆叠 (Shingling)**:
      * **关键步骤**: 我们不比较单拍的相似度，而是比较 **“4拍序列” (Stack Size = 4)** 的相似度。这确保了识别出的 Loop 不仅仅是瞬间音高相同，而是整句乐句的走向一致。
5.  **自相似矩阵与对角线扫描**:
      * 计算自相似矩阵 (Recurrence Matrix)。
      * 沿对角线扫描高相似度区域。如果位置 $i$ 和位置 $j$ 的特征序列高度相似（阈值从 0.85 动态降至 0.55），且长度超过 2 小节，则视为一个 Loop 候选。

### 2.2 如何基于用户时长重组 (Duration-Based Remixing)

重组引擎分为两种截然不同的策略，取决于用户设定的目标时长。

#### A. 延长模式 (Extension Mode): `Target > Original`

当需要延长音乐时，采用 **“多点线性循环规划”**。

1.  **筛选 (Filtering)**: 从候选 Loop 中筛选出互不重叠的高质量 Loop（例如 Verse Loop 和 Chorus Loop），保持其在原曲中的先后顺序。
2.  **计算 (Calculation)**: 计算需要填充的时间差 `Diff = Target - Original`。
3.  **分配 (Distribution)**: 将需要填充的时间按权重分配给各个 Loop。分数越高、长度适中的 Loop 会获得更多的循环次数。
4.  **线性构建**:
      * 播放线性片段 -\> 遇到 Loop 点 -\> **回跳循环 N 次** -\> 继续播放线性片段 -\> ... -\> 播放 Outro。
      * 这种方式完美保留了原曲的 Intro -\> Verse -\> Chorus -\> Outro 结构。

#### B. 缩短模式 (Shortening Mode): `Target < Original`

当需要缩短音乐时，采用 **“弹性加权全局缝合 (Elastic Weighted Stitching)”**。

1.  **目标定义**: 寻找两个时间点 $A$ 和 $B$ ($A < B$)，构建路径 `0 -> A` + `B -> End`。
2.  **弹性搜索**:
      * 我们需要满足：$Time(A) + (Duration - Time(B)) \approx Target$。
      * 算法在全曲节拍网格中搜索最佳的 $(A, B)$ 对。
3.  **评分机制**:
      * `Score = Similarity(A, B) - Time_Error * Penalty`。
      * 我们在“切口平滑度”（相似度）和“时长精准度”之间寻找平衡。允许为了更好的听感稍微牺牲一点时长精度（误差控制在 2-5秒内）。
4.  **结构保护**: 强制保留 Intro 的头部和 Outro 的尾部，只在中间部分进行“手术式”切除。

### 2.3 音质微调与无缝衔接 (Micro-Adjustment)

为了避免拼接处的“爆音”或“吞音”，我们在 `render` 阶段引入了 **微观波形对齐**：

  * **瞬态回溯 (Transient Backtracking)**: 切点不完全卡在节拍点上，而是向前回溯 10-50ms，寻找波形的起音 (Attack/Onset)，防止切断鼓头。
  * **过零点锁定 (Zero-Crossing Lock)**: 在切点附近微调采样点，强制在振幅为 0 处切割，消除物理上的 Click 噪音。
  * **动态交叉淡化 (Dynamic Crossfade)**: 跳转点应用 25-30ms 的等功率淡化，自然连接点则直接拼接。

-----

## 3\. 分析文档解读指南 (Output Interpretation)

系统会生成两份文档：机器可读的 `JSON` 和人机友好的 `TXT`。以下结合您提供的示例进行解读。

### 3.1 JSON 数据结构 (`analysis_data.json`)

这是用于二次开发或前端绘图的核心数据。

```json
{
    "source_music": "night.mp3",      // 原文件名
    "total_duration": 140.93,         // 原曲总时长（单位：秒）
    "looping_points": [               // 提取到的所有可用 Loop 列表
        {
            "duration": 56.47,        // Loop 的长度
            "start_position": 36.5,   // Loop 在原曲中的起始时间点
            "type": "climax",         // Loop 类型 (基于位置和能量推断: melody/climax/beats)
            "confidence": 0.76        // 置信度分数 (0-1)，越高代表 Loop 越完美
        },
        // ... 更多 Loop
        {
            "duration": 22.59,
            "start_position": 14.61,
            "type": "melody",
            "confidence": 0.74
        }
    ]
}
```

  * **解读逻辑**:
      * 如果你想做一个简单的 Extended Mix，选择 `confidence` 最高且 `duration` 较长的 Loop (如第一个)。
      * `start_position` 是 Loop 的起点，`start_position + duration` 是 Loop 的终点（也是回跳点）。

### 3.2 用户分析报告 (`analysis_report.txt`)

这是给最终用户看的摘要。

```text
========== AUDIO ANALYSIS REPORT ==========
Source File : night.mp3
Duration    : 140.93 seconds
Loops Found : 12                          <-- 总共找到了多少个可循环段落

--- DETAILED LOOP POINTS ---
Loop #01 | Start:  36.50s | Duration: 56.47s | Type: CLIMAX
Loop #02 | Start:  14.61s | Duration: 22.59s | Type: MELODY
Loop #03 | Start:  14.61s | Duration: 56.47s | Type: CLIMAX
...
```

  * **Loop \#01**: 这是算法认为“最强”的 Loop。
  * **Type: CLIMAX**: 提示这段音乐可能位于歌曲的高潮部分（通常能量较高或位于歌曲中后段）。
  * **Type: MELODY**: 提示这段可能是主歌或旋律部分。

### 3.3 Remix 结果可视化解读 (Web UI)

在生成 Remix 后，界面上的图表含义如下：

  * **SOURCE Waveform (上图)**:
      * **绿色区域**: 线性播放的区域（原汁原味）。
      * **蓝色/紫色区域**: 被选中的 Loop 区域。
  * **REMIX Structure (下图)**:
      * **Linear (绿色)**: 表示这段是顺序播放的。
      * **Loop Extension (蓝色)**: 表示这是通过回跳产生的“额外内容”。
      * **Jump Marker (小白点)**: 标记了具体的“手术缝合点”。
      * **Outro**: 保证是原曲的真实结尾。