# LoopHunter 智能音频重组解决方案说明文档

## 1. 整体架构 (System Architecture)

LoopHunter 是一个基于音乐信息检索（MIR）技术的智能音频重组系统。它采用 **分层架构 (Layered Architecture)**，将复杂的信号处理逻辑封装在后端，通过流式前端提供交互。

### 1.1 架构分层
* **表现层 (Frontend)**: `app.py` (Streamlit)
    * 负责文件上传、参数交互、波形可视化绘制、播放控制及下载。
* **核心引擎 (Core Engine)**: `division.py` -> `AudioRemixer` 类
    * **信号处理**: Librosa, NumPy, Scipy。
    * **特征工程**: Chroma (和声), MFCC (音色), RMS (能量), ZCR (过零率)。
    * **算法逻辑**: 自相似矩阵计算、动态规划路径搜索、微观波形对齐。
* **数据层 (Data)**:
    * 运行时内存 Session State。
    * 结构化输出: JSON 数据文档与 TXT 用户报告。

### 1.2 处理流水线
1.  **预处理**: 静音切除 (-60dB阈值) -> 瞬态增强 (HPSS)。
2.  **特征分析**: 节拍追踪 -> 特征提取 -> **动态分类** -> Loop 候选提取。
3.  **路径规划**: 根据 `Target Duration` 选择 **多Loop线性扩展** 或 **弹性全局缝合**。
4.  **合成渲染**: 微调切点 (Transient/Zero-Crossing) -> 交叉淡化 (Crossfade) -> 拼接。

---

## 2. 核心技术策略与逻辑 (Core Strategies)

### 2.1 Loop 提取与动态分类 (Extraction & Classification)

Loop 的提取不再仅仅依赖时域上的重复，而是引入了基于 **声学特征 (Acoustic Features)** 的动态分类器。

#### A. 提取算法 (Beat-Synchronous Shingling)
我们计算 **自相似矩阵 (Recurrence Matrix)**，但不是比对单一时刻，而是比对 **4拍序列 (4-Beat Shingles)** 的特征堆叠。这确保了提取出的 Loop 在乐句走向、和声进程上是完整的，而非偶然的音高重合。

#### B. 动态分类逻辑 (Dynamic Classification) [NEW]
为了智能地进行重组，系统会计算每一段 Loop 的 **相对能量 (Relative Energy)** 和 **频谱复杂度 (Spectral Complexity)**，将其归类为以下四种类型：

| Loop 类型 | 代码标识 | 判定逻辑 (阈值) | 音乐含义 |
| :--- | :--- | :--- | :--- |
| **高潮/副歌** | `climax` | `RMS_Ratio > 1.1` | 片段能量显著高于全曲平均水平（>110%）。通常是歌曲最激昂、乐器最丰富的部分。 |
| **氛围/铺垫** | `atmosphere`| `RMS_Ratio < 0.6` | 片段能量显著低于全曲平均水平（<60%）。通常是 Intro、Breakdown 或纯 Pad 音色段落。 |
| **节奏/鼓点** | `beats` | `ZCR > 0.08` | 能量适中，但 **过零率 (Zero Crossing Rate)** 较高。代表高频成分丰富或瞬态密集，通常是纯鼓点 (Drum Break)。 |
| **旋律/主歌** | `melody` | (Default) | 不符合上述条件的常规段落。通常是 Verse 或一般的器乐演奏。 |

### 2.2 智能重组策略 (Smart Remixing Strategy)

根据用户设定的目标时长 ($T_{target}$) 与原曲时长 ($T_{original}$) 的关系，系统自动切换算法。

#### 场景 A: 延长模式 (Extension) -> $T_{target} > T_{original}$
采用 **“多点线性循环规划 (Multi-Loop Linear Planning)”**。

1.  **筛选**: 选取互不重叠的高质量 Loop。
2.  **分配**: 计算需要填充的时间缺口，按权重分配给不同的 Loop（优先循环 `climax` 和 `melody`）。
3.  **构建**: 保持原曲线性结构（Intro -> Body -> Outro），在遇到被选中的 Loop 时，进行 $N$ 次“原地回跳”。

#### 场景 B: 缩短模式 (Shortening) -> $T_{target} < T_{original}$
采用 **“弹性加权全局缝合 (Elastic Weighted Stitching)”**。

1.  **目标**: 寻找最佳切出点 $A$ 和切入点 $B$，使得跳过中间部分后，剩余时长 $\approx T_{target}$。
2.  **全网格搜索**: 遍历所有节拍点，计算每一对 $(A, B)$ 的匹配度。
3.  **评分公式**:
    $$Score = Similarity(A, B) - ( |Time_{actual} - Time_{target}| \times Penalty )$$
    * **解释**: 我们允许为了更好的听感（高相似度）而稍微牺牲时长精度（容忍 2-5秒误差）。
4.  **结构保护**: 强制保留 Intro 头部和 Outro 尾部，只剪辑中间部分。

### 2.3 微观波形对齐 (Micro-Alignment)
为了消除拼接处的“爆音”和“吞音”，我们在渲染层实现了微调：
* **瞬态回溯**: 切点向前回溯 10-50ms，寻找波形起振点 (Onset)，保留鼓头。
* **过零点锁定**: 强制将切点吸附到最近的振幅为 0 的位置 (Zero-Crossing)。

---

## 3. 数据定义与文档解读 (Data Definitions)

### 3.1 分析数据结构 (`analysis_data.json`)

系统生成的 JSON 文档包含完整的分析元数据，可用于二次开发或存档。

```json
{
    "source_music": "night.mp3",       // 源文件名
    "total_duration": 140.93,          // 实际处理时长（去静音后）
    "looping_points": [                // 提取出的 Loop 对象数组
        {
            "duration": 56.47,         // Loop 长度 (秒)
            "start_position": 36.5,    // Loop 在原曲的入点 (秒)
            "type": "climax",          // [核心] 基于特征判定的类型
            "confidence": 0.76         // 置信度 (0-1)，基于自相似矩阵得分
        },
        {
            "duration": 11.33,
            "start_position": 28.03,
            "type": "melody",
            "confidence": 0.60
        }
        // ... 更多 Loop
    ]
}
````

### 3.2 Timeline 结构定义

在重组过程中生成的 Timeline 是音频合成的蓝图，其字段定义如下：

| 字段名 | 含义 | 备注 |
| :--- | :--- | :--- |
| `source_start` | 原曲读取开始点 | |
| `source_end` | 原曲读取结束点 | |
| `duration` | 片段时长 | |
| `type` | 片段类型 | 用于可视化着色。枚举值见下表。 |
| `xfade` | 交叉淡化时长 (ms) | `0` 表示自然连接，`>0` 表示需要混合。 |
| `is_jump` | 跳转标记 | `True` 表示此处发生了非线性跳转（Loop回跳或剪辑跳跃）。 |
| `refine_start` | 起点微调开关 | `True` 表示该点需要执行瞬态回溯算法。 |

### 3.3 可视化映射关系 (Visualization Mapping)

前端界面中的颜色和标记对应关系：

| 视觉元素 | 颜色 (Hex) | 对应数据逻辑 |
| :--- | :--- | :--- |
| **Head / Linear** | 🟩 `#238636` (Green) | 原曲线性播放部分 (Intro, Verse 等未被修改的段落) |
| **Loop Extension** | 🟦 `#1f6feb` (Blue) | `type="Loop Extension"`。通过算法循环生成的额外内容。 |
| **Tail (Jump)** | 🟪 `#d2a8ff` (Purple) | `is_jump=True` 且位于尾部。缩短模式下跳跃后的剩余部分。 |
| **Jump Marker** | ⚪ White Dot | `is_jump=True` 或 `xfade > 0` 的时间点。 |
| **Loop Highlight** | 🟦 Bright Blue | 原曲波形图中，对应 JSON `looping_points` 的区域。 |

-----

## 4\. 总结

LoopHunter 不仅仅是一个剪辑工具，它通过 **“听感特征分类”** 理解音乐内容的能量起伏，通过 **“弹性规划”** 在时长约束和音乐性之间取得最优解，并通过 **“微观对齐”** 保证了工业级的输出质量。
