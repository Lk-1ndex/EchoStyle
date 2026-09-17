# EchoStyle 3.2 (基于多智能体FSM、6组严格单变量消融、20篇渐近收敛与模块化科学评测的个人文风建模系统)

从历史原创文章（微信公众号、Word、PDF）中自适应感知提取语料，结合 **统计语言学客观指纹 (Stylometrics, STTR, 节奏偏离度)** 与 **大模型语义深度解构** 建立高保真深层文风档案，依托 **Style-Aware Hybrid Memory (RRF 混合检索与篇章结构定向切片)** 长期积累，并通过 **严格有限状态机 (FSM)、快照回滚机制与多轮自审重构闭环 (Self-Reflection Loop)** 创作兼具作者呼吸节奏与独立思考质感的全新文章。

---

## 🌟 核心架构与科学严谨性演进 (EchoStyle 3.2)

1. **严格单变量消融实验套件 (Strict Single-Variable Matrix)**：
   - 严格单变量控制隔离 **Style Profile、Dense 纯密集语义 RAG、Hybrid RRF 融合检索、Style-Aware 结构过滤与独立盲审 Critic 自审** 各模块的独立边际贡献：
     - `Condition A (Vanilla Base 0)`: 无档案、无检索、无自审的通用基准
     - `Condition B (+Profile Only)`: 显式注入客观语言学统计特征与质感档案
     - `Condition C1a (+Dense RAG)`: 纯密集语义向量余弦召回（`dense_search`，严格执行 Fail-Closed 阻断，拒绝静默降级）
     - `Condition C1b (+Hybrid RRF RAG)`: 密集与稀疏词频倒数排名融合检索（无结构类型过滤，严格单变量隔离 RRF 融合机制）
     - `Condition C2 (+Style-Aware RAG)`: 基于篇章结构的定向装配检索（`retrieve_dynamic_few_shots` 覆盖 `hook`、`quote`、`argument`，**与 C1b 构成严格单变量对照：C2 - C1b**）
     - `Condition D (Full EchoStyle)`: **严格单变量受控与评测解耦**——复用 C2 检索快照与 C2 生成初稿 `art_c2`，由内部 Critic 自审重写；**终审由独立第三方盲审评测器 (Independent Evaluator, temperature=0.0) 统一打分，杜绝 Critic 既当运动员又当裁判的 Evaluator Overfitting**
   - **领域正交独立选题库 (Domain-Orthogonal Held-out Topics)**：消融评测题目涵盖职场、教育、城市生活、消费、旅行等跨领域题材，提示词仅保留纯客观任务与论述约束，彻底剥离风格特征词（无"犀利"、"体温"、"偏见"等提示），杜绝 Style Leakage。
   - **分层方差建模与独立盲审 (Hierarchical Clustered Statistics)**：严格区分 Topic 间宏观变异与重复采样扰动，采用 Student-t 分布计算 95% 置信区间，客观呈现系统多目标权衡。

2. **20 篇样本规模 Monte Carlo 子集重抽样收敛实验 (Monte Carlo Subsampling Scaling Experiment)**：
   - 彻底打破“仅固定截取前 N 篇样本”的序列偏差质疑，引入 **50 组 Monte Carlo 无放回随机子集重抽样 (Repeated Random Subsampling)** 评估 1 篇、3 篇、5 篇、10 篇与 20 篇样本梯度。
   - 引入语言学参数**均方误差 (MSE, Mean Squared Error)** 与 95% 置信区间：
     $$\text{MSE} = \frac{1}{4} \left[ \left(\frac{\Delta \bar{L}}{\bar{L}_{20}}\right)^2 + \left(\frac{\Delta \sigma}{\sigma_{20}}\right)^2 + \left(\frac{\Delta \text{STTR}}{\text{STTR}_{20}}\right)^2 + \left(\frac{\Delta \text{Entropy}}{\text{Entropy}_{20}}\right)^2 \right]$$
   - **统计严谨性澄清与 5 篇候选 Operating Point 定位**：
     - 20 篇全集 MSE=0 与收敛度 100% 为当前封闭池内的数理定义基准参照系（Mathematical Definition by Reference）；
     - 从 5 篇增至 10 篇，MSE 绝对值由 0.0098 降至 0.0043，在均方误差绝对尺度上下降了约 56.1%；之所以预设收敛度看起来仅从 91.0% 微升至 94.0%，是由于 $1 - \sqrt{\text{MSE}}$ 压缩尺度的数理特性；
     - 结合语料接入成本、检索计算开销与特征逼近收益，5 篇严谨表述为“**当前成本—特征误差权衡下的候选 Operating Point**”，而非绝对断言的黄金平衡点；结论严格建立在同作者语料池内，跨作者泛化仍需外部基准验证。

3. **统一多维评测目标 ($\text{EchoScore}$)**：
   - 确立统一量化目标函数，避免多指标冲突或帕累托前沿的解释混乱：
     $$\text{EchoScore} = 0.35 \times \text{Fidelity} + 0.25 \times \text{DiscourseFit} + 0.20 \times \text{RhythmMatch} + 0.20 \times \text{LexicalAuth} - \text{ClichePenalty}$$
   - 涵盖主观专家评分、宏观篇章拟合、微观节奏吻合度、词汇纯净质感与八股套话严厉惩罚。
   - *(注：当前加权系数为面向工程感知的启发式配比，非标准科学公式，后续将通过真实人类成对偏好进行回归拟合校准。)*

4. **节奏与韵律吻合度指标 (Rhythm Deviation)**：
   - 废除盲目惩罚低句长方差的“AI 匀称性假说”，保护严谨、学术或沉稳型写作者的天然文风。
   - 采用目标文风离散度偏离度：
     $$\text{RhythmDeviation} = \frac{|\bar{L}_{gen} - \bar{L}_{target}|}{\bar{L}_{target}} + \frac{|\sigma_{gen} - \sigma_{target}|}{\sigma_{target}} + |\text{Ratio}_{\text{short}, gen} - \text{Ratio}_{\text{short}, target}|$$
   - 衡量的是与作者本人的节奏吻合度，而非无差别强求句长剧烈起伏。

5. **篇章结构人手标注黄金基准 (`DISCOURSE_GOLD_BENCHMARK`)**：
   - 为避免“大模型自我循环论证 (LLM Self-Evaluation Loop)”，构建包含权威人工标注真值的标准篇章功能测试集。
   - 规则分类器与特征探测模块在黄金基准上达到 $\ge 90\%$ 的分类一致率（*当前为 11 样本启发式基准，存在规则共构局限，正推进构建 100+ 样本独立 held-out 集*）。

6. **任务感知启发式 Token 预算 (Task-Aware Heuristic Token Budget)**：
   - 采用 BPE 粒度与场景启发式预算调配（长文、短帖、重写、深度分析）。
   - **自适应故障重构**：当 Critic 触发反思改写时，自动为批注要点与批判约束增配 +5% 预算权重，已正式接入 WriterAgent 生产链路。

7. **分层多角色评估员面板 (Multi-Persona Evaluator Panel)**：
   - 引入三重视角评估员矩阵（离线回退支持客观特征裁决，杜绝固定偏见）：
     - **普通大众读者 (General Reader)**：最看重行文顺畅度、阅读通俗性与无造作感
     - **忠实读者 (Devoted Follower)**：对作者口癖、标志性观点与思维模式极度敏感
     - **资深主编 (Chief Editor)**：严打 AI 味、违规八股词与篇章结构松散

8. **系统失败模式与边界深度剖析 (Failure Modes Analysis)**：
   - 正视工程落地边界，归纳并给出架构级解决方案：
     - **CASE-01 跨领域题材冲突**：题材冲突侦测守卫与修辞平抑
     - **CASE-02 冷启动样本匮乏**：语料字数准入门槛与纯统计降级
     - **CASE-03 过度自审导致平庸化**：FSM 快照回滚保底机制 (`rollback_to('best_version')`，仅在超越历史最高分时更新快照)，已全面接入 Coordinator

9. **高度模块化评测架构 (`src/evaluation/`)**：
   - 彻底解耦巨型单体文件，拆分为 `lexical_metrics.py`、`rhythm_metrics.py`、`discourse_metrics.py`、`composite_eval.py`、`judge.py` 与兼容门面 `metrics.py`。

---

## 📊 科学评测与实验基准矩阵

### 1. 严格单变量消融实验套件与流程验证 (Ablation Study: Strict Single-Variable Matrix)

#### A. 5×5 仿真流程验证矩阵 (5×5 Simulation Pipeline Verification — NOT Benchmark Result)

> ⚠️ **免责声明与模式标记 (MOCK / SIMULATION PIPELINE VERIFICATION — NOT BENCHMARK RESULT)**：  
> 仓库当前已实现 5 topics × 5 repeats 自动化消融评估流水线，但尚未完成全量真实在线 LLM 的 25 轮并发实测。下表数据由离线分层统计桩 (Simulation Pipeline) 跑通生成，用于工程流程验证与多维指标统计框架测试，**绝不可视作最终实测 Benchmark 性能结论**。真实实测结果需配置有效 API Key 执行 `python main.py benchmark --ablation` 后生成归档于 `reports/ablation_study_report.md`。

| 消融条件 | Profile | Dense检索 | Hybrid RRF | 结构过滤 | Critic重写 | 独立盲审 | 篇章拟合 | 节奏吻合 | 用词质感 | 八股惩罚 | 跨主题均值 (Mean ± Std) | 95% 置信区间 (Student-t) | 严格单变量边际贡献与机制权衡 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A (Vanilla Base)** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 82.8 | 91.0 | 93.1 | -2.0 | **79.0 ± 0.8** | [78.1, 80.0] | 通用大模型零干预基准 |
| **B (+Profile Only)** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | 90.1 | 73.3 | 92.2 | -0.0 | **87.4 ± 0.7** | [86.5, 88.3] | **+8.4 分** (显式注入句法均值/方差与人设约束) |
| **C1a (+Dense RAG)**| ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | 94.7 | 99.0 | 94.9 | -0.0 | **93.4 ± 0.5** | [92.8, 94.0] | **+5.9 分 (相对 B)**: 密集语义召回连续语料段落，节奏高度平滑 |
| **C1b (+Hybrid RRF RAG)**| ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | 95.4 | 93.7 | 95.0 | -0.0 | **92.0 ± 0.5** | [91.4, 92.7] | **-1.4 分 (相对 C1a)**: 词频 RRF 召回稀疏词，增强词汇命中但稍微扰动纯密集连续性 |
| **C2 (+Style-Aware RAG)**| ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 96.6 | 84.0 | 95.0 | -0.0 | **89.5 ± 0.5** | [88.9, 90.1] | **-2.5 分 (相对 C1b 严格单变量)**: 篇章拟合最优(+1.2)，但金句与论据异构拼接打乱句长节奏 |
| **D (Full EchoStyle)**| ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 97.4 | 80.4 | 95.2 | -0.0 | **89.3 ± 0.4** | [88.8, 89.8] | **-0.2 分 (相对 C2 严格单变量)**: 内部 Critic 自审清除八股违规，由独立第三方盲审裁决 |

> 📌 **单次探索性摸底数据对照 (Single-Run Preliminary Result, 1 Topic × 1 Run)**：  
> 前期单次摸底数据为 A: 80.5, B: 88.8, C1: 87.8, C2: 83.9, D: 87.8 (+3.9)。因单次运行无法消除大模型生成与采样的偶然性噪声 (Std = 0.0，95% CI 退化为单点)，仅作为早期的单点探索记录，不作为最终稳态 benchmark 结论。


### 2. 20 篇样本规模 Monte Carlo 子集重抽样实测矩阵 (50-Iteration Repeated Random Subsampling)
| 样本规模配置 | 平均总字数 | 平均句长 (字) | 句长离散度 (σ) | 标准化 STTR | 切片数 | 均方误差 MSE (Mean ± Std) | MSE 95% 置信区间 | 综合收敛度 (Mean ± Std) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 篇 (极简冷启动)** | ~152 字 | 23.5 | 9.3 | 0.959 | 5 块 | 0.0484 ± 0.0349 | [0.0387, 0.0581] | **79.6% ± 8.3%** |
| **3 篇 (初步稳定)** | ~446 字 | 23.1 | 9.7 | 0.957 | 17 块 | 0.0173 ± 0.0140 | [0.0134, 0.0211] | **87.8% ± 4.9%** |
| **5 篇 (候选 Operating Point)** | ~743 字 | 23.3 | 9.7 | 0.955 | 28 块 | **0.0098 ± 0.0094** | **[0.0072, 0.0124]** | **91.0% ± 4.1%** |
| **10 篇 (深度建模)** | ~1475 字 | 23.2 | 9.7 | 0.957 | 54 块 | 0.0043 ± 0.0043 | [0.0031, 0.0054] | **94.0% ± 2.7%** |
| **20 篇 (全量封闭基准)** | 2953 字 | 23.1 | 10.0 | 0.963 | 94 块 | 0.0000 (定义基准) | [0.0000, 0.0000] | **100.0% (基准参照系)** |

> **实验科学结论与局限性澄清**：
> 1. 20 篇为封闭池内的定义基准（MSE=0 为参照系原点）；
> 2. 50 组 Monte Carlo 子集重抽样证实 5 篇样本时 MSE 均值降至 0.0098，方差显著收缩，收敛度突破 91.0%；
> 3. 从 5 篇增至 10 篇，MSE 绝对值仍下降了约 56.1% (从 0.0098 降至 0.0043)，在 $1 - \sqrt{\text{MSE}}$ 压缩尺度下变化约 3.0%；因此 5 篇严谨定位为“当前成本—特征误差权衡下的候选 operating point”；
> 4. 结论严格建立在当前作者同类语料池上，跨作者泛化需后续多作者数据集验证。

---

## 🚀 快速上手指南

### 1. 环境准备
项目基于 `uv` 进行快速可靠的依赖管理：
```bash
git clone https://github.com/Lk-1ndex/EchoStyle.git
cd EchoStyle

uv sync
.\.venv\Scripts\activate
```

### 2. 配置文件
编辑 `config.yaml` 填入你的大模型 API 密钥（兼容任何标准 OpenAI 协议，如 DeepSeek、OpenAI、Moonshot、Qwen 等）：
```yaml
llm:
  api_key: "sk-xxxxxx"
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  max_tokens: 4096

embedding:
  api_key: "sk-xxxxxx"
  base_url: "https://api.deepseek.com/v1"
  model: "text-embedding-v3"

agent:
  max_reflections: 2       # Critic 触发反思重写的最大轮次
  quality_threshold: 80.0  # 质检合格分阈值
```

---

## 🖥️ 运行方式

### 方式一：运行科学评测与基准实验套件 (CLI Benchmark Suite)
```bash
# 1. 运行五组严谨消融实验 (分离 Profile / 普通RAG / 风格RAG / Critic 独立贡献)
python main.py benchmark --ablation

# 2. 运行 20 篇样本规模渐近收敛实验 (验证 MSE 误差曲线与 5 篇黄金拐点)
python main.py benchmark --scaling

# 3. 运行规范化双盲评测 (输出加权胜率与 95% Wilson Score CI)
python main.py benchmark --blind

# 4. 运行系统失败案例与边界深度剖析 (跨领域冲突 / 冷启动退化 / 过度自审平庸化)
python main.py benchmark --failure

# 5. 运行完整 A/B 对照实验
python main.py benchmark --ab
```

### 方式二：启动 Web 可视化交互大屏
```bash
streamlit run src/web/app.py
```
- **【Tab 1: 样文感知与提取】**：输入公众号链接、Word 或 PDF 文档，查看正文抽取与去噪清洗。
- **【Tab 2: 深度文风指纹与记忆库】**：一键解构作者的客观 Stylometrics 指标（平均句长、标准差、STTR、标点熵），并实时测试 Style Memory 的向量检索与篇章结构定向召回。
- **【Tab 3: 智能创作与闭环评测】**：输入新主题，实时监控状态机跳转、草稿演进与 Critic 批注，查看成文与 EchoEval 多维评分卡。

### 方式三：CLI 命令行生产管道
```bash
# 1. 智能感知并提取文档
python main.py extract -s "https://mp.weixin.qq.com/s/xxxxxx" -o "sample.md"

# 2. 深度建模与记忆向量入库
python main.py distill -i "sample1.md" "sample2.md" -n "独立思考风"

# 3. 驱动 Coordinator 协作创作并输出评测报告
python main.py write -p "profiles/独立思考风_deep_profile.json" -t "为什么真挚的文风在当下更稀缺？" -o "final_article.md"
```

---

## 🧪 自动化测试验证与持续集成 (CI)

系统包含覆盖有限状态机、快照回滚重试控制、Fail-Closed 向量检索防降级、篇章感知过滤、清洗管道、客观语言学特征计算、双盲评测及动态 Token 预算的完整测试套件：
```bash
uv run pytest
```
所有 **全量单元与集成测试用例均自动化校验通过 (44+ passed, 1 skipped 用于隔离真实外连网络)**。

- **持续集成 (GitHub Actions)**：CI 工作流已正式部署于 [`.github/workflows/pytest.yml`](.github/workflows/pytest.yml)。
  > *注：若通过 GitHub CLI 或 PAT 提交工作流文件触发权限拦截（`refusing to allow a Personal Access Token to create or update workflow without workflow scope`），需在 GitHub Developer Settings 中为 Token 开启 `workflow` 作用域，或通过 GitHub Web 页面提交合并。*


---

## 📂 项目模块结构
```
EchoStyle/
├── src/
│   ├── core/          # 核心抽象 (DeepStyleProfile, FSM 状态机, 启发式 Token 预算 ModelProvider)
│   ├── extractors/    # 微信/Word/PDF 提取、版面复杂度感知 (inspector) 与文本净化 (sanitizer)
│   ├── analyzer/      # 统计语言学特征计算 (stylometrics, STTR, 标点熵) 与文风逆向蒸馏
│   ├── memory/        # 长期风格记忆库 (Style-Aware RAG, 倒数排名融合 RRF 混合检索)
│   ├── agents/        # 严格 FSM 智能体体系 (Coordinator, ToolRegistry, Writer, Critic, Extractor, Analyst)
│   ├── evaluation/    # 模块化 EchoEval 评测体系
│   │   ├── lexical_metrics.py     # 词汇级指标 (STTR, 套话惩罚)
│   │   ├── rhythm_metrics.py      # 节奏与韵律偏离度 (Rhythm Deviation)
│   │   ├── discourse_metrics.py   # 篇章推进逻辑与人工黄金标注基准 (Gold Benchmark)
│   │   ├── composite_eval.py      # 统一优化目标函数 (EchoScore)
│   │   ├── judge.py               # 多角色评估员面板 (General Reader / Follower / Chief Editor)
│   │   └── metrics.py             # 兼容统一门面 (Facade)
│   └── web/           # Streamlit 可视化工作台
├── experiments/       # 科学实验与评测基准套件
│   ├── ablation_study.py    # 6 组严格单变量消融实验 (6-Condition Matrix)
│   ├── scaling_study.py     # 20 篇样本规模渐近收敛实验 (MSE Curve)
│   ├── failure_analysis.py  # 失败案例与系统边界深度剖析
│   ├── blind_benchmark.py   # 规范化双盲评测套件
│   └── ab_benchmark.py      # 三方 A/B 对照基准
├── tests/             # 单元与集成测试套件 (44+ passed)
├── profiles/          # 文风档案、记忆切片与 Benchmark 评测报告 (Markdown / JSON)
├── benchmark.py       # 基准测试执行脚本
├── main.py            # CLI 命令行调度入口
├── pyproject.toml     # 项目依赖与构建元数据
└── config.yaml        # 运行时系统配置 (已纳入 .gitignore 安全隔离)
```

---

## 📄 License
MIT License.
