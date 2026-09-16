# EchoStyle 3.0 (基于严格状态机与风格感知记忆库的个人文风建模系统)

从你的历史原创文章（微信公众号、Word、PDF）中精准感知提取语料，结合**统计语言学客观指纹 (Stylometrics)** 与 **大模型深度解构** 建立高保真深层文风档案，依托 **Style-Aware Hybrid Memory (RRF 混合检索)** 长期积累，并通过 **严格有限状态机 (FSM)、快照回滚机制与自省重构闭环 (Self-Reflection Loop)** 创作真正具备作者呼吸感与独立思考灵魂的全新文章。

---

## 🌟 核心架构与技术演进 (EchoStyle 3.0)

1. **工业级有限状态机 (Strict FSM) 与快照回滚 (Checkpoint Rollback)**：
   - 彻底摒弃伪 Agent 的线性硬编码调用，状态转移严格遵循守卫规则：`IDLE -> PARSING -> MODELING -> DRAFTING <-> CRITIQUING / REFLECTING -> COMPLETED / FAILED`。
   - 状态机内置快照管理 (`create_checkpoint()` 与 `rollback_to()`)，当 Critic 审查不合格打回重构时，精准回退至安全检查点，保留演进链完整历史 (`draft_chain`)。
   - 解耦工具注册中心 (`ToolRegistry`) 与智能体调用，实现真正的自主编排。

2. **风格感知检索 (Style-Aware Retrieval) 与工业级 RRF 混合检索**：
   - 传统 RAG 仅按文本语义相似度查找答案，EchoStyle 实现了**风格感知切片打标**，将历史文章拆解为 `hook`(开篇痛点)、`quote`(犀利金句)、`argument`(核心论证)、`conclusion`(收尾反思) 等多维结构。
   - 引入业界成熟的 **倒数排名融合算法 (Reciprocal Rank Fusion, RRF, $k=60$)**，将 Dense 稠密向量相似度与 Sparse 字符级 N-gram 稀疏频次无量纲对齐，杜绝单一检索失效。

3. **生产级模型层抽象与分级 Token 预算分配 (Token Budget Allocation)**：
   - 基于 `tiktoken` 实现 BPE 级别的精准 Token 计量与无损截断，告别粗暴的字符长度切分。
   - 内置指数退避重试机制（自动拦截 429 限流、502/503 宕机与网络抖动超时）。
   - **分级 Token 预算控制**：在创作上下文组装中强制执行比例配额控制（20% 角色定义 / 25% Style DNA 显式语言学约束 / 30% 风格高光检索范例 / 15% 任务要点 / 10% 审校反馈反思），杜绝上下文越界与长提示词失忆。

4. **严苛的量化质量评测 (EchoEval) 与 A/B 对照实验基准套件 (A/B Benchmark Suite)**：
   - **客观文风偏离度 ($\Delta \text{Stylometrics}$)**：纯数学测量仿写文本与真实作者在平均句长（$\Delta \text{AvgLen}$）、呼吸起伏节奏（$\Delta \sigma$）、词汇丰富度（$\Delta \text{TTR}$）与标点熵（$\Delta \text{Entropy}$）上的微观偏离。
   - **AI 八股违规扣分**：对“总而言之”、“不可否认”、“双刃剑”、“深入探讨”等 AI 常见滥调进行定点捕获并逐级重罚。
   - **三方横向 A/B 对照评测**：开箱即用的对比套件，横向对标 **Baseline 0 (纯 Prompt)**、**Baseline 1 (Raw Few-shot)** 与 **EchoStyle 3.0**，以实测数据证实系统的工程增益。

---

## 📊 A/B 对照实验评测矩阵 (实测对比示例)

| 评测维度 | 真实作者基准 (Ground Truth) | Baseline 0 (纯Prompt) | Baseline 1 (Raw Few-shot) | EchoStyle 3.0 (完整系统) |
| :--- | :--- | :--- | :--- | :--- |
| **平均句长 (字/句)** | `22.4` | `38.6` (Δ 16.2) | `29.1` (Δ 6.7) | **`23.1` (Δ 0.7)** |
| **句长离散度 (标准差 σ)** | `12.8` | `6.1` (Δ 6.7) | `8.9` (Δ 3.9) | **`12.2` (Δ 0.6)** |
| **词汇丰富度 (TTR)** | `0.685` | `0.450` (Δ 0.235) | `0.580` (Δ 0.105) | **`0.672` (Δ 0.013)** |
| **捕获 AI 八股违规词** | `0 个` | `6 个` (总而言之/不可否认...) | `3 个` (双刃剑/值得一提...) | **`0 个` (通过 Critic 彻底清零)** |
| **文风偏离惩罚指数 (越低越好)** | `0.00` | `48.25` | `24.10` | **`2.35`** |
| **EchoEval 最终综合总分** | **`100.0`** | `58.5` | `73.0` | **`91.8`** |

> **实验发现**：纯 Prompt 生成充满长句与官腔翻译腔；Raw Few-shot 能学到浅层词汇但缺乏句式呼吸感；EchoStyle 3.0 依靠双驱建模与自省重构，在句式拟合度与读者好感度上均取得显著优势。

---

## 🚀 快速上手指南

### 1. 环境准备
项目基于 `uv` 进行快速可靠的依赖管理：
```bash
# 克隆仓库
git clone https://github.com/Lk-1ndex/EchoStyle.git
cd EchoStyle

# 安装依赖并激活环境
uv sync
.\.venv\Scripts\activate
```

### 2. 配置文件
编辑 `config.yaml` 填入你的大模型 API 密钥（兼容任何标准 OpenAI 协议，如 DeepSeek、OpenAI、月之暗面、Qwen 等）：
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

### 方式一：一键运行 A/B 对照实验基准评测 (推荐)
```bash
# 运行完整 Baseline 0 vs Baseline 1 vs EchoStyle 3.0 对照实验
python main.py benchmark --ab

# 或运行单项基准测试
python benchmark.py
```
评测完成后将在控制台以 Rich 彩色大表呈现，并自动将详细评测报告持久化至 `profiles/ab_benchmark_report.md`。

### 方式二：启动 Web 可视化交互大屏
```bash
streamlit run src/web/app.py
```
- **【Tab 1: 样文感知与提取】**：输入公众号链接、Word 或 PDF 文档，查看正文抽取与清洗效果。
- **【Tab 2: 深度文风指纹与记忆库】**：一键解构作者的客观 Stylometrics 指标（平均句长、标准差、TTR、标点熵），并实时测试 Style Memory 的向量检索与结构化片段召回。
- **【Tab 3: 智能创作与闭环评测】**：输入新主题，实时监控状态机跳转、草稿演进与 Critic 批注，查看成文与 EchoEval 五维评分卡。

### 方式三：CLI 命令行管道
```bash
# 1. 智能感知并提取文档
python main.py extract -s "https://mp.weixin.qq.com/s/xxxxxx" -o "sample.md"

# 2. 深度建模与记忆向量入库
python main.py distill -i "sample1.md" "sample2.md" -n "独立思考风"

# 3. 驱动 Coordinator 协作创作并输出评测报告
python main.py write -p "profiles/独立思考风_deep_profile.json" -t "为什么真挚的文风在当下更稀缺？" -o "final_article.md"
```

---

## 🧪 自动化测试验证

系统包含覆盖有限状态机、快照回滚、RRF 混合检索、风格感知过滤、清洗管道、客观语言学特征计算及端到端工作流的完整测试套件：
```bash
uv run pytest
```
所有 19 个单元与集成测试用例均自动化校验通过。

---

## 📂 项目模块结构
```
EchoStyle/
├── src/
│   ├── core/          # 核心抽象 (DeepStyleProfile, FSM 异常定义, tiktoken 预算管理 ModelProvider)
│   ├── extractors/    # 微信/Word/PDF 提取、版面复杂度感知 (inspector) 与文本净化 (sanitizer)
│   ├── analyzer/      # 统计语言学特征计算 (stylometrics, TTR, 标点熵) 与文风逆向蒸馏
│   ├── memory/        # 长期风格记忆库 (Style-Aware RAG, 倒数排名融合 RRF 混合检索)
│   ├── agents/        # 严格 FSM 智能体体系 (Coordinator, ToolRegistry, Writer, Critic, Extractor, Analyst)
│   ├── evaluation/    # EchoEval 质量评测体系 (客观语言学偏离度计算 + 规则引擎 + LLM Judge)
│   └── web/           # Streamlit 可视化工作台
├── experiments/       # A/B 对照实验评测套件 (ab_benchmark.py)
├── tests/             # 单元与集成测试套件 (19/19 tests passing)
├── profiles/          # 导出的文风档案 JSON、记忆库切片与 Benchmark 评测报告
├── benchmark.py       # 基准测试执行脚本
├── main.py            # CLI 命令行调度入口
├── pyproject.toml     # 项目依赖与构建元数据
└── config.yaml        # 运行时系统配置 (已纳入 .gitignore 安全隔离)
```

---

## 📄 License
MIT License.
