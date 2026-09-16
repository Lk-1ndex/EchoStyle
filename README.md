# EchoStyle 2.0 (基于 Multi-Agent 与风格记忆库的个人文风建模与智能创作系统)

从你的历史原创文章（微信公众号、Word、PDF）中精准感知提取语料，结合**统计语言学 (Stylometrics)** 与 **大模型质性解构** 建立高保真深层文风档案，依托 **Style Memory (RAG 驱动)** 长期积累，并通过 **多智能体协作与自省反思闭环 (Self-Reflection Loop)** 创作真正具备作者呼吸感与灵魂的全新文章。

---

## 🌟 核心升级亮点 (v2.0)

1. **多智能体协同架构 (Multi-Agent Architecture)**：
   - **`Coordinator Agent`**：编排全局任务流，管理执行轨迹与自省重试状态机。
   - **`Extractor Agent`**：具备排版感知能力，智能探测文档排版复杂度（单栏/双栏/扫描件），自适应分派极速或深度视觉解析引擎。
   - **`Analyst Agent`**：执行统计客观量化与语义深度解构，自动对历史语料切片入库。
   - **`Writer Agent`**：面向长期记忆库动态语义召回最契合的 Few-shot 范例，结合文风约束精准创稿。
   - **`Critic Agent`**：担当总编辑，执行严格的去 AI 味审查、统计吻合度核验与质量评分，未达标时打回并触发**自省重构（Self-Reflection Loop）**。

2. **深层风格建模：客观量化 (Stylometrics) + 质性解构**：
   - 纯数学计算作者行文的**真实平均句长**、**句长离散波动标准差（呼吸节奏波长）**、**标点符号信息熵**与**高频转折词密度**。
   - 与叙事视角、情感基调、口头禅、篇章结构与负向禁令深度融合为 `DeepStyleProfile`。

3. **长期风格记忆库 (Style Memory RAG)**：
   - 解决文章累积过多时的 Context Window 上下文膨胀问题。
   - 支持语义切片与段落感知（引言、论述、金句、结尾），每次创作根据新选题**动态召回 3~5 个最相关的高光真实段落**作为 Few-shot。
   - 支持双模态检索（远程 Embedding 向量检索 + 离线零依赖本地余弦相似度检索）。

4. **量化质量评测体系 (EchoEval)**：
   - **去 AI 味得分**：基于规则与八股负向词库进行严苛扣分。
   - **句式统计拟合度**：数学比对成文与作者历史库在句长与节奏上的偏离度。
   - **LLM-as-a-Judge**：独立评委视角打分并输出五维雷达指标与具体批注。

---

## 🚀 快速上手指南

### 1. 激活虚拟环境
项目基于 `uv` 管理依赖与环境：
```bash
.\.venv\Scripts\activate
```

### 2. 配置文件
编辑 `config.yaml` 填入你的大模型 API 密钥（兼容任何 OpenAI 协议，如 DeepSeek、OpenAI、月之暗面等）：
```yaml
llm:
  api_key: "sk-xxxxxx"
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"

agent:
  max_reflections: 2       # Critic 触发反思重写的最大轮次
  quality_threshold: 80.0  # 质检合格分阈值
```

---

## 🖥️ 方式一：启动 Web 可视化工作台（推荐）

在终端中执行：
```bash
.\.venv\Scripts\streamlit run src/web/app.py
```
浏览器将自动弹出交互控制台：
- **【Tab 1: 样文感知与提取】**：粘贴公众号链接或上传文档，查看 Extractor Agent 的排版决策与纯净正文。
- **【Tab 2: 深度文风指纹与记忆】**：一键计算平均句长、句长标准差、标点熵，查看质性风格并支持在记忆库中测试动态 Few-shot 召回。
- **【Tab 3: 智能创作与量化评估】**：输入新主题，实时监控 Multi-Agent 的协作与反思轨迹，生成终审成文并查收 **EchoEval 评分卡** 与雷达维度数据！

---

## ⌨️ 方式二：CLI 命令行使用

```bash
# 1. 智能感知并提取文档
python main.py extract -s "https://mp.weixin.qq.com/s/xxxxxx" -o "sample.md"

# 2. 深度建模与记忆向量入库
python main.py distill -i "sample1.md" "sample2.md" -n "我的深度文风"

# 3. 驱动 Multi-Agent 创作并输出评测报告
python main.py write -p "profiles/我的深度文风_deep_profile.json" -t "在机器泛滥的时代，为什么真挚的文风更值钱？" -o "新文章.md"
```

---

## 📂 项目结构
```
write/
├── src/
│   ├── core/          # 核心数据模型 (DeepStyleProfile, EvaluationReport) 与配置
│   ├── extractors/    # 微信/Word/PDF 提取、复杂度感知器 (inspector) 与去噪管道
│   ├── analyzer/      # 统计语言学计算引擎 (stylometrics) 与文风逆向蒸馏器
│   ├── memory/        # 长期风格记忆库 (Style Memory RAG，向量切片与动态召回)
│   ├── agents/        # Multi-Agent 体系 (Coordinator, Extractor, Analyst, Writer, Critic)
│   ├── evaluation/    # EchoEval 质量评测体系 (规则引擎 + LLM Judge)
│   └── web/           # 升级版 Streamlit 交互大屏
├── tests/             # 单元与集成测试套件 (11/11 passed)
├── main.py            # CLI 命令行调度入口
└── config.yaml        # 系统配置文件
```
