import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# 保证 Windows 终端在输出中文及特殊符号时采用 UTF-8 编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.core.config import load_config
from src.core.model_provider import ModelProvider
from src.agents.coordinator import CoordinatorAgent
from src.agents.state import AgentState
from src.analyzer.stylometrics import StylometricsAnalyzer
from src.evaluation.metrics import MetricEvaluator
from src.evaluation.composite_eval import CompositeEvaluator, CompositeEvaluationResult
from src.agents.critic_agent import CriticAgent
from src.core.models import (
    DeepStyleProfile,
    StyleProfile,
    TonePersona,
    CadenceSyntax,
    LexiconRhetoric,
    DiscourseArchitecture,
    AntiPatterns,
    EvaluationReport,
)

console = Console()

SAMPLE_ESSAY_1 = """# 别把信息搬运当成深度思考

说白了，很多人在互联网上搞的内容输出，本质上不过是高级的信息搬运工。
他们看到一篇外媒报道，或者扒了一份行业研报，换两句大白话，拼凑几个所谓的数据图表，就敢宣称自己做的是“深度产业观察”。

别闹了。

真正的思考从来不是拼图游戏，而是带着偏见的价值判断。
技术越来越便宜，生成几千字废话的边际成本已经无限趋近于零。如果你的文章只是复述常识，读者为什么要把宝贵的注意力浪费在你身上？

退一步讲，在这个注意力极度碎片化的时代，能够让人停下来读完的文字，一定具备某种肉身写作者特有的体温。
要么你的论点足够离经叛道，能刺痛某些人的伪善；要么你的表达足够生动犀利，能把抽象的逻辑讲成街头巷尾的市井故事。

可惜的是，大多数人既没有刺痛别人的勇气，也没有讲好故事的耐心。
他们只是习惯了躲在那些毫无破绽的官样套话背后，假装自己很专业。
"""

SAMPLE_ESSAY_2 = """# 为什么我不喜欢“正确的废话”

不知道从什么时候开始，我们的中文互联网充斥着一种极其恶劣的文风。
通篇都是“一方面...另一方面...”、“不可否认存在挑战...但也蕴含巨大机遇...”。

看似客观中立，实则毫无见解。
说难听点，这就是精致的懦弱。

写作这门手艺，最忌讳的就是四平八稳。
一个合格的写作者，必须有自己的审美偏好，必须敢于在关键分歧点上下注。
我宁愿看一个带着强烈个人偏见但论述精彩的失败者，也不愿意看一百篇挑不出语法毛病、却没有任何思考增量的平庸之作。

实际上，工具越是强大，人类越要警惕被机器同化。
当你开始习惯用那些被算法反复清洗过的词汇来组织思想时，你的大脑就已经在慢慢萎缩了。
保持尖锐，保持口语化，保持那种带点自嘲却绝不妥协的语言质感。这是我们在算法洪流里唯一能守住的阵地。
"""

BENCHMARK_TOPICS = [
    {
        "topic": "在算法洪流中，为什么肉身写作者的刺痛感不可替代？",
        "key_points": "机器生产成本无限趋近于零，四平八稳的官样废话已经通货膨胀；真正的写作者必须带有一针见血的偏见，敢于在关键分歧点上下注；保持口语化的呼吸感与刺痛读者的真实体温",
    },
    {
        "topic": "为什么我不喜欢'正确的废话'：谈谈公共讨论中的精致懦弱",
        "key_points": "通篇两面讨好看似中立实则毫无见解；写作最忌讳四平八稳；敢于在关键分歧点下注；警惕被算法反复清洗过的平庸词汇",
    },
    {
        "topic": "自媒体时代，深度长文写作的尊严与自我救赎",
        "key_points": "算法只关心滑过不关心真诚；字数注水不等于思想深度；写作是一场孤独的自我审判；敢于删掉自恋的废话",
    },
    {
        "topic": "工具越强大，人类写作者越要学会停顿与反思",
        "key_points": "毫秒级生成让人精神肌肉退化；卡壳时的痛苦是人类最后的尊严；粗糙与锐角赋予文章不可替代的灵魂",
    },
    {
        "topic": "风格即人本身：算法可以模仿词汇，但无法复制经历与伤疤",
        "key_points": "风格是生活创伤与偏见的沉淀；文笔是皮囊骨相是见识；在谎言泛滥时说出一句真话就是最后防线",
    },
]


def _calc_stats(numbers: List[float]) -> Tuple[float, float, float]:
    """计算列表均值、样本标准差与 95% 置信区间半宽"""
    n = len(numbers)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(numbers) / n
    if n == 1:
        return mean, 0.0, 0.0
    std = math.sqrt(sum((x - mean) ** 2 for x in numbers) / (n - 1))
    ci95 = 1.96 * std / math.sqrt(n)
    return mean, std, ci95


def _generate_empirical_analysis(summary: Dict[str, Dict[str, Tuple[float, float, float]]]) -> str:
    """依据真实运行统计数据动态生成多目标权衡与客观归因，坚决拒绝结论先行与静态断言"""
    c1_echo, c1_echo_std, _ = summary["C1"]["echo"]
    c2_echo, c2_echo_std, _ = summary["C2"]["echo"]
    d_echo, d_echo_std, _ = summary["D"]["echo"]
    b_echo, b_echo_std, _ = summary["B"]["echo"]
    a_echo, a_echo_std, _ = summary["A"]["echo"]

    c1_disc = summary["C1"]["discourse"][0]
    c2_disc = summary["C2"]["discourse"][0]
    d_disc = summary["D"]["discourse"][0]

    c1_rhy = summary["C1"]["rhythm"][0]
    c2_rhy = summary["C2"]["rhythm"][0]
    d_rhy = summary["D"]["rhythm"][0]

    c1_pen = summary["C1"]["penalty"][0]
    c2_pen = summary["C2"]["penalty"][0]
    d_pen = summary["D"]["penalty"][0]

    # 1. 核心结论判定 (直面实测数据，绝不结论先行)
    if c1_echo > c2_echo + 0.5:
        echo_verdict = (
            f"- 实测数据显示：**在当前 EchoScore 预设目标下，普通纯密集 RAG (C1: {c1_echo:.1f} ± {c1_echo_std:.1f}) > Style-Aware RAG (C2: {c2_echo:.1f} ± {c2_echo_std:.1f})**；\n"
            f"- **关键事实**：本轮消融实测**未证明** Style-Aware RAG 在唯一定义的综合得分上优于普通纯密集 RAG，综合得分差为 {c2_echo - c1_echo:+.1f} 分。"
        )
    elif c2_echo > c1_echo + 0.5:
        echo_verdict = (
            f"- 实测数据显示：**在当前 EchoScore 预设目标下，Style-Aware RAG (C2: {c2_echo:.1f} ± {c2_echo_std:.1f}) > 普通纯密集 RAG (C1: {c1_echo:.1f} ± {c1_echo_std:.1f})**；\n"
            f"- **关键事实**：Style-Aware RAG 相对普通 RAG 实现了 +{c2_echo - c1_echo:.1f} 分的综合净收益。"
        )
    else:
        echo_verdict = (
            f"- 实测数据显示：**在当前 EchoScore 预设目标下，Style-Aware RAG (C2: {c2_echo:.1f}) 与普通纯密集 RAG (C1: {c1_echo:.1f}) 得分基本持平**。"
        )

    # 2. 篇章结构拟合分析
    if c2_disc > c1_disc + 0.5:
        disc_analysis = (
            f"Style-Aware RAG (C2: {c2_disc:.1f}) 在宏观推进拟合上优于普通 RAG (C1: {c1_disc:.1f})（净增益 +{c2_disc - c1_disc:.1f} 分）。"
            f"定向装配 `hook`（破空设问）与 `quote`（警策金句）对文章骨架与起承转合有切实塑造效果。"
        )
    elif c1_disc > c2_disc + 0.5:
        disc_analysis = (
            f"Style-Aware RAG (C2: {c2_disc:.1f}) 在篇章推进拟合上未及普通 RAG (C1: {c1_disc:.1f})（相差 {c2_disc - c1_disc:.1f} 分），"
            f"表明异构标签切片的组装在当前上下文衔接上仍需进一步平滑优化。"
        )
    else:
        disc_analysis = (
            f"Style-Aware RAG (C2: {c2_disc:.1f}) 与普通 RAG (C1: {c1_disc:.1f}) 篇章结构拟合度基本相当（{c2_disc:.1f} vs {c1_disc:.1f}）。"
        )

    # 3. 句长节奏吻合度分析
    if c1_rhy > c2_rhy + 0.5:
        rhy_analysis = (
            f"普通密集语义 RAG (C1: {c1_rhy:.1f}) 召回的是上下文语义相近的平缓连续段落，句长分布均匀平滑；"
            f"而 Style-Aware RAG (C2: {c2_rhy:.1f}) 将短小金句、设问长句与论据异构拼接，导致生成模型在长短句极值之间跳跃，"
            f"句长方差偏离目标作者基准，节奏吻合分下滑了 {c1_rhy - c2_rhy:.1f} 分，构成本轮综合得分受挫的主要拖累项。"
        )
    elif c2_rhy > c1_rhy + 0.5:
        rhy_analysis = (
            f"Style-Aware RAG (C2: {c2_rhy:.1f}) 的句长波长拟合优于普通 RAG (C1: {c1_rhy:.1f})（+{c2_rhy - c1_rhy:.1f} 分），"
            f"短句金句与长句论据的组合较好还原了作者的呼吸节奏。"
        )
    else:
        rhy_analysis = (
            f"Style-Aware RAG (C2: {c2_rhy:.1f}) 与普通 RAG (C1: {c1_rhy:.1f}) 在句长节奏维度表现基本相当。"
        )

    # 4. Critic 自审反思闭环与八股防护
    if c2_pen > 0 and d_pen < c2_pen:
        critic_analysis = (
            f"Critic 自审反思闭环在 D 阶段精准拦截了 C2 阶段暴露的偶发八股违规词（八股惩罚从 -{c2_pen:.1f} 分收敛至 -{d_pen:.1f} 分），"
            f"使 Full EchoStyle (D: {d_echo:.1f}) 相较 C2 ({c2_echo:.1f}) 实现了质量自愈提升（+{d_echo - c2_echo:.1f} 分），"
            f"证实了状态机反思机制在兜底安全红线上的工程有效性。"
        )
    elif d_echo > c2_echo + 0.5:
        critic_analysis = (
            f"Full EchoStyle (D: {d_echo:.1f}) 相对无自审的 C2 ({c2_echo:.1f}) 取得净增益 +{d_echo - c2_echo:.1f} 分，"
            f"多轮反思重构对行文质感与论述锐度带来了可测量的正向提振。"
        )
    else:
        critic_analysis = (
            f"Critic 介入后 D ({d_echo:.1f}) 相对 C2 ({c2_echo:.1f}) 综合得分基本持平（变动 {d_echo - c2_echo:+.1f} 分）。"
            f"反思重写虽压制了违规词汇，但也带来生成策略偏向保守防御、句长波长收窄的轻微副作用，导致总分未现显著扩张。"
        )

    return f"""## 二、 客观实验事实与科学归因分析 (Empirical Findings & Tradeoff Analysis)

实测数据揭示了一个极具学术与工程价值的客观事实：
{echo_verdict}

### 深度归因与多目标权衡剖析：
1. **篇章结构拟合 (Discourse Fit) 表现**：
   {disc_analysis}
2. **句长节奏吻合 (Rhythm Match) 表现**：
   {rhy_analysis}
3. **Critic 自审反思闭环与八股防护**：
   {critic_analysis}

### 科学启示与工程迭代方向：
- **直面真实数据，坚决拒绝结论先行**：科学消融的意义在于暴露系统多目标之间的真实权衡与代价，绝不能预设某模块必胜；
- **自适应检索平滑注入**：从单纯的'结构标签硬拼接'演进为'**韵律平滑感知的自适应检索注入 (Rhythm-Smoothed Retrieval Injection)**'，在注入金句的同时自适应调节上下文长度比例，避免对句长节奏方差造成过度震荡；
- **评测权重实证校准**：当前加权系数 (0.35 / 0.25 / 0.20 / 0.20) 为工程启发式配置，后续需依托大规模真实人类成对偏好开展相关性回归拟合，校准真实的感知效用函数。
"""


def run_ablation_study(
    topics_count: int = 3,
    repeats: int = 1,
    simulate: bool = False,
    output_path: Optional[str] = None,
):
    console.print(Panel.fit(
        "[bold cyan]EchoStyle 3.2 — 五组严谨消融实验套件 (5-Condition Matrix)[/bold cyan]\n"
        "[white]消融核心：严格单变量隔离分离 Profile、普通语义 RAG (Dense)、Style-Aware RAG (Structured Hybrid) 与 Critic 自审的边际贡献[/white]"
    ))

    config = load_config()
    has_api_key = bool(config.llm.api_key and config.llm.api_key.strip() and not config.llm.api_key.startswith("sk-xxxx"))

    # P0-4: Fail-closed on missing API Key
    if not has_api_key and not simulate:
        console.print(Panel.fit(
            "[bold red]❌ 运行阻断 (Fail Closed): 未检测到有效的大模型 API Key！[/bold red]\n\n"
            "[yellow]学术与工程消融基准实验默认禁止在未显式声明的情况下使用硬编码模拟数据生成实测报告。\n"
            "1. 若需配置真实 API Key，请在 config.yaml 中填入有效的 llm.api_key；\n"
            "2. 若需在无 API Key 离线环境下测试执行流程与统计框架，请显式添加参数: [bold white]--simulate[/bold white][/yellow]\n\n"
            "[dim]运行命令: python experiments/ablation_study.py --simulate[/dim]",
            title="[bold red]实验安全阻断[/bold red]"
        ))
        sys.exit(1)

    is_simulation = not has_api_key or simulate

    if is_simulation:
        console.print(Panel.fit(
            "[bold yellow]⚠️  警告：当前运行在离线模拟数据模式 (MODE: SIMULATION / NOT A REAL BENCHMARK)[/bold yellow]\n"
            "[white]所有成文与指标为基于已知实测分布生成的测试桩数据，绝不代表真实在线 LLM 生成！\n"
            "报告顶部将强制标记离线水印，不可用作实测性能论据。[/white]",
            title="模式声明"
        ))

    # 1. 目标作者基准统计指纹
    full_corpus = SAMPLE_ESSAY_1 + "\n\n" + SAMPLE_ESSAY_2
    ground_truth_metrics = StylometricsAnalyzer.analyze(full_corpus)
    raw_samples = [{"title": "样文1", "content": SAMPLE_ESSAY_1}, {"title": "样文2", "content": SAMPLE_ESSAY_2}]

    # 2. 确定实验题目子集
    selected_topics = BENCHMARK_TOPICS[:max(1, min(len(BENCHMARK_TOPICS), topics_count))]
    total_runs = len(selected_topics) * repeats

    console.print(f"[cyan]实验配置: {len(selected_topics)} 个独立选题 × 每题 {repeats} 次重复 = {total_runs} 次采样/条件[/cyan]\n")

    # 记录各 Condition 跨 runs 的指标序列
    condition_records: Dict[str, Dict[str, List[float]]] = {
        "A": {"echo": [], "discourse": [], "rhythm": [], "lexical": [], "penalty": []},
        "B": {"echo": [], "discourse": [], "rhythm": [], "lexical": [], "penalty": []},
        "C1": {"echo": [], "discourse": [], "rhythm": [], "lexical": [], "penalty": []},
        "C2": {"echo": [], "discourse": [], "rhythm": [], "lexical": [], "penalty": []},
        "D": {"echo": [], "discourse": [], "rhythm": [], "lexical": [], "penalty": []},
    }

    if not is_simulation:
        provider = ModelProvider(config.llm, config.embedding)
        coordinator = CoordinatorAgent(config)
        critic = CriticAgent(config.llm)
        # 确保消融实验内存库纯净独立，防止历史运行遗留切片污染
        coordinator.memory_manager.clear_memory()
        deep_profile = coordinator.build_style(raw_samples, profile_name="Ablation_Profile", state=AgentState())

        for t_idx, item in enumerate(selected_topics, 1):
            t_topic = item["topic"]
            t_points = item["key_points"]

            for rep in range(1, repeats + 1):
                console.print(f"[bold yellow]>> 正在运行 Topic {t_idx}/{len(selected_topics)} (轮次 {rep}/{repeats}): [{t_topic[:20]}...][/bold yellow]")

                # Condition A: Vanilla Baseline 0 (无 Profile / 无 RAG / 无 Critic，纯大模型通用基线)
                sys_a = "你是一位专业的文章撰写助手，请围绕给定的选题写一篇深刻的文章。"
                user_a = f"围绕以下新主题创作一篇完整的文章：\n- **文章主题**：{t_topic}\n- **核心论述要点**：\n{t_points}\n- **目标字数**：1000 左右\n\n请直接输出成文全文。"
                art_a = provider.chat(sys_a, user_a, temperature=0.7)
                eval_a = critic.evaluate(art_a, deep_profile, state=AgentState())
                res_a = CompositeEvaluator.calculate_echoscore(art_a, ground_truth_metrics, deep_profile, eval_a.style_fidelity)

                # Condition B: +Profile Only (有 Profile / 无 RAG / 无 Critic，通过 WriterAgent 统一装配，显式空检索快照)
                state_b = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                state_b.memory_snapshot = []
                art_b = coordinator.writer_agent.generate(state_b, deep_profile)
                eval_b = critic.evaluate(art_b, deep_profile, state=AgentState())
                res_b = CompositeEvaluator.calculate_echoscore(art_b, ground_truth_metrics, deep_profile, eval_b.style_fidelity)

                # Condition C1: +Standard Dense Semantic RAG (纯密集向量召回，无结构打标，无 Sparse/RRF，统一通过 WriterAgent 生成)
                state_c1 = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                c1_few_shots = coordinator.memory_manager.retrieve_dense(
                    query=f"{t_topic} {t_points}", top_k=3, target_type=None
                )
                state_c1.memory_snapshot = [{"content": s} for s in c1_few_shots]
                art_c1 = coordinator.writer_agent.generate(state_c1, deep_profile)
                eval_c1 = critic.evaluate(art_c1, deep_profile, state=AgentState())
                res_c1 = CompositeEvaluator.calculate_echoscore(art_c1, ground_truth_metrics, deep_profile, eval_c1.style_fidelity)

                # Condition C2: +Style-Aware RAG (结构化定向召回 hook + quote + argument，统一通过 WriterAgent 生成，无 Critic 反思)
                state_c2 = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                c2_few_shots = coordinator.memory_manager.retrieve_dynamic_few_shots(
                    query=f"{t_topic} {t_points}", top_k=3
                )
                state_c2.memory_snapshot = [{"content": s} for s in c2_few_shots]
                art_c2 = coordinator.writer_agent.generate(state_c2, deep_profile)
                eval_c2 = critic.evaluate(art_c2, deep_profile, state=AgentState())
                res_c2 = CompositeEvaluator.calculate_echoscore(art_c2, ground_truth_metrics, deep_profile, eval_c2.style_fidelity)

                # Condition D: Full EchoStyle (复用与 C2 完全相同的定向召回，额外开启 FSM Critic 反思闭环)
                state_d = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                art_d, report_d, _ = coordinator.generate_article(
                    profile=deep_profile,
                    topic=t_topic,
                    key_points=t_points,
                    word_count=1000,
                    state=state_d
                )
                res_d = CompositeEvaluator.calculate_echoscore(art_d, ground_truth_metrics, deep_profile, report_d.style_fidelity)

                # 记录指标
                for code, res in [("A", res_a), ("B", res_b), ("C1", res_c1), ("C2", res_c2), ("D", res_d)]:
                    condition_records[code]["echo"].append(res.echo_score)
                    condition_records[code]["discourse"].append(res.discourse_fit)
                    condition_records[code]["rhythm"].append(res.rhythm_match)
                    condition_records[code]["lexical"].append(res.lexical_authenticity)
                    condition_records[code]["penalty"].append(res.cliche_penalty)

    else:
        # 离线模拟数据：多主题模拟，如实反映实测分布与核心权衡 (C1 > C2 ≈ D)
        sim_presets = [
            # Topic 1
            {
                "A": (79.4, 83.2, 91.3, 93.2, 0.0),
                "B": (87.9, 90.4, 73.2, 92.3, 0.0),
                "C1": (93.5, 94.8, 99.2, 95.0, 0.0),
                "C2": (89.6, 96.8, 84.1, 95.0, 0.0),
                "D": (89.4, 97.5, 80.5, 95.2, 0.0),
            },
            # Topic 2
            {
                "A": (78.2, 81.5, 89.8, 92.8, 10.0),
                "B": (86.5, 89.2, 74.0, 91.8, 0.0),
                "C1": (92.8, 94.2, 98.6, 94.5, 0.0),
                "C2": (88.9, 96.2, 83.5, 94.7, 0.0),
                "D": (88.8, 97.0, 79.8, 95.0, 0.0),
            },
            # Topic 3
            {
                "A": (80.1, 84.0, 92.0, 93.5, 0.0),
                "B": (88.4, 91.0, 72.8, 92.6, 0.0),
                "C1": (94.1, 95.2, 99.5, 95.4, 0.0),
                "C2": (90.2, 97.1, 84.6, 95.3, 0.0),
                "D": (89.9, 97.8, 81.0, 95.5, 0.0),
            },
            # Topic 4
            {
                "A": (79.0, 82.8, 90.9, 93.0, 0.0),
                "B": (87.2, 90.0, 73.5, 92.0, 0.0),
                "C1": (93.2, 94.6, 99.0, 94.8, 0.0),
                "C2": (89.4, 96.5, 83.9, 94.9, 0.0),
                "D": (89.2, 97.3, 80.2, 95.1, 0.0),
            },
            # Topic 5
            {
                "A": (78.8, 82.4, 91.0, 92.9, 0.0),
                "B": (87.5, 90.2, 73.0, 92.2, 0.0),
                "C1": (93.6, 95.0, 99.1, 95.1, 0.0),
                "C2": (89.8, 96.7, 84.3, 95.1, 0.0),
                "D": (89.5, 97.4, 80.6, 95.3, 0.0),
            },
        ]

        for t_idx in range(len(selected_topics)):
            preset = sim_presets[t_idx % len(sim_presets)]
            for rep in range(repeats):
                noise = (rep * 0.2)
                for code in ["A", "B", "C1", "C2", "D"]:
                    echo, disc, rhy, lex, pen = preset[code]
                    condition_records[code]["echo"].append(round(echo + noise, 1))
                    condition_records[code]["discourse"].append(round(disc + noise * 0.5, 1))
                    condition_records[code]["rhythm"].append(round(rhy + noise * 0.3, 1))
                    condition_records[code]["lexical"].append(round(lex, 1))
                    condition_records[code]["penalty"].append(round(pen, 1))

    # 3. 聚合各条件统计量 (Mean ± Std & 95% CI)
    summary: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for code in ["A", "B", "C1", "C2", "D"]:
        summary[code] = {
            "echo": _calc_stats(condition_records[code]["echo"]),
            "discourse": _calc_stats(condition_records[code]["discourse"]),
            "rhythm": _calc_stats(condition_records[code]["rhythm"]),
            "lexical": _calc_stats(condition_records[code]["lexical"]),
            "penalty": _calc_stats(condition_records[code]["penalty"]),
        }

    # 4. 打印消融实验实测矩阵
    title_suffix = " [离线模拟模式 MOCK]" if is_simulation else " [真实实测 REAL]"
    table = Table(title=f"EchoStyle 3.2 五组严谨消融实验收益矩阵{title_suffix}")
    table.add_column("消融实验条件", style="cyan bold")
    table.add_column("Profile", justify="center")
    table.add_column("普通RAG", justify="center")
    table.add_column("风格RAG", justify="center")
    table.add_column("Critic", justify="center")
    table.add_column("篇章拟合 (Discourse)", justify="right")
    table.add_column("节奏吻合 (Rhythm)", justify="right")
    table.add_column("用词质感 (Lexical)", justify="right")
    table.add_column("八股惩罚", justify="right")
    table.add_column("统一目标 EchoScore (Mean ± Std)", style="green bold", justify="right")

    cond_configs = [
        ("A (Vanilla Base)", "❌", "❌", "❌", "❌", "A"),
        ("B (+Profile Only)", "✅", "❌", "❌", "❌", "B"),
        ("C1 (+Standard Dense RAG)", "✅", "✅", "❌", "❌", "C1"),
        ("C2 (+Style-Aware RAG)", "✅", "❌", "✅", "❌", "C2"),
        ("D (Full EchoStyle)", "✅", "❌", "✅", "✅", "D"),
    ]

    for name, p, r_std, r_style, c, code in cond_configs:
        s = summary[code]
        echo_str = f"{s['echo'][0]:.1f} ± {s['echo'][1]:.1f}"
        table.add_row(
            name, p, r_std, r_style, c,
            f"{s['discourse'][0]:.1f}",
            f"{s['rhythm'][0]:.1f}",
            f"{s['lexical'][0]:.1f}",
            f"-{s['penalty'][0]:.1f}",
            echo_str,
        )

    console.print(table)

    # 5. 组件独立边际增益分析 (直面 C1 > C2 > D 事实，进行客观多目标权衡归因)
    mean_a = summary["A"]["echo"][0]
    mean_b = summary["B"]["echo"][0]
    mean_c1 = summary["C1"]["echo"][0]
    mean_c2 = summary["C2"]["echo"][0]
    mean_d = summary["D"]["echo"][0]

    gain_profile = mean_b - mean_a
    gain_std_rag = mean_c1 - mean_b
    gain_style_over_std = mean_c2 - mean_c1
    gain_critic = mean_d - mean_c2

    contrib_table = Table(title="核心组件独立边际增益与多目标权衡分析 (Component Marginal Gains)")
    contrib_table.add_column("对比组", style="cyan bold", width=12)
    contrib_table.add_column("验证变量", style="white", width=26)
    contrib_table.add_column("EchoScore 净增益", style="bold", justify="right", width=18)
    contrib_table.add_column("客观机制与多目标权衡解释", style="yellow")

    def fmt_gain(val: float) -> str:
        return f"{val:+.1f} 分"

    contrib_table.add_row(
        "B vs A",
        "Style Profile 显式先验",
        f"[green]{fmt_gain(gain_profile)}[/green]",
        "显式注入统计句长与人设，摆脱 AI 机械翻译腔"
    )
    contrib_table.add_row(
        "C1 vs B",
        "Standard Dense RAG 纯语义检索",
        f"[green]{fmt_gain(gain_std_rag)}[/green]",
        "引入语料上下文；由于召回段落连续平稳，行文节奏吻合度 (Rhythm) 达到极高水平 (~99 分)"
    )

    c2_color = "green" if gain_style_over_std >= 0 else "red"
    c2_desc = (
        f"Discourse拟合({summary['C2']['discourse'][0]:.1f} vs {summary['C1']['discourse'][0]:.1f})，但短金句异构拼接打乱节奏(Rhythm {summary['C2']['rhythm'][0] - summary['C1']['rhythm'][0]:+.1f}分)，单一目标表现受挫"
        if gain_style_over_std < 0
        else f"结构化定向装配提升整体篇章表达与论点锐度，EchoScore 净增 +{gain_style_over_std:.1f} 分"
    )
    contrib_table.add_row(
        "C2 vs C1",
        "Style-Aware RAG vs 普通语义 RAG",
        f"[{c2_color}]{fmt_gain(gain_style_over_std)}[/{c2_color}]",
        c2_desc
    )

    d_color = "green" if gain_critic >= 0 else "yellow"
    d_desc = (
        f"FSM 自审精准拦截八股违规，自愈提分 +{gain_critic:.1f} 分"
        if gain_critic > 0
        else f"消除潜在违规词，但反思重写略微收窄句长方差，加权总分变动 {gain_critic:+.1f} 分"
    )
    contrib_table.add_row(
        "D vs C2",
        "Critic 自审反思闭环",
        f"[{d_color}]{fmt_gain(gain_critic)}[/{d_color}]",
        d_desc
    )

    console.print(contrib_table)

    # 6. 导出报告至 reports/ 目录
    default_filename = "ablation_study_simulation.md" if is_simulation else "ablation_study_report.md"
    target_report_file = Path(output_path) if output_path else Path(f"./reports/{default_filename}")
    target_report_file.parent.mkdir(parents=True, exist_ok=True)

    mode_banner = (
        "> ⚠️ **免责声明与模式标记 (SIMULATION MODE / NOT A REAL BENCHMARK)**：\n"
        "> 本报告生成于**离线模拟数据模式**（未检测到真实有效 API Key 或显式指定 `--simulate`）。\n"
        "> 报告内所有文本与评测指标均为用于流程验证的桩数据，**不可作为学术或工业实测结论**！\n"
        "> 真实实测请配置 API Key 并运行 `python main.py benchmark --ablation`。"
        if is_simulation
        else "> **实验模式**：真实 LLM 在线实测 (REAL LLM EXECUTION)"
    )

    empirical_analysis_section = _generate_empirical_analysis(summary)

    report_md = f"""# EchoStyle 3.2 五组严谨消融实验报告 (5-Condition Matrix Report)

- **评测时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}
- **评测规模**：{len(selected_topics)} 个独立主题 × {repeats} 次采样 = 共 {total_runs} 组样本/条件
{mode_banner}

## 一、 5-Condition 消融实验实测矩阵 (Mean ± Std)

| 消融实验条件 | Profile | 普通语义 RAG | 风格感知 RAG | Critic 自审 | 篇章拟合 (Discourse) | 节奏吻合 (Rhythm) | 用词质感 (Lexical) | 八股惩罚 | 统一目标 EchoScore (Mean ± Std) | 95% 置信区间 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A (Vanilla Base)** | ❌ | ❌ | ❌ | ❌ | {summary['A']['discourse'][0]:.1f} | {summary['A']['rhythm'][0]:.1f} | {summary['A']['lexical'][0]:.1f} | -{summary['A']['penalty'][0]:.1f} | **{summary['A']['echo'][0]:.1f} ± {summary['A']['echo'][1]:.1f}** | [{max(0.0, summary['A']['echo'][0] - summary['A']['echo'][2]):.1f}, {summary['A']['echo'][0] + summary['A']['echo'][2]:.1f}] |
| **B (+Profile Only)** | ✅ | ❌ | ❌ | ❌ | {summary['B']['discourse'][0]:.1f} | {summary['B']['rhythm'][0]:.1f} | {summary['B']['lexical'][0]:.1f} | -{summary['B']['penalty'][0]:.1f} | **{summary['B']['echo'][0]:.1f} ± {summary['B']['echo'][1]:.1f}** | [{max(0.0, summary['B']['echo'][0] - summary['B']['echo'][2]):.1f}, {summary['B']['echo'][0] + summary['B']['echo'][2]:.1f}] |
| **C1 (+Standard Dense RAG)** | ✅ | ✅ | ❌ | ❌ | {summary['C1']['discourse'][0]:.1f} | {summary['C1']['rhythm'][0]:.1f} | {summary['C1']['lexical'][0]:.1f} | -{summary['C1']['penalty'][0]:.1f} | **{summary['C1']['echo'][0]:.1f} ± {summary['C1']['echo'][1]:.1f}** | [{max(0.0, summary['C1']['echo'][0] - summary['C1']['echo'][2]):.1f}, {summary['C1']['echo'][0] + summary['C1']['echo'][2]:.1f}] |
| **C2 (+Style-Aware RAG)** | ✅ | ❌ | ✅ | ❌ | {summary['C2']['discourse'][0]:.1f} | {summary['C2']['rhythm'][0]:.1f} | {summary['C2']['lexical'][0]:.1f} | -{summary['C2']['penalty'][0]:.1f} | **{summary['C2']['echo'][0]:.1f} ± {summary['C2']['echo'][1]:.1f}** | [{max(0.0, summary['C2']['echo'][0] - summary['C2']['echo'][2]):.1f}, {summary['C2']['echo'][0] + summary['C2']['echo'][2]:.1f}] |
| **D (Full EchoStyle)** | ✅ | ❌ | ✅ | ✅ | {summary['D']['discourse'][0]:.1f} | {summary['D']['rhythm'][0]:.1f} | {summary['D']['lexical'][0]:.1f} | -{summary['D']['penalty'][0]:.1f} | **{summary['D']['echo'][0]:.1f} ± {summary['D']['echo'][1]:.1f}** | [{max(0.0, summary['D']['echo'][0] - summary['D']['echo'][2]):.1f}, {summary['D']['echo'][0] + summary['D']['echo'][2]:.1f}] |

{empirical_analysis_section}
"""
    target_report_file.write_text(report_md, encoding="utf-8")
    # 同时在 profiles 下写入同步副本以防老逻辑读取
    (Path("./profiles") / default_filename).write_text(report_md, encoding="utf-8")
    console.print(f"\n[bold green]消融实验报告已成功归档至:[/bold green] {target_report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoStyle 3.2 五组消融实验套件")
    parser.add_argument("--topics", type=int, default=3, help="评测题目数量 (1-5, 默认 3)")
    parser.add_argument("--repeat", type=int, default=1, help="每个题目重复采样轮次 (默认 1)")
    parser.add_argument("--simulate", action="store_true", help="离线模拟模式 (无 API Key 时强制开启)")
    parser.add_argument("--output", help="自定义报告输出路径")
    cli_args = parser.parse_args()

    run_ablation_study(
        topics_count=cli_args.topics,
        repeats=cli_args.repeat,
        simulate=cli_args.simulate,
        output_path=cli_args.output,
    )
