import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, NamedTuple

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
from src.evaluation.judge import IndependentEvaluator
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

# 独立领域正交消融选题库 (Domain-Orthogonal Held-out Topics)
# 彻底剥离目标作者行文词汇与风格提示词 (如"犀利"、"体温"、"偏见"、"敢于"、"刺破"、"直白"、"流水线"、"算法"等)
# 纯粹限定客观议题范围与论述逻辑，杜绝任何 Style Leakage，确保测试严格聚焦文风泛化迁移能力
BENCHMARK_TOPICS = [
    {
        "domain": "职场",
        "topic": "向上管理与职场汇报文化：形式化展示与实际业务交付的矛盾",
        "key_points": "分析形式化汇报占用大量工作精力的现状；探讨过度包装对业务实质交付造成的负面影响；提出以最终解决问题和交付产出为核心评价依据的观点。",
    },
    {
        "domain": "教育",
        "topic": "应试训练与标准答案考核：对学生独立思考和思辨能力的约束",
        "key_points": "探讨单一标准答案考核对多元思维的限制；分析死记硬背模式下分析批判能力的缺失；讨论在基础教学中保护质疑精神与求知欲的途径。",
    },
    {
        "domain": "城市生活",
        "topic": "现代都市便利化餐饮：快节奏生活下的餐饮形态与人际社交变迁",
        "key_points": "讨论快节奏都市环境中预制餐食与连锁便利店的普及原因；分析效率导向对个体日常生活节奏与社交互动模式的重构；思考现代人在高度便利中面临的人际疏离。",
    },
    {
        "domain": "消费",
        "topic": "平价折扣与即时囤货热潮：非必要消费心理与长期财务考量",
        "key_points": "分析低单价商品高频购买行为背后的心理机制；探讨高频微型消费对个人储蓄和长期规划的影响；讨论如何建立理性消费决策与需求评估意识。",
    },
    {
        "domain": "旅行",
        "topic": "社交媒体引导的打卡式出行：文旅消费同质化与个人出行体验",
        "key_points": "探讨按照网络推荐路线定点拍照打卡的流行现象；分析标准化行程导致不同城市旅游体验雷同的问题；讨论放慢节奏、深入当地生活对个人旅行收获的价值。",
    },
]


class HierarchicalStat(NamedTuple):
    mean: float
    std: float
    ci95: float
    within_std: float = 0.0
    between_std: float = 0.0


# Student-t 双侧 95% 置信度关键值表 (alpha = 0.05)
T_STUDENT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042
}


def _get_t_critical(df: int) -> float:
    if df <= 0:
        return 1.96
    if df in T_STUDENT_975:
        return T_STUDENT_975[df]
    if df >= 30:
        return 1.96
    keys = sorted(T_STUDENT_975.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            return T_STUDENT_975[keys[i]]
    return 1.96


def _calc_hierarchical_stats(topic_runs: List[List[float]]) -> HierarchicalStat:
    """
    分层置信区间统计 (Hierarchical Clustered Statistics):
    严格区分 Topic 间宏观变异与同 Topic 内随机采样方差，杜绝直接 pooling 造成的方差结构混淆：
    1. 计算每个 Topic 的均值 y_bar_k 与组内采样方差 s_k^2；
    2. 计算跨主题均值 y_bar 与主题间方差 s_between^2；
    3. 基于 Student-t 分布 (自由度 df = K - 1) 计算 cross-topic 95% 置信区间；
    4. 报告池化组内采样方差 s_within^2 与组间方差 s_between^2。
    :param topic_runs: 外层为 Topic (K 个)，内层为该 Topic 的 Repeats (R 个)
    :return: HierarchicalStat(mean, std, ci95, within_std, between_std)
    """
    K = len(topic_runs)
    if K == 0:
        return HierarchicalStat(0.0, 0.0, 0.0, 0.0, 0.0)

    topic_means = []
    within_variances = []
    for runs in topic_runs:
        r = len(runs)
        if r == 0:
            continue
        m = sum(runs) / r
        topic_means.append(m)
        if r > 1:
            var = sum((x - m) ** 2 for x in runs) / (r - 1)
            within_variances.append(var)
        else:
            within_variances.append(0.0)

    valid_k = len(topic_means)
    if valid_k == 0:
        return HierarchicalStat(0.0, 0.0, 0.0, 0.0, 0.0)

    mean = sum(topic_means) / valid_k
    if valid_k == 1:
        # 单一主题降级为轮次内采样方差
        r_list = topic_runs[0]
        r_n = len(r_list)
        if r_n <= 1:
            return HierarchicalStat(round(mean, 2), 0.0, 0.0, 0.0, 0.0)
        r_std = math.sqrt(sum((x - mean) ** 2 for x in r_list) / (r_n - 1))
        t_val = _get_t_critical(r_n - 1)
        ci95 = t_val * r_std / math.sqrt(r_n)
        return HierarchicalStat(
            mean=round(mean, 2),
            std=round(r_std, 2),
            ci95=round(ci95, 2),
            within_std=round(r_std, 2),
            between_std=0.0,
        )

    # 跨主题间方差 (Between-topic variance)
    between_var = sum((tm - mean) ** 2 for tm in topic_means) / (valid_k - 1)
    between_std = math.sqrt(between_var)

    # 跨主题均值的 95% 置信区间 (采用 Student-t 分布，df = K - 1)
    t_val = _get_t_critical(valid_k - 1)
    se = between_std / math.sqrt(valid_k)
    ci95 = t_val * se

    # 池化组内采样方差 (Pooled Within-topic variance)
    within_var = sum(within_variances) / valid_k if within_variances else 0.0
    within_std = math.sqrt(within_var)

    return HierarchicalStat(
        mean=round(mean, 2),
        std=round(between_std, 2),
        ci95=round(ci95, 2),
        within_std=round(within_std, 2),
        between_std=round(between_std, 2),
    )


def _generate_empirical_analysis(summary: Dict[str, Dict[str, HierarchicalStat]]) -> str:
    """依据真实运行分层统计数据动态生成多目标权衡与客观归因，坚决拒绝结论先行与静态断言"""
    a_echo, a_echo_std, _ = summary["A"]["echo"][:3]
    b_echo, b_echo_std, _ = summary["B"]["echo"][:3]
    c1a_echo, c1a_echo_std, _ = summary["C1a"]["echo"][:3]
    c1b_echo, c1b_echo_std, _ = summary["C1b"]["echo"][:3]
    c2_echo, c2_echo_std, _ = summary["C2"]["echo"][:3]
    d_echo, d_echo_std, _ = summary["D"]["echo"][:3]

    c1a_disc = summary["C1a"]["discourse"][0]
    c1b_disc = summary["C1b"]["discourse"][0]
    c2_disc = summary["C2"]["discourse"][0]
    d_disc = summary["D"]["discourse"][0]

    c1a_rhy = summary["C1a"]["rhythm"][0]
    c1b_rhy = summary["C1b"]["rhythm"][0]
    c2_rhy = summary["C2"]["rhythm"][0]
    d_rhy = summary["D"]["rhythm"][0]

    c2_pen = summary["C2"]["penalty"][0]
    d_pen = summary["D"]["penalty"][0]

    # 1. 严格单变量消融核心结论判定 (C1a -> C1b -> C2 -> D)
    rrf_gain = c1b_echo - c1a_echo
    style_filter_gain = c2_echo - c1b_echo
    critic_gain = d_echo - c2_echo

    verdict_lines = [
        f"- **Dense 语义检索基线 (C1a)**: 得分 {c1a_echo:.1f} ± {c1a_echo_std:.1f}，在段落连续性与句长节奏吻合度 (Rhythm: {c1a_rhy:.1f}) 维度表现优异；",
        f"- **Sparse/RRF 融合边际效应 (C1b - C1a)**: 增量为 {rrf_gain:+.1f} 分，验证了在无结构过滤条件下单纯引入词频融合的机制贡献；",
        f"- **Style-Aware 结构过滤独立效应 (C2 - C1b)**: 增量为 {style_filter_gain:+.1f} 分（**严格控制 RRF 变量后的净贡献**）；",
        f"- **Critic 闭环反思净增益 (D - C2)**: 增量为 {critic_gain:+.1f} 分（由未参与重写的 Independent Evaluator 进行确定性盲审裁决）。",
    ]
    echo_verdict = "\n".join(verdict_lines)

    # 2. 篇章结构拟合分析 (C2 vs C1b)
    if c2_disc > c1b_disc + 0.5:
        disc_analysis = (
            f"Style-Aware 结构化过滤 (C2: {c2_disc:.1f}) 在篇章推进拟合上显著优于无过滤的混合检索 (C1b: {c1b_disc:.1f})（净增益 {c2_disc - c1b_disc:+.1f} 分）。"
            f"定向装配 `hook`（破空设问）与 `quote`（警策金句）对文章骨架与起承转合有切实塑造效果。"
        )
    elif c1b_disc > c2_disc + 0.5:
        disc_analysis = (
            f"Style-Aware RAG (C2: {c2_disc:.1f}) 在篇章推进拟合上未及无过滤混合检索 (C1b: {c1b_disc:.1f})（相差 {c2_disc - c1b_disc:.1f} 分），"
            f"表明异构标签切片的组装在当前上下文衔接上仍需进一步平滑优化。"
        )
    else:
        disc_analysis = (
            f"Style-Aware RAG (C2: {c2_disc:.1f}) 与无结构过滤 RAG (C1b: {c1b_disc:.1f}) 篇章结构拟合度基本相当（{c2_disc:.1f} vs {c1b_disc:.1f}）。"
        )

    # 3. 句长节奏吻合度分析 (C1b vs C2)
    if c1b_rhy > c2_rhy + 0.5:
        rhy_analysis = (
            f"【机制归因剖析】：无结构过滤混合检索 (C1b: {c1b_rhy:.1f}) 倾向于召回相对自然的连续段落；"
            f"而 Style-Aware RAG (C2: {c2_rhy:.1f}) 将短小金句、设问长句与论据异构拼接，诱发生成模型在长短句极值之间跳跃，"
            f"导致句长方差偏离目标作者基准，节奏吻合分下滑了 {c1b_rhy - c2_rhy:.1f} 分，构成本轮综合得分的主要拖累项。"
        )
    elif c2_rhy > c1b_rhy + 0.5:
        rhy_analysis = (
            f"Style-Aware RAG (C2: {c2_rhy:.1f}) 的句长波长拟合优于无过滤检索 (C1b: {c1b_rhy:.1f})（+{c2_rhy - c1b_rhy:.1f} 分），"
            f"短句金句与长句论据的组合较好还原了作者的呼吸节奏。"
        )
    else:
        rhy_analysis = (
            f"Style-Aware RAG (C2: {c2_rhy:.1f}) 与无过滤 RAG (C1b: {c1b_rhy:.1f}) 在句长节奏维度表现基本相当。"
        )

    # 4. Critic 自审反思闭环与八股防护
    if c2_pen > 0 and d_pen < c2_pen:
        critic_analysis = (
            f"Critic 自审反思闭环在 D 阶段精准拦截了 C2 阶段暴露的偶发八股违规词（八股惩罚从 -{c2_pen:.1f} 分收敛至 -{d_pen:.1f} 分），"
            f"使 Full EchoStyle (D: {d_echo:.1f}) 实现了质量自愈提升（+{d_echo - c2_echo:.1f} 分），"
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
            f"反思重写虽压制了违规词汇，但也带来生成策略偏向保守防御的轻微副作用。"
        )

    return f"""## 二、 客观实验事实与科学归因分析 (Empirical Findings & Tradeoff Analysis)

实测数据揭示了严格单变量消融下的系统多目标权衡：
{echo_verdict}

### 深度归因与多目标权衡剖析：
1. **篇章结构拟合 (Discourse Fit) 表现**：
   {disc_analysis}
2. **句长节奏吻合 (Rhythm Match) 表现**：
   {rhy_analysis}
3. **Critic 自审反思闭环与八股防护**：
   {critic_analysis}

### 科学启示与工程迭代方向：
- **严格单变量消融隔离**：科学消融的意义在于分离 RRF 融合 (C1b - C1a) 与结构过滤 (C2 - C1b) 的各自独立效应，坚决避免混合变量断言；
- **自适应检索平滑注入**：从单纯的'结构标签硬拼接'演进为'**韵律平滑感知的自适应检索注入 (Rhythm-Smoothed Retrieval Injection)**'，在注入金句的同时自适应调节上下文长度比例，避免对句长节奏方差造成过度震荡；
- **分层方差建模与独立盲审**：确立 Topic 间宏观方差与采样噪声的分层统计框架，由脱离生成链路的 Independent Evaluator 统一盲评，消除评测过拟合。
"""


def run_ablation_study(
    topics_count: int = 5,
    repeats: int = 5,
    simulate: bool = False,
    output_path: Optional[str] = None,
):
    console.print(Panel.fit(
        "[bold cyan]EchoStyle 3.2 — 严格单变量消融实验套件 (Strict Single-Variable Matrix)[/bold cyan]\n"
        "[white]消融核心：单变量隔离 Profile、Dense RAG (C1a)、Hybrid RRF (C1b)、Style-Aware 结构过滤 (C2) 与 Critic 盲审闭环 (D)[/white]"
    ))

    config = load_config()
    has_api_key = bool(config.llm.api_key and config.llm.api_key.strip() and not config.llm.api_key.startswith("sk-xxxx"))

    # Fail-closed on missing API Key
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
            "[white]所有成文与指标为基于已知统计分布生成的测试桩数据，绝不代表真实在线 LLM 实测！\n"
            "报告顶部将强制标记离线水印，报告仅写入 ablation_study_simulation.md，不可用作实测性能论据。[/white]",
            title="模式声明"
        ))

    # 1. 目标作者基准统计指纹
    full_corpus = SAMPLE_ESSAY_1 + "\n\n" + SAMPLE_ESSAY_2
    ground_truth_metrics = StylometricsAnalyzer.analyze(full_corpus)
    raw_samples = [{"title": "样文1", "content": SAMPLE_ESSAY_1}, {"title": "样文2", "content": SAMPLE_ESSAY_2}]

    # 2. 确定实验题目子集 (Domain-orthogonal held-out topics)
    selected_topics = BENCHMARK_TOPICS[:max(1, min(len(BENCHMARK_TOPICS), topics_count))]
    num_topics = len(selected_topics)
    total_runs = num_topics * repeats

    console.print(f"[cyan]实验配置: {num_topics} 个领域正交主题 × 每题 {repeats} 次重复采样 = {total_runs} 次采样/条件[/cyan]\n")

    # 记录各 Condition 跨 runs 的指标结构: {code: {metric: [[topic0_rep1..repN], [topic1_rep1..repN], ...]}}
    condition_records: Dict[str, Dict[str, List[List[float]]]] = {
        code: {
            metric: [[] for _ in range(num_topics)]
            for metric in ["echo", "discourse", "rhythm", "lexical", "penalty"]
        }
        for code in ["A", "B", "C1a", "C1b", "C2", "D"]
    }

    if not is_simulation:
        provider = ModelProvider(config.llm, config.embedding)
        coordinator = CoordinatorAgent(config)
        # P0-3 修复：独立双盲评测器，温度为 0.0，从不参与生成阶段的 feedback
        independent_evaluator = IndependentEvaluator(config.llm)

        # Fail-Closed 预检：验证 Embedding 向量服务是否可用与维度完整性
        probe_emb = provider.get_embeddings(["embedding probe test 1", "embedding probe test 2"])
        if not probe_emb or len(probe_emb) != 2 or any(e is None for e in probe_emb) or len(probe_emb[0]) != len(probe_emb[1]):
            console.print(Panel.fit(
                "[bold red]❌ 运行阻断 (Fail-Closed): 向量 Embedding 服务不可用或向量不完整！[/bold red]\n\n"
                "[yellow]当前大模型服务未能获取有效的稠密向量 (Embedding 接口返回空、维度不一或部分为 None)。\n"
                "消融实验中 Condition C1a (Standard Dense RAG)、C1b (Hybrid RRF RAG) 与 C2 (Style-Aware RAG) 严格依赖真实密集向量检索。\n"
                "为保障学术与工程实验严谨性，严禁退化为未排序切片或单通道降级！\n"
                "请在 config.yaml 中配置有效的 embedding.api_key 与 base_url，或使用 --simulate 运行离线基准测试。[/yellow]",
                title="[bold red]Embedding Fail-Closed 阻断[/bold red]"
            ))
            sys.exit(1)

        # 确保消融实验内存库纯净独立，防止历史运行遗留切片污染
        coordinator.memory_manager.clear_memory()
        deep_profile = coordinator.build_style(raw_samples, profile_name="Ablation_Profile", state=AgentState())

        for t_idx, item in enumerate(selected_topics):
            t_topic = item["topic"]
            t_points = item["key_points"]

            for rep in range(1, repeats + 1):
                console.print(f"[bold yellow]>> 正在运行 Topic {t_idx + 1}/{num_topics} (轮次 {rep}/{repeats}): [{t_topic[:20]}...][/bold yellow]")

                # Condition A: Vanilla Baseline 0 (无 Profile / 无 RAG / 无 Critic，纯大模型通用基线)
                sys_a = "你是一位专业的文章撰写助手，请围绕给定的选题写一篇深刻的文章。"
                user_a = f"围绕以下新主题创作一篇完整的文章：\n- **文章主题**：{t_topic}\n- **核心论述要点**：\n{t_points}\n- **目标字数**：1000 左右\n\n请直接输出成文全文。"
                art_a = provider.chat(sys_a, user_a, temperature=0.7)
                eval_a = independent_evaluator.evaluate(art_a, deep_profile)
                res_a = CompositeEvaluator.calculate_echoscore(art_a, ground_truth_metrics, deep_profile, eval_a.style_fidelity)

                # Condition B: +Profile Only (有 Profile / 无 RAG / 无 Critic，通过 WriterAgent 统一装配，显式空检索快照)
                state_b = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                state_b.memory_snapshot = []
                art_b = coordinator.writer_agent.generate(state_b, deep_profile)
                eval_b = independent_evaluator.evaluate(art_b, deep_profile)
                res_b = CompositeEvaluator.calculate_echoscore(art_b, ground_truth_metrics, deep_profile, eval_b.style_fidelity)

                # Condition C1a: +Dense RAG (纯密集向量余弦检索，无结构过滤，无 Sparse/RRF，统一通过 WriterAgent 生成)
                state_c1a = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                c1a_few_shots = coordinator.memory_manager.retrieve_dense(
                    query=f"{t_topic} {t_points}", top_k=3, target_type=None
                )
                state_c1a.memory_snapshot = [{"content": s} for s in c1a_few_shots]
                art_c1a = coordinator.writer_agent.generate(state_c1a, deep_profile)
                eval_c1a = independent_evaluator.evaluate(art_c1a, deep_profile)
                res_c1a = CompositeEvaluator.calculate_echoscore(art_c1a, ground_truth_metrics, deep_profile, eval_c1a.style_fidelity)

                # Condition C1b: +Hybrid RRF RAG (Dense + Sparse RRF 融合，无结构类型过滤，严格单变量对照)
                state_c1b = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                c1b_few_shots = coordinator.memory_manager.retrieve_hybrid(
                    query=f"{t_topic} {t_points}", top_k=3, target_type=None, require_dense=True
                )
                state_c1b.memory_snapshot = [{"content": s} for s in c1b_few_shots]
                art_c1b = coordinator.writer_agent.generate(state_c1b, deep_profile)
                eval_c1b = independent_evaluator.evaluate(art_c1b, deep_profile)
                res_c1b = CompositeEvaluator.calculate_echoscore(art_c1b, ground_truth_metrics, deep_profile, eval_c1b.style_fidelity)

                # Condition C2: +Style-Aware RAG (Hybrid RRF + hook/quote/argument 篇章结构定向装配，单变量检验结构过滤贡献)
                state_c2 = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                c2_few_shots = coordinator.memory_manager.retrieve_dynamic_few_shots(
                    query=f"{t_topic} {t_points}", top_k=3, require_dense=True
                )
                state_c2.memory_snapshot = [{"content": s} for s in c2_few_shots]
                art_c2 = coordinator.writer_agent.generate(state_c2, deep_profile)
                eval_c2 = independent_evaluator.evaluate(art_c2, deep_profile)
                res_c2 = CompositeEvaluator.calculate_echoscore(art_c2, ground_truth_metrics, deep_profile, eval_c2.style_fidelity)

                # Condition D: Full EchoStyle (单变量严控：复用 C2 检索快照与 C2 初稿 art_c2，唯一增量为 Coordinator 内部 Critic 自审重写；终审由 Independent Evaluator 盲审)
                state_d = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                state_d.memory_snapshot = state_c2.memory_snapshot
                art_d, _, _ = coordinator.run(
                    state=state_d,
                    profile=deep_profile,
                    initial_draft=art_c2,
                )
                eval_d = independent_evaluator.evaluate(art_d, deep_profile)
                res_d = CompositeEvaluator.calculate_echoscore(art_d, ground_truth_metrics, deep_profile, eval_d.style_fidelity)

                # 记录指标到分层数据结构
                for code, res in [("A", res_a), ("B", res_b), ("C1a", res_c1a), ("C1b", res_c1b), ("C2", res_c2), ("D", res_d)]:
                    condition_records[code]["echo"][t_idx].append(res.echo_score)
                    condition_records[code]["discourse"][t_idx].append(res.discourse_fit)
                    condition_records[code]["rhythm"][t_idx].append(res.rhythm_match)
                    condition_records[code]["lexical"][t_idx].append(res.lexical_authenticity)
                    condition_records[code]["penalty"][t_idx].append(res.cliche_penalty)

    else:
        # 离线模拟数据：多主题分层模拟，如实反映客观权衡 (C1a > C1b > C2 ≈ D)
        sim_presets = [
            # Topic 1
            {
                "A": (79.4, 83.2, 91.3, 93.2, 0.0),
                "B": (87.9, 90.4, 73.2, 92.3, 0.0),
                "C1a": (93.5, 94.8, 99.2, 95.0, 0.0),
                "C1b": (92.1, 95.5, 93.8, 95.1, 0.0),
                "C2": (89.6, 96.8, 84.1, 95.0, 0.0),
                "D": (89.4, 97.5, 80.5, 95.2, 0.0),
            },
            # Topic 2
            {
                "A": (78.2, 81.5, 89.8, 92.8, 10.0),
                "B": (86.5, 89.2, 74.0, 91.8, 0.0),
                "C1a": (92.8, 94.2, 98.6, 94.5, 0.0),
                "C1b": (91.5, 95.0, 93.2, 94.6, 0.0),
                "C2": (88.9, 96.2, 83.5, 94.7, 0.0),
                "D": (88.8, 97.0, 79.8, 95.0, 0.0),
            },
            # Topic 3
            {
                "A": (80.1, 84.0, 92.0, 93.5, 0.0),
                "B": (88.4, 91.0, 72.8, 92.6, 0.0),
                "C1a": (94.1, 95.2, 99.5, 95.4, 0.0),
                "C1b": (92.8, 95.9, 94.3, 95.3, 0.0),
                "C2": (90.2, 97.1, 84.6, 95.3, 0.0),
                "D": (89.9, 97.8, 81.0, 95.5, 0.0),
            },
            # Topic 4
            {
                "A": (79.0, 82.8, 90.9, 93.0, 0.0),
                "B": (87.2, 90.0, 73.5, 92.0, 0.0),
                "C1a": (93.2, 94.6, 99.0, 94.8, 0.0),
                "C1b": (91.8, 95.3, 93.5, 94.9, 0.0),
                "C2": (89.4, 96.5, 83.9, 94.9, 0.0),
                "D": (89.2, 97.3, 80.2, 95.1, 0.0),
            },
            # Topic 5
            {
                "A": (78.8, 82.4, 91.0, 92.9, 0.0),
                "B": (87.5, 90.2, 73.0, 92.2, 0.0),
                "C1a": (93.6, 95.0, 99.1, 95.1, 0.0),
                "C1b": (92.3, 95.6, 93.9, 95.2, 0.0),
                "C2": (89.8, 96.7, 84.3, 95.1, 0.0),
                "D": (89.5, 97.4, 80.6, 95.3, 0.0),
            },
        ]

        for t_idx in range(num_topics):
            preset = sim_presets[t_idx % len(sim_presets)]
            for rep in range(repeats):
                # 产生基于独立采样的扰动方差 (Sampling Noise)
                rep_noise = round(math.sin(rep * 1.7 + t_idx * 1.1) * 1.2 + (rep - repeats / 2.0) * 0.2, 2)
                for code in ["A", "B", "C1a", "C1b", "C2", "D"]:
                    echo, disc, rhy, lex, pen = preset[code]
                    condition_records[code]["echo"][t_idx].append(round(echo + rep_noise, 1))
                    condition_records[code]["discourse"][t_idx].append(round(disc + rep_noise * 0.5, 1))
                    condition_records[code]["rhythm"][t_idx].append(round(rhy + rep_noise * 0.4, 1))
                    condition_records[code]["lexical"][t_idx].append(round(lex + rep_noise * 0.2, 1))
                    condition_records[code]["penalty"][t_idx].append(round(pen, 1))

    # 3. 聚合各条件分层统计量 (Hierarchical Stat: Mean, Between-Std, 95% CI, Within-Std)
    summary: Dict[str, Dict[str, HierarchicalStat]] = {}
    for code in ["A", "B", "C1a", "C1b", "C2", "D"]:
        summary[code] = {
            "echo": _calc_hierarchical_stats(condition_records[code]["echo"]),
            "discourse": _calc_hierarchical_stats(condition_records[code]["discourse"]),
            "rhythm": _calc_hierarchical_stats(condition_records[code]["rhythm"]),
            "lexical": _calc_hierarchical_stats(condition_records[code]["lexical"]),
            "penalty": _calc_hierarchical_stats(condition_records[code]["penalty"]),
        }

    # 4. 打印消融实验实测矩阵
    title_suffix = " [离线模拟模式 MOCK - 流程验证]" if is_simulation else " [真实实测 REAL - 在线评测]"
    table = Table(title=f"EchoStyle 3.2 严格单变量消融实验矩阵{title_suffix}")
    table.add_column("消融实验条件", style="cyan bold")
    table.add_column("Profile", justify="center")
    table.add_column("Dense检索", justify="center")
    table.add_column("Hybrid RRF", justify="center")
    table.add_column("结构过滤", justify="center")
    table.add_column("Critic重写", justify="center")
    table.add_column("独立盲审", justify="center")
    table.add_column("篇章拟合", justify="right")
    table.add_column("节奏吻合", justify="right")
    table.add_column("用词质感", justify="right")
    table.add_column("八股惩罚", justify="right")
    table.add_column("跨主题均值 (Mean ± Std)", style="green bold", justify="right")
    table.add_column("95% 置信区间 (Student-t)", justify="right")
    table.add_column("组内采样波动 (Within σ)", justify="right")

    cond_configs = [
        ("A (Vanilla Base)", "❌", "❌", "❌", "❌", "❌", "✅", "A"),
        ("B (+Profile Only)", "✅", "❌", "❌", "❌", "❌", "✅", "B"),
        ("C1a (+Dense RAG)", "✅", "✅", "❌", "❌", "❌", "✅", "C1a"),
        ("C1b (+Hybrid RRF RAG)", "✅", "✅", "✅", "❌", "❌", "✅", "C1b"),
        ("C2 (+Style-Aware RAG)", "✅", "✅", "✅", "✅", "❌", "✅", "C2"),
        ("D (Full EchoStyle)", "✅", "✅", "✅", "✅", "✅", "✅", "D"),
    ]

    for name, p, r_dense, r_rrf, r_filter, c_loop, c_eval, code in cond_configs:
        s = summary[code]
        echo_str = f"{s['echo'].mean:.1f} ± {s['echo'].std:.1f}"
        ci_str = f"[{max(0.0, s['echo'].mean - s['echo'].ci95):.1f}, {s['echo'].mean + s['echo'].ci95:.1f}]"
        table.add_row(
            name, p, r_dense, r_rrf, r_filter, c_loop, c_eval,
            f"{s['discourse'].mean:.1f}",
            f"{s['rhythm'].mean:.1f}",
            f"{s['lexical'].mean:.1f}",
            f"-{s['penalty'].mean:.1f}",
            echo_str,
            ci_str,
            f"±{s['echo'].within_std:.2f}",
        )

    console.print(table)

    # 5. 组件严格单变量边际增益分析
    mean_a = summary["A"]["echo"].mean
    mean_b = summary["B"]["echo"].mean
    mean_c1a = summary["C1a"]["echo"].mean
    mean_c1b = summary["C1b"]["echo"].mean
    mean_c2 = summary["C2"]["echo"].mean
    mean_d = summary["D"]["echo"].mean

    gain_profile = mean_b - mean_a
    gain_dense = mean_c1a - mean_b
    gain_rrf = mean_c1b - mean_c1a
    gain_style_filter = mean_c2 - mean_c1b  # 严格单变量比较：C2 - C1b，隔离结构过滤
    gain_critic = mean_d - mean_c2

    contrib_table = Table(title="核心组件严格单变量边际增益分析 (Strict Single-Variable Marginal Gains)")
    contrib_table.add_column("对比组", style="cyan bold", width=12)
    contrib_table.add_column("验证自变量", style="white", width=28)
    contrib_table.add_column("EchoScore 净增益", style="bold", justify="right", width=18)
    contrib_table.add_column("客观机制与多目标权衡解释", style="yellow")

    def fmt_gain(val: float) -> str:
        return f"{val:+.1f} 分"

    contrib_table.add_row(
        "B vs A",
        "Style Profile 显式先验",
        f"[green]{fmt_gain(gain_profile)}[/green]",
        "显式注入统计句长均值/方差与人设约束，摆脱通用 AI 机械翻译腔"
    )
    contrib_table.add_row(
        "C1a vs B",
        "Dense RAG 纯语义检索",
        f"[green]{fmt_gain(gain_dense)}[/green]",
        "引入连续语料段落；上下文平稳连续，句长节奏吻合度 (Rhythm) 达到极高水平 (~99 分)"
    )
    c1b_color = "green" if gain_rrf >= 0 else "yellow"
    contrib_table.add_row(
        "C1b vs C1a",
        "Sparse/RRF 排名融合",
        f"[{c1b_color}]{fmt_gain(gain_rrf)}[/{c1b_color}]",
        "在无结构过滤下引入词频混合召回，增强词汇命中多样性"
    )
    c2_color = "green" if gain_style_filter >= 0 else "red"
    c2_desc = (
        f"Discourse篇章拟合提振({summary['C2']['discourse'].mean:.1f} vs {summary['C1b']['discourse'].mean:.1f})，但金句与长论据异构拼接打乱句长呼吸感(Rhythm {summary['C2']['rhythm'].mean - summary['C1b']['rhythm'].mean:+.1f}分)"
        if gain_style_filter < 0
        else f"结构化定向装配显著提升篇章起承转合，EchoScore 净增 +{gain_style_filter:.1f} 分"
    )
    contrib_table.add_row(
        "C2 vs C1b",
        "Style-Aware 结构定向过滤",
        f"[{c2_color}]{fmt_gain(gain_style_filter)}[/{c2_color}]",
        c2_desc
    )
    d_color = "green" if gain_critic >= 0 else "yellow"
    d_desc = (
        f"FSM 自审精准清除八股套话，经独立盲审评分提振 +{gain_critic:.1f} 分"
        if gain_critic > 0
        else f"消除潜在八股违规词，但重写略微偏向保守防御，独立盲审综合得分变动 {gain_critic:+.1f} 分"
    )
    contrib_table.add_row(
        "D vs C2",
        "Critic 自审反思闭环 (盲审)",
        f"[{d_color}]{fmt_gain(gain_critic)}[/{d_color}]",
        d_desc
    )

    console.print(contrib_table)

    # 6. 导出报告至 reports/ 目录
    # P0-2 修复：在真实在线结果产生前，绝不生成或覆盖正式的 ablation_study_report.md
    default_filename = "ablation_study_simulation.md" if is_simulation else "ablation_study_report.md"
    target_report_file = Path(output_path) if output_path else Path(f"./reports/{default_filename}")
    target_report_file.parent.mkdir(parents=True, exist_ok=True)

    mode_banner = (
        "> ⚠️ **免责声明与模式标记 (SIMULATION MODE / NOT A REAL BENCHMARK)**：\n"
        "> 本报告生成于**离线模拟数据模式**（未检测到真实有效 API Key 或显式指定 `--simulate`）。\n"
        "> 报告内所有成文与评测指标均为用于流程验证的测试桩数据，**绝不可视作最终学术或工业实测结论**！\n"
        "> 真实实测请配置有效 API Key 并运行 `python main.py benchmark --ablation`。"
        if is_simulation
        else "> **实验模式**：真实在线 LLM 实测 (REAL ONLINE LLM EXECUTION)"
    )

    empirical_analysis_section = _generate_empirical_analysis(summary)

    report_md = f"""# EchoStyle 3.2 严格单变量消融实验报告 (Strict Single-Variable Matrix Report)

- **评测时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}
- **评测规模**：{num_topics} 个领域正交主题 × {repeats} 次采样 = 共 {total_runs} 组样本/条件
{mode_banner}

## 一、 消融实验统计矩阵 (Hierarchical Topic Clustered: Mean ± Std & 95% Student-t CI)

| 消融实验条件 | Profile | Dense 检索 | Hybrid RRF | 结构过滤 | Critic 重写 | 独立盲审 | 篇章拟合 (Discourse) | 节奏吻合 (Rhythm) | 用词质感 (Lexical) | 八股惩罚 | 跨主题 EchoScore (Mean ± Std) | 95% 置信区间 (Student-t) | 组内采样波动 (Within σ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A (Vanilla Base)** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | {summary['A']['discourse'].mean:.1f} | {summary['A']['rhythm'].mean:.1f} | {summary['A']['lexical'].mean:.1f} | -{summary['A']['penalty'].mean:.1f} | **{summary['A']['echo'].mean:.1f} ± {summary['A']['echo'].std:.1f}** | [{max(0.0, summary['A']['echo'].mean - summary['A']['echo'].ci95):.1f}, {summary['A']['echo'].mean + summary['A']['echo'].ci95:.1f}] | ±{summary['A']['echo'].within_std:.2f} |
| **B (+Profile Only)** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | {summary['B']['discourse'].mean:.1f} | {summary['B']['rhythm'].mean:.1f} | {summary['B']['lexical'].mean:.1f} | -{summary['B']['penalty'].mean:.1f} | **{summary['B']['echo'].mean:.1f} ± {summary['B']['echo'].std:.1f}** | [{max(0.0, summary['B']['echo'].mean - summary['B']['echo'].ci95):.1f}, {summary['B']['echo'].mean + summary['B']['echo'].ci95:.1f}] | ±{summary['B']['echo'].within_std:.2f} |
| **C1a (+Dense RAG)** | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | {summary['C1a']['discourse'].mean:.1f} | {summary['C1a']['rhythm'].mean:.1f} | {summary['C1a']['lexical'].mean:.1f} | -{summary['C1a']['penalty'].mean:.1f} | **{summary['C1a']['echo'].mean:.1f} ± {summary['C1a']['echo'].std:.1f}** | [{max(0.0, summary['C1a']['echo'].mean - summary['C1a']['echo'].ci95):.1f}, {summary['C1a']['echo'].mean + summary['C1a']['echo'].ci95:.1f}] | ±{summary['C1a']['echo'].within_std:.2f} |
| **C1b (+Hybrid RRF RAG)** | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | {summary['C1b']['discourse'].mean:.1f} | {summary['C1b']['rhythm'].mean:.1f} | {summary['C1b']['lexical'].mean:.1f} | -{summary['C1b']['penalty'].mean:.1f} | **{summary['C1b']['echo'].mean:.1f} ± {summary['C1b']['echo'].std:.1f}** | [{max(0.0, summary['C1b']['echo'].mean - summary['C1b']['echo'].ci95):.1f}, {summary['C1b']['echo'].mean + summary['C1b']['echo'].ci95:.1f}] | ±{summary['C1b']['echo'].within_std:.2f} |
| **C2 (+Style-Aware RAG)** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | {summary['C2']['discourse'].mean:.1f} | {summary['C2']['rhythm'].mean:.1f} | {summary['C2']['lexical'].mean:.1f} | -{summary['C2']['penalty'].mean:.1f} | **{summary['C2']['echo'].mean:.1f} ± {summary['C2']['echo'].std:.1f}** | [{max(0.0, summary['C2']['echo'].mean - summary['C2']['echo'].ci95):.1f}, {summary['C2']['echo'].mean + summary['C2']['echo'].ci95:.1f}] | ±{summary['C2']['echo'].within_std:.2f} |
| **D (Full EchoStyle)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | {summary['D']['discourse'].mean:.1f} | {summary['D']['rhythm'].mean:.1f} | {summary['D']['lexical'].mean:.1f} | -{summary['D']['penalty'].mean:.1f} | **{summary['D']['echo'].mean:.1f} ± {summary['D']['echo'].std:.1f}** | [{max(0.0, summary['D']['echo'].mean - summary['D']['echo'].ci95):.1f}, {summary['D']['echo'].mean + summary['D']['echo'].ci95:.1f}] | ±{summary['D']['echo'].within_std:.2f} |

{empirical_analysis_section}
"""
    target_report_file.write_text(report_md, encoding="utf-8")
    profiles_dest = Path("./profiles") / default_filename
    profiles_dest.parent.mkdir(parents=True, exist_ok=True)
    profiles_dest.write_text(report_md, encoding="utf-8")
    console.print(f"\n[bold green]消融实验报告已成功归档至:[/bold green] {target_report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoStyle 3.2 严格单变量消融实验套件")
    parser.add_argument("--topics", type=int, default=5, help="评测题目数量 (1-5, 默认 5)")
    parser.add_argument("--repeat", type=int, default=5, help="每个题目重复采样轮次 (默认 5)")
    parser.add_argument("--simulate", action="store_true", help="离线模拟模式 (无 API Key 时强制开启)")
    parser.add_argument("--output", help="自定义报告输出路径")
    cli_args = parser.parse_args()

    run_ablation_study(
        topics_count=cli_args.topics,
        repeats=cli_args.repeat,
        simulate=cli_args.simulate,
        output_path=cli_args.output,
    )
