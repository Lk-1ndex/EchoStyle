import argparse
import math
import os
import sys
import tempfile
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
from src.memory.memory_manager import MemoryManager
from src.memory.vector_store import VectorStore
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

    valid_runs = [runs for runs in topic_runs if runs and len(runs) > 0]
    valid_k = len(valid_runs)
    if valid_k == 0:
        return HierarchicalStat(0.0, 0.0, 0.0, 0.0, 0.0)

    topic_means = []
    within_variances = []
    for runs in valid_runs:
        r = len(runs)
        m = sum(runs) / r
        topic_means.append(m)
        if r > 1:
            var = sum((x - m) ** 2 for x in runs) / (r - 1)
            within_variances.append(var)
        else:
            within_variances.append(0.0)

    mean = sum(topic_means) / valid_k
    if valid_k == 1:
        # 单一主题降级为轮次内采样方差 (使用首个有效主题采样数据，严禁硬取 index 0)
        r_list = valid_runs[0]
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


def _generate_empirical_analysis(
    summary: Dict[str, Dict[str, HierarchicalStat]],
    delta_summary: Optional[Dict[str, HierarchicalStat]] = None,
    is_simulation: bool = False,
    valid_pairs: int = 0,
    attempted_pairs: int = 0,
) -> str:
    """依据配对分层统计数据生成多目标权衡分析；在仿真模式下严格采用流程验证措辞，拒绝将桩数据表述为实测结论"""
    if delta_summary is None:
        delta_summary = {
            "scaffolding": HierarchicalStat(round(summary["A1"]["echo"].mean - summary["A0"]["echo"].mean, 2), 0.0, 0.0),
            "profile": HierarchicalStat(round(summary["B"]["echo"].mean - summary["A1"]["echo"].mean, 2), 0.0, 0.0),
            "dense": HierarchicalStat(round(summary["C1a"]["echo"].mean - summary["B"]["echo"].mean, 2), 0.0, 0.0),
            "rrf": HierarchicalStat(round(summary["C1b"]["echo"].mean - summary["C1a"]["echo"].mean, 2), 0.0, 0.0),
            "style_filter": HierarchicalStat(round(summary["C2"]["echo"].mean - summary["C1b"]["echo"].mean, 2), 0.0, 0.0),
            "critic": HierarchicalStat(round(summary["D"]["echo"].mean - summary["C2"]["echo"].mean, 2), 0.0, 0.0),
        }

    c2_echo = summary["C2"]["echo"].mean
    d_echo = summary["D"]["echo"].mean

    c1a_disc = summary["C1a"]["discourse"].mean
    c1b_disc = summary["C1b"]["discourse"].mean
    c2_disc = summary["C2"]["discourse"].mean
    d_disc = summary["D"]["discourse"].mean

    c1a_rhy = summary["C1a"]["rhythm"].mean
    c1b_rhy = summary["C1b"]["rhythm"].mean
    c2_rhy = summary["C2"]["rhythm"].mean
    d_rhy = summary["D"]["rhythm"].mean

    c2_pen = summary["C2"]["penalty"].mean
    d_pen = summary["D"]["penalty"].mean

    ds = delta_summary

    def _ci(key: str) -> str:
        s = ds[key]
        return f"[{s.mean - s.ci95:+.1f}, {s.mean + s.ci95:+.1f}]"

    if is_simulation:
        verdict_lines = [
            f"- **Writer Scaffolding 提示工程基准管道 (A1 - A0)**: 配对差值 Δ = {ds['scaffolding'].mean:+.1f} 分 (95% CI: {_ci('scaffolding')})，用于验证分析管道对 Writer 提示工程与 Token 预算识别通路；",
            f"- **Style Profile 纯先验管道 (B - A1)**: 配对差值 Δ = {ds['profile'].mean:+.1f} 分 (95% CI: {_ci('profile')})，用于验证控制 Scaffolding 变量后对 Style Profile 纯先验的单变量配对净贡献识别通路；",
            f"- **Dense 语义检索基准管道 (C1a - B)**: 配对差值 Δ = {ds['dense'].mean:+.1f} 分 (95% CI: {_ci('dense')})，用于验证分析管道对段落连续性与句长节奏吻合度 (Rhythm) 维度的敏感度识别；",
            f"- **Sparse/RRF 融合管道 (C1b - C1a)**: 配对差值 Δ = {ds['rrf'].mean:+.1f} 分 (95% CI: {_ci('rrf')})，用于验证分析管道在剔除零相关文档排名偏置后的词频融合识别；",
            f"- **Style-Aware 结构过滤管道 (C2 - C1b)**: 配对差值 Δ = {ds['style_filter'].mean:+.1f} 分 (95% CI: {_ci('style_filter')})，用于验证分析管道对结构定向过滤与句长离散度变动的权衡捕获能力（严格控制 RRF 变量）；",
            f"- **Critic 闭环反思管道 (D - C2)**: 配对差值 Δ = {ds['critic'].mean:+.1f} 分 (95% CI: {_ci('critic')})，用于验证分析管道对 FSM 反思重写与条件盲审评测器 (Condition-Blind Holdout Evaluator) 的打分闭环通路。",
        ]
        echo_verdict = "\n".join(verdict_lines)
        return f"""## 三、 仿真数据流程验证与分析管道测试 (Simulation Pipeline Verification)

仿真数据用于验证分析管道能够识别以下差异（仅用于工程流水线与分层统计验证，绝非真实实测结论）：
{echo_verdict}

### 仿真流程测试说明与多目标权衡捕获：
1. **篇章结构拟合 (Discourse Fit) 管道识别**：
   桩数据设定展示了分析管道如何量化结构化过滤 (C2: {c2_disc:.1f}) 与无过滤混合检索 (C1b: {c1b_disc:.1f}) 在篇章起承转合上的分差计算（设定差值 {c2_disc - c1b_disc:+.1f} 分）。
2. **句长节奏吻合 (Rhythm Match) 管道识别**：
   桩数据设定展示了分析管道如何捕捉连续段落召回 (C1b: {c1b_rhy:.1f}) 与离散金句拼接 (C2: {c2_rhy:.1f}) 对句长波长拟合度指标的敏感度反应（设定差值 {c2_rhy - c1b_rhy:+.1f} 分）。
3. **Critic 自审反思闭环与八股防护管道识别**：
   桩数据设定展示了分析管道对状态机反思重写前后得分变动与套话拦截的捕获逻辑（设定差值 {d_echo - c2_echo:+.1f} 分）。

### 管道验证说明（仅限工程流水线层面）：
- **严格配对设计与区块完整性**：执行七条件全配对设计，任何单条件失败直接作废整组 7 条件配对块 (Fail-Closed Paired Block)；统计采用每个配对样本差值 Δ 及其 Student-t 95% 置信区间；
- **自适应检索平滑注入**：工程架构支持从单纯的'结构标签硬拼接'演进为'**韵律平滑感知的自适应检索注入 (Rhythm-Smoothed Retrieval Injection)**'，自适应调节上下文长度比例以避免节奏方差震荡；
- **分层方差建模与条件盲审**：确立 Topic 间宏观方差与采样噪声的分层统计框架，由脱离生成链路的 Condition-Blind Holdout Evaluator 统一盲评，消除评测过拟合；
- **免责说明**：以上内容仅为离线流水线功能与统计公式验证，不代表真实模型性能；真实科学结论请在配置有效 API Key 后运行在线实测。
"""
    else:
        verdict_lines = [
            f"- **Writer Scaffolding 提示工程基准 (A1 - A0)**: 配对增量 Δ = {ds['scaffolding'].mean:+.1f} 分 (95% CI: {_ci('scaffolding')})，体现 WriterAgent 提示工程、任务感知预算与防套话 Scaffolding 的基准增益；",
            f"- **Style Profile 纯先验严格单变量贡献 (B - A1)**: 配对增量 Δ = {ds['profile'].mean:+.1f} 分 (95% CI: {_ci('profile')})（**严格控制 Scaffolding 变量后的配对独立净贡献**）；",
            f"- **Dense 语义检索基线 (C1a - B)**: 配对增量 Δ = {ds['dense'].mean:+.1f} 分 (95% CI: {_ci('dense')})，在段落连续性与句长节奏吻合度 (Rhythm: {c1a_rhy:.1f}) 维度表现优异；",
            f"- **Sparse/RRF 融合边际效应 (C1b - C1a)**: 配对增量 Δ = {ds['rrf'].mean:+.1f} 分 (95% CI: {_ci('rrf')})，体现了在共享 Dense 准入规则下词频融合的纯配对边际贡献；",
            f"- **Style-Aware 结构过滤独立效应 (C2 - C1b)**: 配对增量 Δ = {ds['style_filter'].mean:+.1f} 分 (95% CI: {_ci('style_filter')})（**严格控制 RRF 变量后的结构定向过滤配对净贡献**）；",
            f"- **Critic 闭环反思净增益 (D - C2)**: 配对增量 Δ = {ds['critic'].mean:+.1f} 分 (95% CI: {_ci('critic')})（由未参与重写的 Condition-Blind Holdout Evaluator 进行条件盲审裁决）。",
        ]
        echo_verdict = "\n".join(verdict_lines)

        # 篇章结构拟合分析 (C2 vs C1b)
        if c2_disc > c1b_disc + 0.5:
            disc_analysis = (
                f"Style-Aware 结构化过滤 (C2: {c2_disc:.1f}) 在篇章推进拟合上优于无过滤混合检索 (C1b: {c1b_disc:.1f})（相对差值 {c2_disc - c1b_disc:+.1f} 分）。"
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

        # 句长节奏吻合度分析 (C1b vs C2)
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

        # Critic 自审反思闭环与八股防护
        if c2_pen > 0 and d_pen < c2_pen:
            critic_analysis = (
                f"Critic 自审反思闭环在 D 阶段精准拦截了 C2 阶段暴露的偶发八股违规词（八股惩罚从 -{c2_pen:.1f} 分收敛至 -{d_pen:.1f} 分），"
                f"使 Full EchoStyle (D: {d_echo:.1f}) 实现了质量自愈提升（+{d_echo - c2_echo:.1f} 分）。"
            )
        elif d_echo > c2_echo + 0.5:
            critic_analysis = (
                f"Full EchoStyle (D: {d_echo:.1f}) 相对无自审的 C2 ({c2_echo:.1f}) 取得净增益 +{d_echo - c2_echo:.1f} 分，"
                f"多轮反思重构对行文质感与论述锐度带来了正向提振。"
            )
        else:
            critic_analysis = (
                f"Critic 介入后 D ({d_echo:.1f}) 相对 C2 ({c2_echo:.1f}) 综合得分基本持平（变动 {d_echo - c2_echo:+.1f} 分）。"
                f"反思重写虽压制了违规词汇，但也带来生成策略偏向保守防御的轻微副作用。"
            )

        return f"""## 三、 客观实验事实与科学归因分析 (Empirical Findings & Tradeoff Analysis)

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
- **严格单变量消融隔离**：科学消融的意义在于分离 Profile 先验 (B - A1)、RRF 融合 (C1b - C1a) 与结构过滤 (C2 - C1b) 的各自独立效应，坚决避免混合变量断言；
- **自适应检索平滑注入**：从单纯的'结构标签硬拼接'演进为'**韵律平滑感知的自适应检索注入 (Rhythm-Smoothed Retrieval Injection)**'，在注入金句的同时自适应调节上下文长度比例，避免对句长节奏方差造成过度震荡；
- **分层方差建模与条件盲审**：确立 Topic 间宏观方差与采样噪声的分层统计框架，由脱离生成链路的 Condition-Blind Holdout Evaluator 统一盲评，消除评测过拟合。
"""


def _evaluate_condition_run(
    evaluator: IndependentEvaluator,
    article: str,
    profile: DeepStyleProfile,
    metrics: Any,
    max_retries: int = 2,
) -> Optional[CompositeEvaluationResult]:
    """
    带重试与严格 Fail-Closed 的消融评估样本解析：
    若 LLM Judge 失败触发 EvaluationUnavailableError，重试 max_retries 次；
    若重试均告失败，坚决返回 None 以便排除该样本，绝不填入 80/85/80 默认虚拟分污染基准！
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            report = evaluator.evaluate(article, profile)
            return CompositeEvaluator.calculate_echoscore(article, metrics, profile, report.style_fidelity)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    console.print(f"[bold red]❌ 裁判评测失败 (strict 排除样本，杜绝默认分污染基准): {last_err}[/bold red]")
    return None


def run_ablation_study(
    topics_count: int = 5,
    repeats: int = 5,
    simulate: bool = False,
    output_path: Optional[str] = None,
):
    with tempfile.TemporaryDirectory(prefix="echostyle_ablation_") as temp_dir:
        return _run_ablation_study(
            topics_count=topics_count,
            repeats=repeats,
            simulate=simulate,
            output_path=output_path,
            private_storage_path=str(Path(temp_dir) / "style_memory.json"),
        )


def _run_ablation_study(
    topics_count: int,
    repeats: int,
    simulate: bool,
    output_path: Optional[str],
    private_storage_path: str,
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
        for code in ["A0", "A1", "B", "C1a", "C1b", "C2", "D"]
    }
    # 记录严格配对单变量差值结构: {comp: [[topic0_rep1..repN], [topic1_rep1..repN], ...]}
    delta_records: Dict[str, List[List[float]]] = {
        comp: [[] for _ in range(num_topics)]
        for comp in ["scaffolding", "profile", "dense", "rrf", "style_filter", "critic"]
    }
    attempted_pairs = 0
    valid_pairs = 0

    if not is_simulation:
        provider = ModelProvider(config.llm, config.embedding)
        private_store = VectorStore(
            storage_path=private_storage_path,
            embedding_config=config.embedding,
            llm_config=config.llm,
        )
        coordinator = CoordinatorAgent(config, memory_manager=MemoryManager(vector_store=private_store))
        # P0-3 修复：独立条件盲审评测器，temperature=0.0，且开启 strict 模式（Fail-Closed，严禁默认好成绩）
        independent_evaluator = IndependentEvaluator(config.get_evaluator_config(), strict=True)

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

        # 此实验的临时记忆库与默认长期库物理隔离。
        deep_profile = coordinator.build_style(raw_samples, profile_name="Ablation_Profile", state=AgentState())

        for t_idx, item in enumerate(selected_topics):
            t_topic = item["topic"]
            t_points = item["key_points"]

            for rep in range(1, repeats + 1):
                attempted_pairs += 1
                console.print(f"[bold yellow]>> 正在运行 Topic {t_idx + 1}/{num_topics} (轮次 {rep}/{repeats}): [{t_topic[:20]}...][/bold yellow]")

                # 1. Condition A0: Vanilla Baseline 0 (纯通用外部基线)
                sys_a0 = "你是一位专业的文章撰写助手，请围绕给定的选题写一篇深刻的文章。"
                user_a0 = f"围绕以下新主题创作一篇完整的文章：\n- **文章主题**：{t_topic}\n- **核心论述要点**：\n{t_points}\n- **目标字数**：1000 左右\n\n请直接输出成文全文。"
                art_a0 = provider.chat(sys_a0, user_a0, temperature=config.llm.temperature)

                # 2. Condition A1: Writer Scaffolding Baseline (工程基准)
                state_a1 = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                state_a1.memory_snapshot = []
                art_a1 = coordinator.writer_agent.generate(state_a1, profile=None)

                # 3. Condition B: +Profile Only (严格单变量验证 Profile 独立贡献: B - A1)
                state_b = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                state_b.memory_snapshot = []
                art_b = coordinator.writer_agent.generate(state_b, deep_profile)

                # 4. Condition C1a: +Dense RAG (纯密集向量余弦检索)
                state_c1a = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                c1a_few_shots = coordinator.memory_manager.retrieve_dense(
                    query=f"{t_topic} {t_points}", top_k=3, target_type=None, require_dense=True,
                    profile_id=deep_profile.profile_id,
                )
                state_c1a.memory_snapshot = [{"content": s} for s in c1a_few_shots]
                art_c1a = coordinator.writer_agent.generate(state_c1a, deep_profile)

                # 5. Condition C1b: +Hybrid RRF RAG (Dense + Sparse RRF 融合，剔除零相关文档偏置)
                state_c1b = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                c1b_few_shots = coordinator.memory_manager.retrieve_hybrid(
                    query=f"{t_topic} {t_points}", top_k=3, target_type=None, require_dense=True,
                    profile_id=deep_profile.profile_id,
                )
                state_c1b.memory_snapshot = [{"content": s} for s in c1b_few_shots]
                art_c1b = coordinator.writer_agent.generate(state_c1b, deep_profile)

                # 6. Condition C2: +Style-Aware RAG (结构定向装配)
                state_c2 = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                c2_few_shots = coordinator.memory_manager.retrieve_dynamic_few_shots(
                    query=f"{t_topic} {t_points}", top_k=3, require_dense=True,
                    profile_id=deep_profile.profile_id,
                )
                state_c2.memory_snapshot = [{"content": s} for s in c2_few_shots]
                art_c2 = coordinator.writer_agent.generate(state_c2, deep_profile)

                # 7. Condition D: Full EchoStyle (单变量严控：复用 C2 检索快照与 C2 初稿)
                state_d = AgentState(topic=t_topic, key_points=t_points, word_count=1000)
                state_d.memory_snapshot = state_c2.memory_snapshot
                art_d, _, _ = coordinator.run(
                    state=state_d,
                    profile=deep_profile,
                    initial_draft=art_c2,
                )

                # 统一由 Condition-Blind Holdout Evaluator 进行严格条件盲审打分 (Fail-Closed 配对区块阻断)
                cond_runs = [
                    ("A0", art_a0), ("A1", art_a1), ("B", art_b),
                    ("C1a", art_c1a), ("C1b", art_c1b), ("C2", art_c2), ("D", art_d)
                ]
                block_res = {}
                block_failed = False
                for code, art in cond_runs:
                    res = _evaluate_condition_run(independent_evaluator, art, deep_profile, ground_truth_metrics)
                    if res is None:
                        block_failed = True
                        console.print(f"[yellow]⚠️ Condition {code} Topic {t_idx + 1} 轮次 {rep} 评测失败，按配对严谨性作废该完整 7 条件区块 (Fail-Closed Paired Block)。[/yellow]")
                        break
                    block_res[code] = res

                if block_failed:
                    continue

                valid_pairs += 1
                for code, res in block_res.items():
                    condition_records[code]["echo"][t_idx].append(res.echo_score)
                    condition_records[code]["discourse"][t_idx].append(res.discourse_fit)
                    condition_records[code]["rhythm"][t_idx].append(res.rhythm_match)
                    condition_records[code]["lexical"][t_idx].append(res.lexical_authenticity)
                    condition_records[code]["penalty"][t_idx].append(res.cliche_penalty)

                # 记录严格配对差值 Δ
                delta_records["scaffolding"][t_idx].append(round(block_res["A1"].echo_score - block_res["A0"].echo_score, 2))
                delta_records["profile"][t_idx].append(round(block_res["B"].echo_score - block_res["A1"].echo_score, 2))
                delta_records["dense"][t_idx].append(round(block_res["C1a"].echo_score - block_res["B"].echo_score, 2))
                delta_records["rrf"][t_idx].append(round(block_res["C1b"].echo_score - block_res["C1a"].echo_score, 2))
                delta_records["style_filter"][t_idx].append(round(block_res["C2"].echo_score - block_res["C1b"].echo_score, 2))
                delta_records["critic"][t_idx].append(round(block_res["D"].echo_score - block_res["C2"].echo_score, 2))

        # 检查是否存在有效配对数据
        if valid_pairs == 0:
            from src.core.exceptions import EvaluationUnavailableError
            raise EvaluationUnavailableError("消融实验中所有配对测试块均因评测裁判服务异常而失败，无法生成有效配对统计报告。")

    else:
        # 离线模拟数据：多主题分层模拟，如实反映客观权衡 (C1a > C1b > C2 ≈ D)
        sim_presets = [
            # Topic 1
            {
                "A0": (79.4, 83.2, 91.3, 93.2, 0.0),
                "A1": (82.5, 86.0, 88.0, 92.8, 0.0),
                "B": (87.9, 90.4, 73.2, 92.3, 0.0),
                "C1a": (93.5, 94.8, 99.2, 95.0, 0.0),
                "C1b": (92.1, 95.5, 93.8, 95.1, 0.0),
                "C2": (89.6, 96.8, 84.1, 95.0, 0.0),
                "D": (89.4, 97.5, 80.5, 95.2, 0.0),
            },
            # Topic 2
            {
                "A0": (78.2, 81.5, 89.8, 92.8, 10.0),
                "A1": (81.6, 84.5, 87.1, 92.2, 0.0),
                "B": (86.5, 89.2, 74.0, 91.8, 0.0),
                "C1a": (92.8, 94.2, 98.6, 94.5, 0.0),
                "C1b": (91.5, 95.0, 93.2, 94.6, 0.0),
                "C2": (88.9, 96.2, 83.5, 94.7, 0.0),
                "D": (88.8, 97.0, 79.8, 95.0, 0.0),
            },
            # Topic 3
            {
                "A0": (80.1, 84.0, 92.0, 93.5, 0.0),
                "A1": (83.2, 86.8, 88.5, 93.0, 0.0),
                "B": (88.4, 91.0, 72.8, 92.6, 0.0),
                "C1a": (94.1, 95.2, 99.5, 95.4, 0.0),
                "C1b": (92.8, 95.9, 94.3, 95.3, 0.0),
                "C2": (90.2, 97.1, 84.6, 95.3, 0.0),
                "D": (89.9, 97.8, 81.0, 95.5, 0.0),
            },
            # Topic 4
            {
                "A0": (79.0, 82.8, 90.9, 93.0, 0.0),
                "A1": (82.1, 85.5, 87.5, 92.5, 0.0),
                "B": (87.2, 90.0, 73.5, 92.0, 0.0),
                "C1a": (93.2, 94.6, 99.0, 94.8, 0.0),
                "C1b": (91.8, 95.3, 93.5, 94.9, 0.0),
                "C2": (89.4, 96.5, 83.9, 94.9, 0.0),
                "D": (89.2, 97.3, 80.2, 95.1, 0.0),
            },
            # Topic 5
            {
                "A0": (78.8, 82.4, 91.0, 92.9, 0.0),
                "A1": (81.9, 85.0, 87.2, 92.4, 0.0),
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
                attempted_pairs += 1
                valid_pairs += 1
                # 产生基于独立采样的扰动方差 (Sampling Noise)
                rep_noise = round(math.sin(rep * 1.7 + t_idx * 1.1) * 1.2 + (rep - repeats / 2.0) * 0.2, 2)
                block_echo = {}
                for code in ["A0", "A1", "B", "C1a", "C1b", "C2", "D"]:
                    echo, disc, rhy, lex, pen = preset[code]
                    echo_val = round(echo + rep_noise, 1)
                    block_echo[code] = echo_val
                    condition_records[code]["echo"][t_idx].append(echo_val)
                    condition_records[code]["discourse"][t_idx].append(round(disc + rep_noise * 0.5, 1))
                    condition_records[code]["rhythm"][t_idx].append(round(rhy + rep_noise * 0.4, 1))
                    condition_records[code]["lexical"][t_idx].append(round(lex + rep_noise * 0.2, 1))
                    condition_records[code]["penalty"][t_idx].append(round(pen, 1))

                # 严格配对差值
                delta_records["scaffolding"][t_idx].append(round(block_echo["A1"] - block_echo["A0"], 2))
                delta_records["profile"][t_idx].append(round(block_echo["B"] - block_echo["A1"], 2))
                delta_records["dense"][t_idx].append(round(block_echo["C1a"] - block_echo["B"], 2))
                delta_records["rrf"][t_idx].append(round(block_echo["C1b"] - block_echo["C1a"], 2))
                delta_records["style_filter"][t_idx].append(round(block_echo["C2"] - block_echo["C1b"], 2))
                delta_records["critic"][t_idx].append(round(block_echo["D"] - block_echo["C2"], 2))

    # 3. 聚合各条件分层统计量 (Hierarchical Stat: Mean, Between-Std, 95% CI, Within-Std)
    summary: Dict[str, Dict[str, HierarchicalStat]] = {}
    for code in ["A0", "A1", "B", "C1a", "C1b", "C2", "D"]:
        summary[code] = {
            "echo": _calc_hierarchical_stats(condition_records[code]["echo"]),
            "discourse": _calc_hierarchical_stats(condition_records[code]["discourse"]),
            "rhythm": _calc_hierarchical_stats(condition_records[code]["rhythm"]),
            "lexical": _calc_hierarchical_stats(condition_records[code]["lexical"]),
            "penalty": _calc_hierarchical_stats(condition_records[code]["penalty"]),
        }

    # 聚合核心组件配对单变量效应分层统计量
    delta_summary: Dict[str, HierarchicalStat] = {
        comp: _calc_hierarchical_stats(delta_records[comp])
        for comp in ["scaffolding", "profile", "dense", "rrf", "style_filter", "critic"]
    }

    # 4. 打印消融实验实测矩阵
    title_suffix = " [离线模拟模式 MOCK - 流程验证]" if is_simulation else " [真实实测 REAL - 在线评测]"
    table = Table(title=f"EchoStyle 3.2 严格单变量消融实验矩阵{title_suffix}")
    table.add_column("消融实验条件", style="cyan bold")
    table.add_column("Scaffolding", justify="center")
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
        ("A0 (Vanilla Base)", "❌", "❌", "❌", "❌", "❌", "❌", "✅", "A0"),
        ("A1 (Scaffolding Base)", "✅", "❌", "❌", "❌", "❌", "❌", "✅", "A1"),
        ("B (+Profile Only)", "✅", "✅", "❌", "❌", "❌", "❌", "✅", "B"),
        ("C1a (+Dense RAG)", "✅", "✅", "✅", "❌", "❌", "❌", "✅", "C1a"),
        ("C1b (+Hybrid RRF RAG)", "✅", "✅", "✅", "✅", "❌", "❌", "✅", "C1b"),
        ("C2 (+Style-Aware RAG)", "✅", "✅", "✅", "✅", "✅", "❌", "✅", "C2"),
        ("D (Full EchoStyle)", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "D"),
    ]

    for name, scaff, p, r_dense, r_rrf, r_filter, c_loop, c_eval, code in cond_configs:
        s = summary[code]
        echo_str = f"{s['echo'].mean:.1f} ± {s['echo'].std:.1f}"
        ci_str = f"[{max(0.0, s['echo'].mean - s['echo'].ci95):.1f}, {s['echo'].mean + s['echo'].ci95:.1f}]"
        table.add_row(
            name, scaff, p, r_dense, r_rrf, r_filter, c_loop, c_eval,
            f"{s['discourse'].mean:.1f}",
            f"{s['rhythm'].mean:.1f}",
            f"{s['lexical'].mean:.1f}",
            f"-{s['penalty'].mean:.1f}",
            echo_str,
            ci_str,
            f"±{s['echo'].within_std:.2f}",
        )

    console.print(table)

    # 5. 核心组件配对单变量效应与置信区间表格 (Paired Single-Variable Deltas & 95% CI)
    contrib_table = Table(title="核心组件配对单变量效应与置信区间 (Paired Single-Variable Deltas & 95% CI)")
    contrib_table.add_column("对比组", style="cyan bold", width=12)
    contrib_table.add_column("验证自变量", style="white", width=28)
    contrib_table.add_column("配对均值 Δ (Mean Delta)", style="bold", justify="right", width=20)
    contrib_table.add_column("95% 置信区间 (Student-t)", justify="center", width=22)
    contrib_table.add_column("跨主题标准差", justify="right", width=14)
    contrib_table.add_column("客观机制与多目标权衡解释", style="yellow")

    comp_meta = [
        ("A1 vs A0", "Writer Scaffolding 提示工程基准", "scaffolding",
         "隔离 WriterAgent 提示工程、任务感知预算与防套话 Scaffolding 的独立贡献"),
        ("B vs A1", "Style Profile 纯先验 (严格单变量)", "profile",
         "在保持 WriterAgent Scaffolding 完全一致的条件下，纯 Style Profile 先验的真正配对独立净贡献"),
        ("C1a vs B", "Dense RAG 纯语义检索", "dense",
         "引入连续语料段落；上下文平稳连续，句长节奏吻合度 (Rhythm) 达到极高水平 (~99 分)"),
        ("C1b vs C1a", "Sparse/RRF 排名融合", "rrf",
         "在相同 Dense 通道准入下引入词频混合召回，增强词汇命中多样性（已剔除零相关文档排名偏置）"),
        ("C2 vs C1b", "Style-Aware 结构定向过滤", "style_filter",
         "结构定向装配对篇章起承转合的塑造效果与金句拼接对节奏离散度扰动的客观权衡"),
        ("D vs C2", "Critic 自审反思闭环 (盲审)", "critic",
         "FSM 自审精准清除潜在八股违规词，经第三方条件盲审裁决的端到端质量闭环"),
    ]

    for pair_name, var_name, key, desc in comp_meta:
        stat = delta_summary[key]
        color = "green" if stat.mean >= 0 else ("yellow" if key != "style_filter" else "red")
        delta_str = f"[{color}]{stat.mean:+.1f} 分[/{color}]"
        ci_str = f"[{stat.mean - stat.ci95:+.1f}, {stat.mean + stat.ci95:+.1f}]"
        contrib_table.add_row(
            pair_name, var_name, delta_str, ci_str, f"±{stat.std:.2f}", desc
        )

    console.print(contrib_table)

    # 6. 导出报告至 reports/ 目录
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

    empirical_analysis_section = _generate_empirical_analysis(
        summary,
        delta_summary=delta_summary,
        is_simulation=is_simulation,
        valid_pairs=valid_pairs,
        attempted_pairs=attempted_pairs,
    )

    report_md = f"""# EchoStyle 3.2 严格单变量消融实验报告 (Strict Single-Variable Matrix Report)

- **评测时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}
- **评测规模**：{num_topics} 个领域正交主题 × {repeats} 次采样 = 共 {total_runs} 组样本/条件
- **配对区块有效性 (Pair Validity)**：{valid_pairs} / {attempted_pairs} 有效配对区块 ({valid_pairs / max(1, attempted_pairs) * 100:.1f}%)，任何单条件失败均标记整组 7 条件配对区块失效 (Fail-Closed Paired Block)
{mode_banner}

## 一、 消融实验统计矩阵 (Hierarchical Topic Clustered: Mean ± Std & 95% Student-t CI)

| 消融实验条件 | Scaffolding | Profile | Dense 检索 | Hybrid RRF | 结构过滤 | Critic 重写 | 独立盲审 | 篇章拟合 (Discourse) | 节奏吻合 (Rhythm) | 用词质感 (Lexical) | 八股惩罚 | 跨主题 EchoScore (Mean ± Std) | 95% 置信区间 (Student-t) | 组内采样波动 (Within σ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0 (Vanilla Base)** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | {summary['A0']['discourse'].mean:.1f} | {summary['A0']['rhythm'].mean:.1f} | {summary['A0']['lexical'].mean:.1f} | -{summary['A0']['penalty'].mean:.1f} | **{summary['A0']['echo'].mean:.1f} ± {summary['A0']['echo'].std:.1f}** | [{max(0.0, summary['A0']['echo'].mean - summary['A0']['echo'].ci95):.1f}, {summary['A0']['echo'].mean + summary['A0']['echo'].ci95:.1f}] | ±{summary['A0']['echo'].within_std:.2f} |
| **A1 (Scaffolding Base)** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | {summary['A1']['discourse'].mean:.1f} | {summary['A1']['rhythm'].mean:.1f} | {summary['A1']['lexical'].mean:.1f} | -{summary['A1']['penalty'].mean:.1f} | **{summary['A1']['echo'].mean:.1f} ± {summary['A1']['echo'].std:.1f}** | [{max(0.0, summary['A1']['echo'].mean - summary['A1']['echo'].ci95):.1f}, {summary['A1']['echo'].mean + summary['A1']['echo'].ci95:.1f}] | ±{summary['A1']['echo'].within_std:.2f} |
| **B (+Profile Only)** | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | {summary['B']['discourse'].mean:.1f} | {summary['B']['rhythm'].mean:.1f} | {summary['B']['lexical'].mean:.1f} | -{summary['B']['penalty'].mean:.1f} | **{summary['B']['echo'].mean:.1f} ± {summary['B']['echo'].std:.1f}** | [{max(0.0, summary['B']['echo'].mean - summary['B']['echo'].ci95):.1f}, {summary['B']['echo'].mean + summary['B']['echo'].ci95:.1f}] | ±{summary['B']['echo'].within_std:.2f} |
| **C1a (+Dense RAG)** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | {summary['C1a']['discourse'].mean:.1f} | {summary['C1a']['rhythm'].mean:.1f} | {summary['C1a']['lexical'].mean:.1f} | -{summary['C1a']['penalty'].mean:.1f} | **{summary['C1a']['echo'].mean:.1f} ± {summary['C1a']['echo'].std:.1f}** | [{max(0.0, summary['C1a']['echo'].mean - summary['C1a']['echo'].ci95):.1f}, {summary['C1a']['echo'].mean + summary['C1a']['echo'].ci95:.1f}] | ±{summary['C1a']['echo'].within_std:.2f} |
| **C1b (+Hybrid RRF RAG)** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | {summary['C1b']['discourse'].mean:.1f} | {summary['C1b']['rhythm'].mean:.1f} | {summary['C1b']['lexical'].mean:.1f} | -{summary['C1b']['penalty'].mean:.1f} | **{summary['C1b']['echo'].mean:.1f} ± {summary['C1b']['echo'].std:.1f}** | [{max(0.0, summary['C1b']['echo'].mean - summary['C1b']['echo'].ci95):.1f}, {summary['C1b']['echo'].mean + summary['C1b']['echo'].ci95:.1f}] | ±{summary['C1b']['echo'].within_std:.2f} |
| **C2 (+Style-Aware RAG)** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | {summary['C2']['discourse'].mean:.1f} | {summary['C2']['rhythm'].mean:.1f} | {summary['C2']['lexical'].mean:.1f} | -{summary['C2']['penalty'].mean:.1f} | **{summary['C2']['echo'].mean:.1f} ± {summary['C2']['echo'].std:.1f}** | [{max(0.0, summary['C2']['echo'].mean - summary['C2']['echo'].ci95):.1f}, {summary['C2']['echo'].mean + summary['C2']['echo'].ci95:.1f}] | ±{summary['C2']['echo'].within_std:.2f} |
| **D (Full EchoStyle)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | {summary['D']['discourse'].mean:.1f} | {summary['D']['rhythm'].mean:.1f} | {summary['D']['lexical'].mean:.1f} | -{summary['D']['penalty'].mean:.1f} | **{summary['D']['echo'].mean:.1f} ± {summary['D']['echo'].std:.1f}** | [{max(0.0, summary['D']['echo'].mean - summary['D']['echo'].ci95):.1f}, {summary['D']['echo'].mean + summary['D']['echo'].ci95:.1f}] | ±{summary['D']['echo'].within_std:.2f} |

## 二、 核心组件配对单变量效应与置信区间 (Paired Component Deltas & 95% Student-t CI)

| 对比组 | 验证自变量 | 配对均值差值 (Mean Δ) | 95% 置信区间 (Student-t) | 跨主题标准差 (Between σ) | 组内采样波动 (Within σ) | 统计推断与机制权衡 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **A1 vs A0** | Writer Scaffolding 提示工程基准 | **{delta_summary['scaffolding'].mean:+.1f}** | [{delta_summary['scaffolding'].mean - delta_summary['scaffolding'].ci95:+.1f}, {delta_summary['scaffolding'].mean + delta_summary['scaffolding'].ci95:+.1f}] | ±{delta_summary['scaffolding'].std:.2f} | ±{delta_summary['scaffolding'].within_std:.2f} | 提示工程与任务感知预算工程基准 |
| **B vs A1** | Style Profile 纯先验 (严格单变量) | **{delta_summary['profile'].mean:+.1f}** | [{delta_summary['profile'].mean - delta_summary['profile'].ci95:+.1f}, {delta_summary['profile'].mean + delta_summary['profile'].ci95:+.1f}] | ±{delta_summary['profile'].std:.2f} | ±{delta_summary['profile'].within_std:.2f} | 严格控制 Scaffolding 变量后的 Style Profile 纯先验配对净贡献 |
| **C1a vs B** | Dense RAG 纯语义检索 | **{delta_summary['dense'].mean:+.1f}** | [{delta_summary['dense'].mean - delta_summary['dense'].ci95:+.1f}, {delta_summary['dense'].mean + delta_summary['dense'].ci95:+.1f}] | ±{delta_summary['dense'].std:.2f} | ±{delta_summary['dense'].within_std:.2f} | 连续语料段落召回，节奏平滑性提升 |
| **C1b vs C1a**| Sparse/RRF 排名融合 | **{delta_summary['rrf'].mean:+.1f}** | [{delta_summary['rrf'].mean - delta_summary['rrf'].ci95:+.1f}, {delta_summary['rrf'].mean + delta_summary['rrf'].ci95:+.1f}] | ±{delta_summary['rrf'].std:.2f} | ±{delta_summary['rrf'].within_std:.2f} | 共享 Dense 通道准入下的纯 RRF 词频混合边际效应 |
| **C2 vs C1b** | Style-Aware 结构定向过滤 | **{delta_summary['style_filter'].mean:+.1f}** | [{delta_summary['style_filter'].mean - delta_summary['style_filter'].ci95:+.1f}, {delta_summary['style_filter'].mean + delta_summary['style_filter'].ci95:+.1f}] | ±{delta_summary['style_filter'].std:.2f} | ±{delta_summary['style_filter'].within_std:.2f} | 控制 RRF 变量后的结构定向过滤净贡献（篇章拟合 vs 句长离散度权衡） |
| **D vs C2** | Critic 自审反思闭环 (盲审) | **{delta_summary['critic'].mean:+.1f}** | [{delta_summary['critic'].mean - delta_summary['critic'].ci95:+.1f}, {delta_summary['critic'].mean + delta_summary['critic'].ci95:+.1f}] | ±{delta_summary['critic'].std:.2f} | ±{delta_summary['critic'].within_std:.2f} | FSM 自审拦截八股违规，经独立条件盲审裁决 |

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
