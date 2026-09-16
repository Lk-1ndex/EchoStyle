import os
import sys
import time
from pathlib import Path

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
from src.agents.coordinator import CoordinatorAgent
from src.agents.state import AgentState

console = Console()

# ================= 真实作者公开样文语料基准 (Benchmark Corpus) =================
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


def run_benchmark():
    console.print(Panel.fit("[bold magenta]EchoStyle 2.5 — 真实作者文风建模与智能仿写 Benchmark 评测基准[/bold magenta]"))

    config = load_config()
    if not config.llm.api_key:
        console.print("[bold red]错误：未检测到有效 API Key，请在 config.yaml 中配置！[/bold red]")
        sys.exit(1)

    coordinator = CoordinatorAgent(config)
    state = AgentState(topic="在机器泛滥的时代，为什么真挚独特的文风更稀缺？")

    # 1. 准备样文语料
    corpus_dir = Path("./profiles/benchmark_samples")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    sample_file_1 = corpus_dir / "sample_1.txt"
    sample_file_2 = corpus_dir / "sample_2.txt"
    sample_file_1.write_text(SAMPLE_ESSAY_1, encoding="utf-8")
    sample_file_2.write_text(SAMPLE_ESSAY_2, encoding="utf-8")

    console.print("\n[bold cyan]>> Step 1: Extractor Agent 样文感知与提取[/bold cyan]")
    start_time = time.time()
    extracted_samples = coordinator.extract_sources([str(sample_file_1), str(sample_file_2)], state=state)
    extract_duration = time.time() - start_time
    console.print(f"[green]+[/green] 成功提取 {len(extracted_samples)} 篇样文，耗时: {extract_duration:.2f}s")

    # 2. 深度文风建模与记忆切片入库
    console.print("\n[bold cyan]>> Step 2: Analyst Agent 深度建模与 Style Memory 切片[/bold cyan]")
    model_start = time.time()
    deep_profile = coordinator.build_style(
        extracted_samples,
        profile_name="Benchmark_犀利独立思考风",
        state=state
    )
    model_duration = time.time() - model_start
    console.print(f"[green]+[/green] 深度建模与记忆切片入库完成，耗时: {model_duration:.2f}s")

    # 打印客观量化指标
    q = deep_profile.quantitative
    stat_table = Table(title="作者客观统计语言学指纹 (Stylometrics)")
    stat_table.add_column("指标维度", style="cyan")
    stat_table.add_column("数值", style="green")
    stat_table.add_column("学术参考意义", style="yellow")
    stat_table.add_row("平均句长", f"{q.avg_sentence_length:.1f} 字/句", "短句爆发力控制")
    stat_table.add_row("句长标准差 (离散度)", f"{q.sentence_length_std:.1f}", "呼吸节奏起伏波长")
    stat_table.add_row("词汇丰富度 (TTR)", f"{q.ttr:.3f}", "用词多样性与独立词频比")
    stat_table.add_row("标点符号信息熵", f"{q.punctuation_entropy:.2f}", "标点多样性分布")
    stat_table.add_row("转折词密度", f"{q.transition_density:.1f} 次/千字", "逻辑递进频率")
    console.print(stat_table)

    # 3. 执行创作与自省重构
    console.print("\n[bold cyan]>> Step 3: Writer & Critic 协作创作与 EchoEval 严苛质检[/bold cyan]")
    gen_start = time.time()
    benchmark_topic = "在机器泛滥的时代，为什么真挚独特的文风更稀缺？"
    benchmark_points = """
- 现状：大模型批量产出海量‘正确却毫无灵魂’的翻译腔废话
- 根源：文风是个人阅历、伤疤与偏见的结晶，无法被均值算法替代
- 态度：拒绝四平八稳的官样文章，保持语言的尖锐与肉身呼吸感
"""
    final_article, report, final_state = coordinator.generate_article(
        profile=deep_profile,
        topic=benchmark_topic,
        key_points=benchmark_points,
        word_count=1200,
        target_audience="热爱深度思考的创作者",
        state=state
    )
    gen_duration = time.time() - gen_start

    # 4. 打印 EchoEval 最终打分卡
    console.print(Panel(final_article[:600] + "\n\n...(节选，完整文章已保存)...", title="[bold green]终审成文预览[/bold green]"))

    eval_table = Table(title="EchoEval 标准化质量评测报告卡")
    eval_table.add_column("评测指标", style="cyan")
    eval_table.add_column("权重", style="magenta")
    eval_table.add_column("得分", style="green")
    eval_table.add_row("文风神似度 (Style Fidelity)", "35%", f"{report.style_fidelity:.1f} / 100")
    eval_table.add_row("人类读者好感度 (LLM Judge)", "25%", f"{report.llm_judge_score:.1f} / 100")
    eval_table.add_row("句式统计拟合度 (Stylometrics Fit)", "20%", f"{report.stylometric_similarity:.1f} / 100")
    eval_table.add_row("逻辑深度 (Logic Depth)", "20%", f"{report.logic_depth:.1f} / 100")
    eval_table.add_row("去 AI 味八股惩罚", "扣分项", f"-{report.ai_penalty:.1f} 分 (捕获: {len(report.detected_cliches)}个)")
    eval_table.add_row("[bold yellow]综合最终得分 (Final Weighted Score)[/bold yellow]", "[bold yellow]100%[/bold yellow]", f"[bold yellow]{report.overall_score:.1f} / 100[/bold yellow]")
    console.print(eval_table)

    console.print(f"[bold yellow]总编辑批注反馈:[/bold yellow] {report.feedback}")
    console.print(f"[bold cyan]草稿演进链版本数:[/bold cyan] {len(final_state.draft_chain)} 版 (反思重写轮次: {final_state.retry_count} 轮)")
    console.print(f"[bold cyan]全流程总耗时:[/bold cyan] {extract_duration + model_duration + gen_duration:.2f}s")

    # 保存 Benchmark 报告
    report_file = Path("./profiles/benchmark_report.md")
    report_md = f"""# EchoStyle 2.5 基准测试报告 (Benchmark Report)

- **基准测试时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}
- **评测主题**：{benchmark_topic}
- **综合评测最终得分**：**{report.overall_score:.1f} / 100**
- **草稿版本迭代数**：{len(final_state.draft_chain)} 版 (反思重试: {final_state.retry_count} 轮)

## 一、 客观统计语言学指纹 (Stylometrics)
- 平均句长：{q.avg_sentence_length:.1f} 字/句
- 句长标准差：{q.sentence_length_std:.1f}
- 词汇丰富度 (TTR)：{q.ttr:.3f}
- 标点多样性信息熵：{q.punctuation_entropy:.2f}
- 转折词密度：{q.transition_density:.1f} 次/千字

## 二、 EchoEval 细分维度得分
- 文风神似度：{report.style_fidelity:.1f} (权重 35%)
- 读者好感度：{report.llm_judge_score:.1f} (权重 25%)
- 统计拟合度：{report.stylometric_similarity:.1f} (权重 20%)
- 逻辑论述深度：{report.logic_depth:.1f} (权重 20%)
- 违规八股惩罚：-{report.ai_penalty:.1f} 分 (违规词列表: {report.detected_cliches})

## 三、 终审成文全文
{final_article}
"""
    report_file.write_text(report_md, encoding="utf-8")
    console.print(f"\n[bold green]基准测试圆满完成！详细报告已持久化至:[/bold green] {report_file}")


if __name__ == "__main__":
    run_benchmark()
