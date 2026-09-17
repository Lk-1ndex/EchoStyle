import argparse
import sys
from pathlib import Path

# 保证 Windows 控制台 UTF-8 输出无乱码
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
from src.core.models import DeepStyleProfile, StyleProfile
from src.agents.coordinator import CoordinatorAgent
from src.agents.state import AgentState

console = Console()


def main():
    parser = argparse.ArgumentParser(description="EchoStyle 2.5: 基于 LLM Agent 的个人文风建模与可控智能创作系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 提取命令
    p_extract = subparsers.add_parser("extract", help="Extractor Agent 自适应感知并提取文章")
    p_extract.add_argument("-s", "--source", required=True, help="文章 URL 或文件路径")
    p_extract.add_argument("-o", "--output", help="输出 Markdown 路径 (可选)")

    # 建模与记忆入库命令
    p_distill = subparsers.add_parser("distill", help="Analyst Agent 双驱建模与风格记忆向量入库")
    p_distill.add_argument("-i", "--inputs", nargs="+", required=True, help="样例文档路径列表或 URL")
    p_distill.add_argument("-n", "--name", default="深度文风档案", help="文风档案名称")

    # 协作创作与评测命令
    p_write = subparsers.add_parser("write", help="Coordinator 统筹多智能体协作创作与自查反思")
    p_write.add_argument("-p", "--profile", required=True, help="深度文风档案 JSON 路径")
    p_write.add_argument("-t", "--topic", required=True, help="新文章主题")
    p_write.add_argument("-k", "--keypoints", default="", help="核心论点或要点")
    p_write.add_argument("-w", "--words", type=int, default=1500, help="目标字数")
    p_write.add_argument("-o", "--output", help="最终成文保存路径")

    # 基准与科学评测命令
    p_bench = subparsers.add_parser("benchmark", help="运行文风建模与科学评测基准套件")
    p_bench.add_argument("--ab", action="store_true", help="运行完整 A/B 对照实验 (Baseline 0 vs Baseline 1 vs EchoStyle)")
    p_bench.add_argument("--ablation", action="store_true", help="运行 5 组严谨消融实验 (Ablation Study: 5-Condition Matrix)")
    p_bench.add_argument("--scaling", action="store_true", help="运行 20 篇样本规模渐近收敛实验 (20-Sample Scaling Experiment with MSE)")
    p_bench.add_argument("--blind", action="store_true", help="运行规范化双盲评测 (Blind Pairwise Benchmark with 95%% Wilson CI)")
    p_bench.add_argument("--failure", action="store_true", help="运行失败案例与系统边界深度剖析 (Failure Modes & Case Studies)")
    p_bench.add_argument("--simulate", action="store_true", help="离线模拟模式 (无真实 API Key 时用于测试演示)")
    p_bench.add_argument("--topics", type=int, default=3, help="消融实验测试题目数量 (1-5, 默认 3)")
    p_bench.add_argument("--repeat", type=int, default=1, help="每个题目重复轮次 (默认 1)")

    args = parser.parse_args()
    config = load_config()
    coordinator = CoordinatorAgent(config)
    state = AgentState()

    if args.command == "extract":
        console.print(f"[bold cyan]Extractor Agent 正在感知并解析:[/bold cyan] {args.source}")
        results = coordinator.extract_sources([args.source], state=state)
        article = results[0]
        text = article["content"]
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            console.print(f"[bold green]已保存至:[/bold green] {args.output}")
        else:
            console.print(Panel(text[:1000] + ("\n...(已截断预览)..." if len(text) > 1000 else ""), title=f"提取结果 ({article['engine_used']})"))

    elif args.command == "distill":
        if not config.llm.api_key:
            console.print("[bold red]错误：未在 config.yaml 中配置大模型 API Key！[/bold red]")
            sys.exit(1)

        extracted_articles = coordinator.extract_sources(args.inputs, state=state)
        console.print("[bold cyan]Analyst Agent 正在计算统计语言学特征并进行语义解构...[/bold cyan]")
        deep_profile = coordinator.build_style(extracted_articles, profile_name=args.name, state=state)

        # 保存档案
        out_dir = Path("./profiles")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{args.name}_deep_profile.json"
        json_path.write_text(deep_profile.model_dump_json(indent=2), encoding="utf-8")

        console.print(f"[bold green]深度文风建模完成！档案已保存至:[/bold green] {json_path}")
        if deep_profile.quantitative:
            q = deep_profile.quantitative
            table = Table(title="统计语言学客观指标 (Stylometrics)")
            table.add_column("指标名", style="cyan")
            table.add_column("数值", style="magenta")
            table.add_row("平均句长 (字/句)", f"{q.avg_sentence_length:.1f}")
            table.add_row("句长波动标准差", f"{q.sentence_length_std:.1f}")
            table.add_row("词汇丰富度 (TTR)", f"{q.ttr:.3f}")
            table.add_row("标点信息熵", f"{q.punctuation_entropy:.2f}")
            table.add_row("转折词密度 (次/千字)", f"{q.transition_density:.1f}")
            console.print(table)

    elif args.command == "write":
        if not config.llm.api_key:
            console.print("[bold red]错误：未在 config.yaml 中配置大模型 API Key！[/bold red]")
            sys.exit(1)

        profile_path = Path(args.profile)
        if not profile_path.exists():
            console.print(f"[bold red]文风档案不存在: {args.profile}[/bold red]")
            sys.exit(1)

        deep_profile = DeepStyleProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))
        console.print(f"[bold cyan]Coordinator 启动创作任务: [{args.topic}] (文风: {deep_profile.name})...[/bold cyan]")

        final_article, report, final_state = coordinator.generate_article(
            profile=deep_profile,
            topic=args.topic,
            key_points=args.keypoints,
            word_count=args.words,
            state=state
        )

        # 打印评测报告
        table = Table(title="EchoEval 质量量化评估报告")
        table.add_column("评测维度", style="cyan")
        table.add_column("得分", style="green")
        table.add_row("综合质量得分", f"{report.overall_score:.1f} / 100")
        table.add_row("去 AI 味纯净度", f"{report.anti_ai_score:.1f} / 100")
        table.add_row("句式统计拟合度", f"{report.stylometric_similarity:.1f} / 100")
        table.add_row("LLM 专家仲裁分", f"{report.llm_judge_score:.1f} / 100")
        console.print(table)
        console.print(f"[bold yellow]总编辑审校评语:[/bold yellow] {report.feedback}")

        if args.output:
            Path(args.output).write_text(final_article, encoding="utf-8")
            console.print(f"[bold green]成文已成功保存至:[/bold green] {args.output}")
        else:
            console.print(Panel(final_article, title="终审成文"))

    elif args.command == "benchmark":
        if args.ablation:
            from experiments.ablation_study import run_ablation_study
            run_ablation_study(topics_count=args.topics, repeats=args.repeat, simulate=args.simulate)
        elif args.scaling:
            from experiments.scaling_study import run_scaling_study
            run_scaling_study()
        elif args.blind:
            from experiments.blind_benchmark import run_blind_benchmark
            run_blind_benchmark(simulate=args.simulate)
        elif args.failure:
            from experiments.failure_analysis import run_failure_analysis
            run_failure_analysis()
        elif args.ab:
            from experiments.ab_benchmark import run_ab_benchmark
            run_ab_benchmark()
        else:
            from benchmark import run_benchmark
            run_benchmark()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
