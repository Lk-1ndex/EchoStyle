import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

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

console = Console()

FAILURE_CASES = [
    {
        "case_id": "CASE-01",
        "title": "跨领域题材冲突 (Cross-Domain Stylistic Incompatibility)",
        "scenario": "将强烈的【犀利冷峻文学批判文风】强行套用于【严肃技术规范 / API 架构说明】",
        "prompt_input": "请以作者风格撰写一份《分布式微服务 OAuth2.0 Token 刷新机制规范》。",
        "manifestation": "生成文本充斥着‘说白了’、‘精致的懦弱’等文学批判口吻，严重破坏了技术文档客观、严谨、无歧义的交付标准。",
        "root_cause": "文风记忆库中的高光隐喻与强情绪词，与任务领域的严肃客观语用场景发生语义对抗。",
        "system_solution": "【题材冲突侦测守卫】在 Coordinator 阶段对任务主题进行领域敏感度分类；技术/法律类主题自动抑制情绪化口癖并降级修辞强度。"
    },
    {
        "case_id": "CASE-02",
        "title": "语料极度匮乏导致检索退化 (Sparse Sample Cold-Start Degradation)",
        "scenario": "用户仅上传 1 篇极短碎语（不足 150 字），要求进行全流程建模与生成",
        "prompt_input": "提供 1 篇仅 80 字的生活随笔，要求生成 1500 字深度商业评论。",
        "manifestation": "Style Memory 仅切出 1 个片段，无法实现 hook/quote/conclusion 的结构化定向装配，RAG 检索退化为单一文本复读。",
        "root_cause": "篇章功能特征提取需要足够的段落密度支持，过短样本导致切片方差为零。",
        "system_solution": "【冷启动准入阈值】Extractor Agent 设定最低字数质检（>=300字）；样本不足时提示‘语料不足以支撑篇章结构切片，降级为纯统计句法约束模式’。"
    },
    {
        "case_id": "CASE-03",
        "title": "过度自审导致平庸化 (Over-Refinement into Mediocrity)",
        "scenario": "Critic 质检阈值设为极端高位 (quality_threshold >= 95.0)，触发 3 轮以上连续强制反思改写",
        "prompt_input": "主题《为何现代人热衷于精神内耗》，设置 quality_threshold=95.0 进行 3 轮连续自审改写。",
        "manifestation": "每一轮改写模型都为了迎合‘无违规套话’而删改棱角，最终第 3 版成文虽毫无挑剔之处，但也丧失了作者独特的刺痛感，退化为光滑温吞的平庸之作。",
        "root_cause": "多轮自审的边际惩罚累计，导致生成模型采取过度防御性生成策略（Defensive Generation）。",
        "system_solution": "【FSM 快照回滚保底机制】Coordinator 记录每轮草稿的 EchoScore；若重构后分数反而下降或个性抹平，自动执行 `rollback_to('best_draft_checkpoint')` 回退至最优历史版本！"
    }
]


def run_failure_analysis():
    console.print(Panel.fit(
        "[bold cyan]EchoStyle 3.2 — 失败案例深度剖析套件 (Failure Modes & Case Studies)[/bold cyan]\n"
        "[white]技术严谨性必须经受边界检验：深入解剖跨领域题材冲突、冷启动样本匮乏与过度自审平庸化 3 大失败场景及系统级防御应对[/white]"
    ))

    table = Table(title="系统失败边界与防御缓解策略 (Failure Modes Matrix)")
    table.add_column("编号", style="cyan bold", width=8)
    table.add_column("失败场景分类", style="yellow bold", width=22)
    table.add_column("典型表现与症状", style="white", width=28)
    table.add_column("根因分析", style="magenta", width=24)
    table.add_column("EchoStyle 架构级解法", style="green", width=30)

    for case in FAILURE_CASES:
        table.add_row(
            case["case_id"],
            case["title"],
            case["manifestation"],
            case["root_cause"],
            case["system_solution"]
        )

    console.print(table)

    # 导出报告
    report_file = Path("./profiles/failure_cases_analysis.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_md = f"""# EchoStyle 3.2 失败案例与系统边界深度剖析报告 (Failure Analysis Report)

- **分析时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}
- **核心宗旨**：真正的工程落地系统不回避失败，必须明确系统工作边界，并在架构中设计自愈与降级机制。

---

"""
    for case in FAILURE_CASES:
        report_md += f"""## [{case['case_id']}] {case['title']}

### 1. 触发场景与输入
{case['scenario']}
- **测试输入**：`{case['prompt_input']}`

### 2. 失败症状与具体表现
{case['manifestation']}

### 3. 根因技术剖析 (Root Cause)
{case['root_cause']}

### 4. EchoStyle 架构级防御与缓解策略
{case['system_solution']}

---
"""

    report_file.write_text(report_md, encoding="utf-8")
    console.print(f"\n[bold green]失败案例分析报告已成功归档至:[/bold green] {report_file}")


if __name__ == "__main__":
    run_failure_analysis()
