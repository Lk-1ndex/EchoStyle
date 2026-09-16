from src.analyzer.stylometrics import StylometricsAnalyzer
from src.evaluation.rhythm_metrics import RhythmEvaluator


def test_steady_academic_author_rhythm_fit():
    # 模拟严肃/学术作者：长句多、长短句波动标准差本身就很小 (std=5.0)
    academic_corpus = """根据本次实验对大语言模型微调收敛特性的系统性观测与数理统计分析。
通过引入严格的状态机守卫机制使得多智能体交互的执行轨迹具备了可解释性与完备性。
在不同参数规模与网络拓扑结构约束下的收敛表现均符合渐近正态分布的一般规律。"""
    academic_target = StylometricsAnalyzer.analyze(academic_corpus)
    assert academic_target.sentence_length_std < 8.0  # 真实方差确实很小

    # 仿写文本同样保持了学术严谨与稳定的平缓节奏 (std 也很小)
    good_academic_draft = """基于严谨控制变量法对两组异构智能体系统的推理延迟与显存开销进行基准测定。
该架构有效隔离了模型交互过程中的随机扰动从而显著提升了端到端输出的一致性。
进一步结合倒数排名融合算法完成了多模态异构特征向量空间的无量纲对齐与投影。"""

    # 在旧版中，如果盲目惩罚小方差，就会错误地惩罚这个学术仿写
    # 在新版 Rhythm Deviation 中，因为生成文本与目标作者节奏高度一致，得分应保持在高位 (>=80)
    score, breakdown = RhythmEvaluator.evaluate_rhythm_fit(good_academic_draft, academic_target)
    assert score >= 80.0
    assert breakdown["delta_std"] < 5.0
