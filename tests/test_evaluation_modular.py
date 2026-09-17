from src.core.models import DeepStyleProfile, StyleProfile, TonePersona, CadenceSyntax, LexiconRhetoric, DiscourseArchitecture, AntiPatterns
from src.analyzer.stylometrics import StylometricsAnalyzer
from src.evaluation.composite_eval import CompositeEvaluator


def test_composite_echoscore_formula():
    profile = DeepStyleProfile(
        name="测试档案",
        qualitative=StyleProfile(
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="犀利", persona_traits=["独立"]),
            cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="紧凑", punctuation_habits=[]),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=["说白了"], metaphor_style="具象", vocabulary_richness="丰富"),
            discourse=DiscourseArchitecture(opening_hook="设问", body_progression="推进", ending_style="金句"),
            anti_patterns=AntiPatterns(forbidden_words=["总而言之"]),
        ),
        quantitative=StylometricsAnalyzer.analyze("说白了，很多人在搞高级信息搬运。真正的思考是带着偏见的价值判断。别闹了。")
    )

    text = """说白了，很多人不过是高级信息搬运工。
别闹了！真正的思考从来不是拼图游戏。
我们必须守住这个阵地！"""

    result = CompositeEvaluator.calculate_echoscore(
        generated_text=text,
        target_metrics=profile.quantitative,
        target_profile=profile,
        llm_fidelity_score=90.0
    )

    assert 0.0 <= result.echo_score <= 100.0
    assert result.style_fidelity == 90.0
    assert result.discourse_fit > 70.0
    assert result.rhythm_match > 70.0
    assert result.cliche_penalty == 0.0


def test_composite_echoscore_penalizes_custom_forbidden_words():
    """验证 CompositeEvaluator 正确继承并惩罚 DeepStyleProfile 中声明的特定违规词"""
    profile = DeepStyleProfile(
        name="测试自定义违规词",
        qualitative=StyleProfile(
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="直白"),
            cadence_syntax=CadenceSyntax(sentence_style="短句", paragraph_habit="紧凑"),
            lexicon_rhetoric=LexiconRhetoric(catchphrases=[], metaphor_style="生活化", vocabulary_richness="通俗"),
            discourse=DiscourseArchitecture(opening_hook="设问", body_progression="递进", ending_style="金句"),
            anti_patterns=AntiPatterns(forbidden_words=["不可替代", "大势所趋"]),
        ),
        quantitative=StylometricsAnalyzer.analyze("短小精悍的基准测试样文。")
    )

    # 包含自定义违规词 "不可替代" 的文本
    text = "这种手艺在当下是不可替代的。我们必须坚持到底。"

    result = CompositeEvaluator.calculate_echoscore(
        generated_text=text,
        target_metrics=profile.quantitative,
        target_profile=profile,
        llm_fidelity_score=85.0
    )

    # 核心断言：检测到自定义违规词，扣除 10 分
    assert result.cliche_penalty == 10.0
    assert "不可替代" in result.detected_cliches

