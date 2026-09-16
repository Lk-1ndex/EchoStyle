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
