import json
from unittest.mock import MagicMock

import pytest

from src.analyzer.distiller import StyleDistiller
from src.core.config import LLMConfig


def valid_profile_json() -> str:
    return json.dumps(
        {
            "tone_persona": {
                "perspective": "第一人称",
                "emotional_tone": "冷静克制",
                "persona_traits": ["严谨", "善于类比"],
            },
            "cadence_syntax": {
                "sentence_style": "长短句交替",
                "paragraph_habit": "紧凑",
                "punctuation_habits": ["常用分号"],
            },
            "lexicon_rhetoric": {
                "catchphrases": ["换句话说"],
                "metaphor_style": "使用具体物理图景",
                "vocabulary_richness": "学术术语与通俗表达结合",
            },
            "discourse": {
                "opening_hook": "从问题切入",
                "body_progression": "由现象递进到机制",
                "ending_style": "开放式结论",
                "opening_pattern": "paradox_hook",
                "progression_pattern": "inductive",
                "ending_pattern": "open_question",
            },
            "anti_patterns": {
                "forbidden_words": ["总而言之"],
                "forbidden_structures": ["僵硬三段论"],
            },
            "exemplar_snippets": ["这是原文中的代表性片段。"],
        },
        ensure_ascii=False,
    )


def valid_recovery_sections():
    payload = json.loads(valid_profile_json())
    return [
        json.dumps(
            {key: payload[key] for key in ("tone_persona", "cadence_syntax")},
            ensure_ascii=False,
        ),
        json.dumps(
            {key: payload[key] for key in ("lexicon_rhetoric", "discourse")},
            ensure_ascii=False,
        ),
        json.dumps(
            {key: payload[key] for key in ("anti_patterns", "exemplar_snippets")},
            ensure_ascii=False,
        ),
    ]


@pytest.mark.parametrize("first_response", ["not-json", "{}"])
def test_style_distiller_retries_parse_or_schema_failure(first_response):
    distiller = StyleDistiller(LLMConfig(max_tokens=4096))
    distiller.model_provider.chat_completion = MagicMock(
        side_effect=[first_response, *valid_recovery_sections()]
    )

    profile = distiller.distill(["这是一篇用于测试文风分析的中文样文。"], profile_name="测试画像")

    assert profile.name == "测试画像"
    assert profile.tone_persona.perspective == "第一人称"
    assert distiller.last_retry_used is True
    assert distiller.model_provider.chat_completion.call_count == 4
    retry_call = distiller.model_provider.chat_completion.call_args_list[1]
    assert retry_call.kwargs["temperature"] == 0.0
    assert retry_call.kwargs["json_mode"] is True
    assert retry_call.kwargs["max_tokens"] == 2048
    assert "本次只分析" in retry_call.kwargs["system_prompt"]


def test_style_distiller_extracts_embedded_json_without_retry():
    distiller = StyleDistiller(LLMConfig())
    wrapped = f"分析完成。\n```json\n{valid_profile_json()}\n```\n以上为结果。"
    distiller.model_provider.chat_completion = MagicMock(return_value=wrapped)

    profile = distiller.distill(["测试正文"])

    assert profile.tone_persona.emotional_tone == "冷静克制"
    assert distiller.last_retry_used is False
    distiller.model_provider.chat_completion.assert_called_once()


def test_style_distiller_surfaces_error_after_retry_is_also_invalid():
    distiller = StyleDistiller(LLMConfig())
    distiller.model_provider.chat_completion = MagicMock(side_effect=["broken", "still broken"])

    with pytest.raises(ValueError, match="分段恢复失败"):
        distiller.distill(["测试正文"])

    assert distiller.last_retry_used is True


def test_style_distiller_never_accepts_complete_nested_section_as_full_profile():
    distiller = StyleDistiller(LLMConfig())
    truncated_outer_json = (
        '{"tone_persona": '
        '{"perspective":"第一人称","emotional_tone":"冷静","persona_traits":[]},'
        '"cadence_syntax":'
    )

    with pytest.raises(ValueError, match="必须包含顶层字段"):
        distiller._extract_json(
            truncated_outer_json,
            expected_keys={
                "tone_persona",
                "cadence_syntax",
                "lexicon_rhetoric",
                "discourse",
                "anti_patterns",
                "exemplar_snippets",
            },
        )
