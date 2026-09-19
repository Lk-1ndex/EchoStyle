import pytest
from src.extractors.sanitizer import TextSanitizer
from src.core.models import (
    StyleProfile,
    TonePersona,
    CadenceSyntax,
    LexiconRhetoric,
    DiscourseArchitecture,
    AntiPatterns,
)


def test_wechat_noise_removal():
    raw_text = """
点击上方蓝字关注我们获取每日干货！
这是一篇关于人工智能的深度文章。我们正在经历一场前所未有的技术变革。
喜欢本文请点个在看或者分享给好友！
作者：张三
往期推荐：2024年最值得关注的AI技术
"""
    cleaned = TextSanitizer.clean(raw_text, repair_linebreaks=False)
    assert "点击上方" not in cleaned
    assert "在看" not in cleaned
    assert "作者：" not in cleaned
    assert "往期推荐" not in cleaned
    assert "这是一篇关于人工智能的深度文章" in cleaned


def test_broken_line_repair_chinese():
    raw_text = "今天的天气非常不错，我打算去公园散散步\n顺便思考一下最近的项目架构。\n\n这是新的一段话。"
    cleaned = TextSanitizer.clean(raw_text, repair_linebreaks=True)
    # 因为上一行末尾是“散步”，无句末标点，下一行以“顺便”开始，应该被平滑合并
    assert "今天的天气非常不错，我打算去公园散散步顺便思考一下最近的项目架构。" in cleaned
    # 新段落空行应该被保留
    assert "这是新的一段话。" in cleaned


def test_english_hyphenation_repair():
    raw_text = "This is a great connec-\ntion between two concepts."
    cleaned = TextSanitizer.clean(raw_text, repair_linebreaks=True)
    assert "connection" in cleaned


def test_markdown_structure_preservation():
    raw_text = """
# 核心标题
- 列表项1
- 列表项2

> 这是引言块
正文第一句话。
"""
    cleaned = TextSanitizer.clean(raw_text, repair_linebreaks=True)
    assert "# 核心标题" in cleaned
    assert "- 列表项1" in cleaned
    assert "> 这是引言块" in cleaned


def test_pdf_artifact_cleanup_removes_control_cid_and_html_wrappers():
    raw_text = "正文\x03\x06 (cid:42) <small><span class=\"footnote\">脚注<sup>1</sup></span></small>"
    cleaned = TextSanitizer.clean(raw_text, clean_pdf_artifacts=True)

    assert "\x03" not in cleaned
    assert "\x06" not in cleaned
    assert "cid:42" not in cleaned
    assert "<small>" not in cleaned
    assert "<span" not in cleaned
    assert "脚注1" in cleaned


def test_style_profile_system_prompt():
    profile = StyleProfile(
        name="测试文风",
        tone_persona=TonePersona(
            perspective="第一人称'我'",
            emotional_tone="犀利且带自嘲",
            persona_traits=["资深黑客", "技术理想主义者"]
        ),
        cadence_syntax=CadenceSyntax(
            sentence_style="短句密集，节奏轻快",
            paragraph_habit="1-2句话成段",
            punctuation_habits=["频繁使用破折号", "喜欢用设问句"]
        ),
        lexicon_rhetoric=LexiconRhetoric(
            catchphrases=["说白了", "别闹了"],
            metaphor_style="用日常柴米油盐类比技术",
            vocabulary_richness="通俗大白话混搭硬核技术黑话"
        ),
        discourse=DiscourseArchitecture(
            opening_hook="直击痛点开局",
            body_progression="层层驳斥认知误区",
            ending_style="留白引发思考"
        ),
        anti_patterns=AntiPatterns(
            forbidden_words=["总而言之", "不可否认", "值得一提的是"]
        ),
        exemplar_snippets=["说白了，技术从来不是壁垒，傲慢才是。"]
    )

    sys_prompt = profile.to_system_prompt()
    assert "犀利且带自嘲" in sys_prompt
    assert "说白了" in sys_prompt
    assert "总而言之" in sys_prompt
    assert "范例 1" in sys_prompt
