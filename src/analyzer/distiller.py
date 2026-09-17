import json
import re
from pathlib import Path
from typing import List, Optional

from src.core.models import StyleProfile
from src.core.config import LLMConfig
from src.core.model_provider import ModelProvider
from .prompts import STYLE_DISTILLATION_SYSTEM_PROMPT, STYLE_DISTILLATION_USER_PROMPT_TEMPLATE


class StyleDistiller:
    """
    文风逆向工程蒸馏器：
    利用大模型反向解构样本文章，提炼出可操作、结构化的 StyleProfile。
    """

    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config
        self.model_provider = ModelProvider(llm_config)

    def distill(self, articles: List[str], profile_name: str = "我的文风") -> StyleProfile:
        """
        对多篇文章样本进行文风提炼
        """
        if not articles:
            raise ValueError("至少需要提供 1 篇文章样本进行分析！")

        # 组合文章样本
        combined_corpus = ""
        for idx, art in enumerate(articles, 1):
            combined_corpus += f"\n\n=================== 样本文章 {idx} ===================\n{art.strip()}\n"

        user_prompt = STYLE_DISTILLATION_USER_PROMPT_TEMPLATE.format(corpus_content=combined_corpus)

        # 调用统一 ModelProvider
        raw_response = self.model_provider.chat_completion(
            system_prompt=STYLE_DISTILLATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            json_mode=True
        )

        # 解析与容错提取 JSON
        profile_dict = self._extract_json(raw_response)
        profile_dict["name"] = profile_name

        # 实例化 Pydantic 模型
        style_profile = StyleProfile(**profile_dict)
        return style_profile

    def save_profile(self, profile: StyleProfile, output_dir: str = "./profiles") -> Path:
        """
        将文风档案保存为 JSON 和易于阅读的 Markdown
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. 保存 JSON
        json_file = out_path / f"{profile.name}_profile.json"
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(profile.model_dump_json(indent=2))

        # 2. 保存易读的 Markdown 风格指南
        md_file = out_path / f"{profile.name}_guide.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(profile.to_system_prompt())

        return json_file

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n", "", text)
            text = re.sub(r"\n```$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                raw_json = match.group(0)
                try:
                    return json.loads(raw_json)
                except json.JSONDecodeError:
                    # 去除行末多余逗号等常见格式瑕疵
                    cleaned = re.sub(r",\s*([\]}])", r"\1", raw_json)
                    try:
                        return json.loads(cleaned)
                    except Exception:
                        pass
            raise ValueError(f"无法从 LLM 返回的内容中解析出合法 JSON: {text[:200]}...")
