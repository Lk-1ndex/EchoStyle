from typing import List, Optional, Tuple
import httpx
from src.core.models import StyleProfile
from src.core.config import LLMConfig
from .prompts import (
    DRAFT_USER_PROMPT_TEMPLATE,
    CRITIQUE_SYSTEM_PROMPT,
    CRITIQUE_USER_PROMPT_TEMPLATE,
)


class ArticleSynthesizer:
    """
    风格注入创作生成器：
    执行【初稿创作】 -> 【去 AI 味二次自审校准】的双阶段闭环生成。
    """

    def __init__(self, llm_config: LLMConfig):
        self.config = llm_config

    def generate(
        self,
        style_profile: StyleProfile,
        topic: str,
        key_points: str,
        word_count: int = 1500,
        target_audience: str = "大众读者 / 关注该领域的专业群体",
        enable_critique: bool = True,
    ) -> Tuple[str, Optional[str]]:
        """
        生成文章：
        :return: (最终成文, 初稿内容)
        """
        # ================= 阶段 1：初稿生成 =================
        system_prompt = style_profile.to_system_prompt()
        user_prompt = DRAFT_USER_PROMPT_TEMPLATE.format(
            topic=topic,
            key_points=key_points.strip() if key_points else "由作者根据主题自由构思，突出深度与独特见解",
            word_count=word_count,
            target_audience=target_audience,
        )

        draft = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
        )

        if not enable_critique:
            return draft, None

        # ================= 阶段 2：去 AI 味自查与终审润色 =================
        # 将原作者的 StyleProfile 作为上下文注入到审校专家中
        critique_system = f"{system_prompt}\n\n{CRITIQUE_SYSTEM_PROMPT}"
        critique_user = CRITIQUE_USER_PROMPT_TEMPLATE.format(draft_content=draft)

        final_article = self._call_llm(
            system_prompt=critique_system,
            user_prompt=critique_user,
            temperature=max(0.3, self.config.temperature - 0.2), # 审校时温度稍低更精确
        )

        return final_article, draft

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.config.max_tokens,
        }

        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"LLM 接口调用失败 ({resp.status_code}): {resp.text}")

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
