import os
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx
from src.core.config import LLMConfig, EmbeddingConfig


class ModelProvider:
    """
    统一模型抽象层 (Model Layer)：
    解耦底层大模型与向量服务（支持 DeepSeek, OpenAI, Claude, Qwen, 本地 Ollama 等），
    并提供 Token 预算滑窗与上下文超窗截断保护。
    """

    # 常见模型的上下文窗口限制 (Tokens)
    MODEL_CONTEXT_WINDOWS = {
        "deepseek-chat": 65536,
        "deepseek-reasoner": 65536,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "claude-3-5-sonnet": 200000,
        "qwen2.5-72b": 32768,
        "default": 32768,
    }

    def __init__(self, llm_config: LLMConfig, embedding_config: Optional[EmbeddingConfig] = None):
        self.llm_config = llm_config
        self.embedding_config = embedding_config or EmbeddingConfig()

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        timeout: float = 120.0,
    ) -> str:
        """
        统一发起 Chat 对话请求，内置上下文预算检查
        """
        temp = temperature if temperature is not None else self.llm_config.temperature
        m_tokens = max_tokens if max_tokens is not None else self.llm_config.max_tokens

        # 执行上下文预算裁剪，避免超窗
        safe_sys_prompt, safe_user_prompt = self.fit_context_window(system_prompt, user_prompt)

        url = f"{self.llm_config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.llm_config.model,
            "messages": [
                {"role": "system", "content": safe_sys_prompt},
                {"role": "user", "content": safe_user_prompt},
            ],
            "temperature": temp,
            "max_tokens": m_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Model API 调用失败 [{resp.status_code}]: {resp.text}")
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def get_embeddings(self, texts: List[str], timeout: float = 30.0) -> Optional[List[List[float]]]:
        """获取密集向量 (Dense Embeddings)"""
        api_key = self.embedding_config.api_key or self.llm_config.api_key
        base_url = self.embedding_config.base_url or self.llm_config.base_url
        if not api_key:
            return None

        url = f"{base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.embedding_config.model,
            "input": texts,
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return [item["embedding"] for item in data["data"]]
        except Exception:
            pass
        return None

    def fit_context_window(self, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
        """
        Token 预算与滑窗管理：根据当前模型的最大窗口动态裁剪过长的输入，优先保证 System Prompt
        """
        model_name = self.llm_config.model.lower()
        max_window = self.MODEL_CONTEXT_WINDOWS.get("default", 32768)
        for key, window in self.MODEL_CONTEXT_WINDOWS.items():
            if key in model_name:
                max_window = window
                break

        # 留出 4000 tokens 给生成与安全边际
        available_input_tokens = max(4000, max_window - self.llm_config.max_tokens - 1000)

        sys_tokens = self.estimate_tokens(system_prompt)
        user_tokens = self.estimate_tokens(user_prompt)

        if sys_tokens + user_tokens <= available_input_tokens:
            return system_prompt, user_prompt

        # 如果超预算，先保证 system_prompt（最高占 40%），其余给 user_prompt
        max_sys_tokens = int(available_input_tokens * 0.4)
        if sys_tokens > max_sys_tokens:
            system_prompt = self.truncate_by_tokens(system_prompt, max_sys_tokens)
            sys_tokens = max_sys_tokens

        remaining_user_tokens = available_input_tokens - sys_tokens
        if user_tokens > remaining_user_tokens:
            user_prompt = self.truncate_by_tokens(user_prompt, remaining_user_tokens)

        return system_prompt, user_prompt

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """快速估算中英文字符的 Token 数量 (1个中文字符约 0.6-0.8 token，英文单词约 1.3 token)"""
        chinese_chars = len(re.findall(r"[\u4e00-\u9fa5]", text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 0.8 + other_chars * 0.3) + 10

    @classmethod
    def truncate_by_tokens(cls, text: str, max_tokens: int) -> str:
        """安全截断文本"""
        if cls.estimate_tokens(text) <= max_tokens:
            return text
        # 预估字符数进行切片
        target_char_count = int(max_tokens * 1.5)
        return text[:target_char_count] + "\n...(由于上下文窗口预算限制，已执行安全裁剪)..."
