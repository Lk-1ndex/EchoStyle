import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
import tiktoken
from src.core.config import LLMConfig, EmbeddingConfig
from src.core.exceptions import ModelProviderError


class ModelProvider:
    """
    生产级模型抽象层 (Production Model Layer)：
    1. 集成 tiktoken 工业标准 BPE 分词器，进行精准 Token 计费与截断；
    2. 实现确定性的 Token 分级预算配比策略 (20% 角色 / 25% 风格指纹 / 30% 记忆范例 / 15% 任务要点 / 10% 审校历史)；
    3. 指数退避重试 (Exponential Backoff) 与主备模型降级。
    """

    MODEL_CONTEXT_WINDOWS = {
        "deepseek-chat": 65536,
        "deepseek-reasoner": 65536,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "claude-3-5-sonnet": 200000,
        "qwen2.5-72b": 32768,
        "default": 32768,
    }

    def __init__(
        self,
        llm_config: LLMConfig,
        embedding_config: Optional[EmbeddingConfig] = None,
        fallback_model: Optional[str] = None,
    ):
        self.llm_config = llm_config
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.fallback_model = fallback_model

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> str:
        temp = temperature if temperature is not None else self.llm_config.temperature
        m_tokens = max_tokens if max_tokens is not None else self.llm_config.max_tokens

        safe_sys_prompt, safe_user_prompt = self.fit_context_window(system_prompt, user_prompt)

        models_to_try = [self.llm_config.model]
        if self.fallback_model and self.fallback_model != self.llm_config.model:
            models_to_try.append(self.fallback_model)

        last_err = None
        for current_model in models_to_try:
            delay = 1.0
            for attempt in range(max_retries):
                try:
                    return self._execute_chat_http(
                        model=current_model,
                        system_prompt=safe_sys_prompt,
                        user_prompt=safe_user_prompt,
                        temperature=temp,
                        max_tokens=m_tokens,
                        json_mode=json_mode,
                        timeout=timeout,
                    )
                except (httpx.TimeoutException, httpx.NetworkError) as net_err:
                    last_err = net_err
                    print(f"[ModelProvider 警告] 网络异常 ({net_err})，第 {attempt + 1}/{max_retries} 次重试，等待 {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2.0
                except httpx.HTTPStatusError as http_err:
                    last_err = http_err
                    status = http_err.response.status_code
                    if status in [429, 500, 502, 503, 504]:
                        print(f"[ModelProvider 警告] HTTP {status} 临时错误，第 {attempt + 1}/{max_retries} 次重试，等待 {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= 2.0
                    else:
                        break
                except Exception as e:
                    last_err = e
                    break

        raise ModelProviderError(f"所有模型及重试策略均已耗尽，调用失败: {str(last_err)}") from last_err

    def assemble_budgeted_prompt(
        self,
        system_persona: str,
        style_dna: str,
        memory_exemplars: List[str],
        user_task: str,
        critique_feedback: str = "",
    ) -> Tuple[str, str]:
        """
        显式 Token 分级预算策略 (Token Budget Allocation Policy)：
        - 角色人设 (System Persona): 20%
        - 风格指纹 (Style DNA): 25%
        - 风格记忆范例 (Memory Exemplars): 30%
        - 任务要求与要点 (User Task): 15%
        - 审校与反思历史 (Critique Feedback): 10%
        """
        model_name = self.llm_config.model.lower()
        max_window = self.MODEL_CONTEXT_WINDOWS.get("default", 32768)
        for key, window in self.MODEL_CONTEXT_WINDOWS.items():
            if key in model_name:
                max_window = window
                break

        available_tokens = max(3000, max_window - self.llm_config.max_tokens - 1000)

        # 各维度硬性预算配比
        quota_persona = int(available_tokens * 0.20)
        quota_style = int(available_tokens * 0.25)
        quota_memory = int(available_tokens * 0.30)
        quota_task = int(available_tokens * 0.15)
        quota_critique = int(available_tokens * 0.10)

        # 1. 裁剪并组装 System 部分
        safe_persona = self.truncate_tokens(system_persona, quota_persona)
        safe_style = self.truncate_tokens(style_dna, quota_style)

        # 2. 裁剪并组装 Memory Few-shot
        joined_memory = ""
        for i, ex in enumerate(memory_exemplars, 1):
            joined_memory += f"\n### 高光范例 {i}：\n> {ex.strip()}\n"
        safe_memory = self.truncate_tokens(joined_memory, quota_memory)

        system_prompt = f"{safe_persona}\n\n{safe_style}"
        if safe_memory.strip():
            system_prompt += f"\n\n## 风格感知检索召回的历史范例\n{safe_memory}"

        # 3. 裁剪并组装 User Task 与 Critique 部分
        safe_task = self.truncate_tokens(user_task, quota_task)
        user_prompt = safe_task
        if critique_feedback:
            safe_critique = self.truncate_tokens(critique_feedback, quota_critique)
            user_prompt += f"\n\n## 上一轮总编辑审校批注（重点反思修正）\n{safe_critique}"

        return system_prompt, user_prompt

    def fit_context_window(self, system_prompt: str, user_prompt: str) -> Tuple[str, str]:
        """通用的保底滑窗检查"""
        model_name = self.llm_config.model.lower()
        max_window = self.MODEL_CONTEXT_WINDOWS.get("default", 32768)
        for key, window in self.MODEL_CONTEXT_WINDOWS.items():
            if key in model_name:
                max_window = window
                break

        available = max(3000, max_window - self.llm_config.max_tokens - 1000)
        sys_tokens = self.count_tokens(system_prompt)
        user_tokens = self.count_tokens(user_prompt)

        if sys_tokens + user_tokens <= available:
            return system_prompt, user_prompt

        max_sys = int(available * 0.5)
        if sys_tokens > max_sys:
            system_prompt = self.truncate_tokens(system_prompt, max_sys)
            sys_tokens = max_sys

        remaining = available - sys_tokens
        if user_tokens > remaining:
            user_prompt = self.truncate_tokens(user_prompt, remaining)

        return system_prompt, user_prompt

    def _execute_chat_http(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        timeout: float,
    ) -> str:
        url = f"{self.llm_config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def get_embeddings(self, texts: List[str], timeout: float = 30.0) -> Optional[List[List[float]]]:
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

    def count_tokens(self, text: str) -> int:
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, disallowed_special=()))
            except Exception:
                pass
        return int(len(text) * 0.7) + 5

    def truncate_tokens(self, text: str, max_tokens: int) -> str:
        if not text:
            return ""
        if self.count_tokens(text) <= max_tokens:
            return text

        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, disallowed_special=())[:max_tokens]
                return self.tokenizer.decode(tokens) + "\n...(已执行分级预算安全裁剪)..."
            except Exception:
                pass

        char_limit = int(max_tokens * 1.4)
        return text[:char_limit] + "\n...(已执行分级预算安全裁剪)..."
