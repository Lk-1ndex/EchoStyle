import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
import tiktoken
from src.core.config import LLMConfig, EmbeddingConfig
from src.core.exceptions import ModelProviderError


class ModelProvider:
    """
    生产级模型抽象层 (Production Model Layer)：
    1. 集成 tiktoken 工业标准 BPE 分词器，进行精准 Token 计费与滑窗截断；
    2. 实现指数退避重试机制 (Exponential Backoff)，自动容错 429、502、503、超时等偶发网络异常；
    3. 支持主备模型热降级 (Fallback Strategy)。
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

        # 初始化标准分词器 (cl100k_base 为 GPT-4/DeepSeek 广泛兼容的分词字典)
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
        """
        发起 Chat 请求，带指数退避重试与主备模型降级
        """
        temp = temperature if temperature is not None else self.llm_config.temperature
        m_tokens = max_tokens if max_tokens is not None else self.llm_config.max_tokens

        # 执行精准 Token 预算管理，避免超窗
        safe_sys_prompt, safe_user_prompt = self.fit_context_window(system_prompt, user_prompt)

        # 优先使用主模型，失败重试耗尽后尝试备用模型
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
                    print(f"[ModelProvider 警告] 网络异常 ({net_err})，第 {attempt + 1}/{max_retries} 次重试，退避等待 {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2.0
                except httpx.HTTPStatusError as http_err:
                    last_err = http_err
                    status = http_err.response.status_code
                    # 针对 429 限流或 5xx 服务器临时故障进行指数退避重试
                    if status in [429, 500, 502, 503, 504]:
                        print(f"[ModelProvider 警告] HTTP {status} 临时错误，第 {attempt + 1}/{max_retries} 次重试，等待 {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= 2.0
                    else:
                        # 401 鉴权或 400 参数错误直接中断重试
                        break
                except Exception as e:
                    last_err = e
                    break

        raise ModelProviderError(f"所有模型及重试策略均已耗尽，调用失败: {str(last_err)}") from last_err

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
        """使用精准 Token 估算管理上下文滑窗"""
        model_name = self.llm_config.model.lower()
        max_window = self.MODEL_CONTEXT_WINDOWS.get("default", 32768)
        for key, window in self.MODEL_CONTEXT_WINDOWS.items():
            if key in model_name:
                max_window = window
                break

        # 预留输出 Token 与安全边界
        available_input_tokens = max(3000, max_window - self.llm_config.max_tokens - 1000)

        sys_tokens = self.count_tokens(system_prompt)
        user_tokens = self.count_tokens(user_prompt)

        if sys_tokens + user_tokens <= available_input_tokens:
            return system_prompt, user_prompt

        # 若超额，保障 System Prompt 优先占比
        max_sys = int(available_input_tokens * 0.45)
        if sys_tokens > max_sys:
            system_prompt = self.truncate_tokens(system_prompt, max_sys)
            sys_tokens = max_sys

        remaining_user = available_input_tokens - sys_tokens
        if user_tokens > remaining_user:
            user_prompt = self.truncate_tokens(user_prompt, remaining_user)

        return system_prompt, user_prompt

    def count_tokens(self, text: str) -> int:
        """使用 tiktoken 精准计算 Token 数量"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text, disallowed_special=()))
            except Exception:
                pass
        # 降级备用
        return int(len(text) * 0.7) + 5

    def truncate_tokens(self, text: str, max_tokens: int) -> str:
        """精准截断文本至指定 Token 阈值以内"""
        if not text:
            return ""
        if self.count_tokens(text) <= max_tokens:
            return text

        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, disallowed_special=())[:max_tokens]
                return self.tokenizer.decode(tokens) + "\n...(由于上下文窗口预算限制，已执行精准裁剪)..."
            except Exception:
                pass

        char_limit = int(max_tokens * 1.4)
        return text[:char_limit] + "\n...(由于上下文窗口预算限制，已执行精准裁剪)..."
