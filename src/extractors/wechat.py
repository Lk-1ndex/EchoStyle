import httpx
import trafilatura
from typing import Optional
from .base import BaseExtractor
from .sanitizer import TextSanitizer


class WeChatExtractor(BaseExtractor):
    """
    微信公众号文章高精提取器：
    优先使用 Jina Reader 高保真服务转换为 Markdown，若网络受限则优雅降级为 Trafilatura 本地正文抽取，
    最后经过 TextSanitizer 深度清洗。
    """

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.last_engine_used: Optional[str] = None

    @staticmethod
    def _is_challenge_page(text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in (
            "warning: this page maybe requiring captcha",
            "secitptpage/template/verify.js",
            "captcha.gtimg.com/tcaptcha.js",
        ))

    def extract(self, url: str) -> str:
        url = url.strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            raise ValueError(f"无效的微信文章 URL: {url}")

        raw_md: Optional[str] = None
        self.last_engine_used = None

        # 策略 1：通过 Jina Reader API 提取 (高保真 Markdown 转译)
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "X-Return-Format": "markdown",
        }
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.get(jina_url, headers=headers)
                if resp.status_code == 200 and len(resp.text.strip()) > 100 and not self._is_challenge_page(resp.text):
                    raw_md = resp.text
                    self.last_engine_used = "jina_reader"
        except Exception:
            # 捕获网络异常，进入本地降级机制
            pass

        # 策略 2：本地降级方案 (Trafilatura 纯净正文抽取)
        if not raw_md:
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    if self._is_challenge_page(downloaded):
                        raise RuntimeError("微信公众号页面要求验证，请提供可访问链接或文章正文。")
                    raw_md = trafilatura.extract(
                        downloaded,
                        output_format="markdown",
                        include_links=False,
                        include_images=False,
                    )
                    if raw_md:
                        self.last_engine_used = "trafilatura"
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"微信文章抓取失败: {str(e)}") from e

        if not raw_md:
            raise RuntimeError(f"未能从链接中提取到有效正文内容: {url}")

        # 策略 3：通过专属 Sanitizer 清洗平台噪点与引导语
        cleaned_md = TextSanitizer.clean(raw_md, repair_linebreaks=False)
        if not cleaned_md:
            raise RuntimeError(f"未能从链接中提取到有效正文内容: {url}")
        return cleaned_md
