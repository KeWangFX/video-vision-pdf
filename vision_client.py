"""
多提供方视觉模型统一客户端。

支持的提供方（provider）：
  openai      — OpenAI 官方（GPT-4o / GPT-4o-mini 等），需 OPENAI_API_KEY
  gemini      — Google Gemini（gemini-2.5-flash 等），需 GEMINI_API_KEY
  claude      — Anthropic Claude（claude-sonnet-4-20250514 等），需 ANTHROPIC_API_KEY
  ollama      — 本地 Ollama（qwen3-vl / llava 等），免费无需密钥
  compatible  — 任意 OpenAI 兼容接口（vLLM / LM Studio 等）

所有提供方统一返回 (text: str) 的分析结果。
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Provider registry: { id: (display_name, default_models, needs_key, env_key) }
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "name": "OpenAI",
        "models": ("gpt-4o", "gpt-4o-mini", "o4-mini"),
        "env_key": "OPENAI_API_KEY",
    },
    "gemini": {
        "name": "Google Gemini",
        "models": ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"),
        "env_key": "GEMINI_API_KEY",
    },
    "claude": {
        "name": "Anthropic Claude",
        "models": ("claude-sonnet-4-20250514", "claude-sonnet-4-20250514"),
        "env_key": "ANTHROPIC_API_KEY",
    },
    "ollama": {
        "name": "Ollama（本地免费）",
        "models": ("qwen3-vl:30b", "qwen2.5-vl:7b", "llava:latest",
                   "minicpm-v:latest", "gemma3:4b"),
        "env_key": None,
    },
    "compatible": {
        "name": "OpenAI 兼容接口",
        "models": ("自定义模型名",),
        "env_key": None,
    },
}

DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434/v1"


def provider_ids() -> list[str]:
    return list(PROVIDERS.keys())


def provider_display(pid: str) -> str:
    return PROVIDERS.get(pid, {}).get("name", pid)


def provider_models(pid: str) -> tuple[str, ...]:
    return PROVIDERS.get(pid, {}).get("models", ())


def provider_env_key(pid: str) -> str | None:
    return PROVIDERS.get(pid, {}).get("env_key")


# ---------------------------------------------------------------------------
# Unified chat-with-vision interface
# ---------------------------------------------------------------------------

def _b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def _normalize_base(url: str | None) -> str:
    u = (url or "").strip() or os.environ.get("LLM_BASE_URL", "").strip()
    if not u:
        u = DEFAULT_OLLAMA_BASE
    u = u.rstrip("/")
    if not u.endswith("/v1"):
        u += "/v1"
    return u


class VisionClient:
    """统一视觉分析客户端，封装各提供方差异。"""

    def __init__(self, provider: str, api_key: str = "", api_base: str = ""):
        self.provider = provider.strip().lower()
        self.api_key = api_key.strip()
        self.api_base = api_base.strip()

        if not self.api_key:
            env = provider_env_key(self.provider)
            if env:
                self.api_key = os.environ.get(env, "").strip()

        self._openai_client: Any = None
        self._anthropic_client: Any = None

    # ----- public API -----

    def chat_vision(
        self,
        model: str,
        text_prompt: str,
        image_paths: list[Path],
        *,
        system: str | None = None,
        max_tokens: int = 1200,
    ) -> str:
        """发送带图片的 prompt，返回模型文本回复。"""
        p = self.provider
        if p == "gemini":
            return self._gemini_call(model, text_prompt, image_paths,
                                     system=system, max_tokens=max_tokens)
        if p == "claude":
            return self._claude_call(model, text_prompt, image_paths,
                                     system=system, max_tokens=max_tokens)
        return self._openai_call(model, text_prompt, image_paths,
                                 system=system, max_tokens=max_tokens)

    def chat_text(
        self,
        model: str,
        text_prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1200,
    ) -> str:
        """纯文本调用（用于总结）。"""
        p = self.provider
        if p == "gemini":
            return self._gemini_call(model, text_prompt, [],
                                     system=system, max_tokens=max_tokens)
        if p == "claude":
            return self._claude_call(model, text_prompt, [],
                                     system=system, max_tokens=max_tokens)
        return self._openai_call(model, text_prompt, [],
                                 system=system, max_tokens=max_tokens)

    # ----- OpenAI / Ollama / Compatible -----

    def _get_openai(self) -> Any:
        if self._openai_client:
            return self._openai_client
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("请安装: pip install openai") from e

        p = self.provider
        if p in ("ollama", "compatible"):
            base = _normalize_base(self.api_base)
            key = self.api_key or os.environ.get("LLM_API_KEY", "").strip() or "ollama"
            self._openai_client = OpenAI(
                api_key=key, base_url=base, max_retries=0, timeout=600.0)
        else:
            if not self.api_key:
                raise RuntimeError(
                    f"请设置 {provider_env_key(p) or 'API Key'} 或在界面填入密钥。")
            self._openai_client = OpenAI(
                api_key=self.api_key, max_retries=0, timeout=180.0)
        return self._openai_client

    def _openai_call(
        self, model: str, text: str, images: list[Path],
        *, system: str | None, max_tokens: int,
    ) -> str:
        client = self._get_openai()
        content: list[dict] = [{"type": "text", "text": text}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_b64(img)}"},
            })
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content if images else text})

        rsp = self._retry(
            lambda: client.chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens),
            label=f"openai/{model}",
        )
        return (rsp.choices[0].message.content or "").strip()

    # ----- Google Gemini -----

    def _gemini_call(
        self, model: str, text: str, images: list[Path],
        *, system: str | None, max_tokens: int,
    ) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise RuntimeError("请安装: pip install google-genai") from e

        if not self.api_key:
            raise RuntimeError("请设置 GEMINI_API_KEY 或在界面填入密钥。")

        client = genai.Client(api_key=self.api_key)

        parts: list[Any] = []
        for img in images:
            data = img.read_bytes()
            parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
        parts.append(text)

        config = types.GenerateContentConfig(max_output_tokens=max_tokens)
        if system:
            config.system_instruction = system

        rsp = self._retry(
            lambda: client.models.generate_content(
                model=model, contents=parts, config=config),
            label=f"gemini/{model}",
        )
        return (rsp.text or "").strip()

    # ----- Anthropic Claude -----

    def _claude_call(
        self, model: str, text: str, images: list[Path],
        *, system: str | None, max_tokens: int,
    ) -> str:
        if self._anthropic_client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError("请安装: pip install anthropic") from e
            if not self.api_key:
                raise RuntimeError("请设置 ANTHROPIC_API_KEY 或在界面填入密钥。")
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)

        content: list[dict] = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _b64(img),
                },
            })
        content.append({"type": "text", "text": text})

        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        if system:
            kwargs["system"] = system

        rsp = self._retry(
            lambda: self._anthropic_client.messages.create(**kwargs),
            label=f"claude/{model}",
        )
        parts = rsp.content
        texts = [b.text for b in parts if getattr(b, "type", "") == "text"]
        return "\n".join(texts).strip()

    # ----- Retry wrapper -----

    @staticmethod
    def _retry(fn, *, label: str = "api", max_attempts: int = 6) -> Any:
        last: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as e:
                last = e
                s = str(e).lower()
                code = getattr(e, "status_code", 0) or 0
                if "insufficient_quota" in s or "exceeded your current quota" in s:
                    raise RuntimeError(
                        f"模型服务额度不足（{label}），请检查账户余额或更换密钥。"
                    ) from e
                if code == 429 or "rate" in s:
                    wait = min(60.0, 2.0 ** attempt + 0.5 * attempt)
                    logging.warning(
                        "%s 速率限制，%.1fs 后重试（%d/%d）…",
                        label, wait, attempt + 1, max_attempts)
                    time.sleep(wait)
                    continue
                raise
        if last:
            raise last
        raise RuntimeError(f"{label} 调用重试次数用尽")
