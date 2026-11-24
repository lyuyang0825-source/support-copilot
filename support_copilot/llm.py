"""OpenAI-compatible 模型适配层：只润色措辞，不参与事实与决策。

设计边界：
- 默认关闭；只有配置了 ``LLM_BASE_URL``（以及托管服务需要的 ``LLM_API_KEY``）才启用。
- 模型只拿到“用户问题 + 已命中的知识原文 + 规则答案 + 动作”，只被允许改写表达。
- 输出必须通过三道校验：不得新增数字、不得出现承诺词、不得丢掉原文的关键词。
- 任何失败都回退到规则答案，并在 ``notes`` 里留下原因；不会静默换成模型输出。

兼容 OpenAI Chat Completions 协议，因此可以直接接 Qwen、GLM、Llama 的托管推理，
也可以接本地 vLLM / Ollama / llama.cpp server。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .bm25 import tokenize
from .policy import scan_promises, untraceable_numbers

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "support_reply.md"

# 模型不该出现在客服正文里的自我暴露与固定格式
FORBIDDEN_PHRASES = ("作为AI", "作为 AI", "作为人工智能", "依据：", "引用：")
MAX_CHARS = 600


class LLMError(RuntimeError):
    """模型调用或返回格式错误。"""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str = ""
    model: str = DEFAULT_MODEL
    timeout: float = 20.0
    temperature: float = 0.2
    max_tokens: int = 500

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        base = os.getenv("LLM_BASE_URL", "").strip()
        if not base:
            return None
        return cls(
            base_url=base,
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            timeout=float(os.getenv("LLM_TIMEOUT", "20")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
        )

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"


@dataclass
class PolishOutcome:
    text: str | None
    note: str


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@lru_cache(maxsize=1)
def system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "你是客服回复润色器，不是决策者。只改写给定的规则答案，"
        "不得新增事实、数字、时间、金额、条件或承诺；无法安全改写时原样返回。"
    )


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().strip('"“”')


def validate_polish(text: str, source_answer: str, question: str) -> str | None:
    """通过返回 ``None``；不通过返回可读的拒绝原因。"""
    text = _clean(text)
    if not text:
        return "模型返回空内容"
    if len(text) > MAX_CHARS:
        return f"模型输出超过 {MAX_CHARS} 字"
    if any(p in text for p in FORBIDDEN_PHRASES):
        return "模型输出包含系统不该拼接的内容"

    promises = scan_promises(text)
    if promises:
        return f"模型输出含承诺词 {promises}"

    if untraceable_numbers(text, [source_answer, question]):
        return "模型输出含知识原文里没有的数字"

    source_digits = set(re.findall(r"\d+(?:\.\d+)?", source_answer + question))
    output_digits = set(re.findall(r"\d+(?:\.\d+)?", text))
    extra = output_digits - source_digits
    if extra:
        return f"模型输出新增了数字 {sorted(extra)}"

    source_tokens = set(tokenize(source_answer + " " + question))
    output_tokens = set(tokenize(text))
    if not output_tokens:
        return "模型输出没有有效词元"
    overlap = len(source_tokens & output_tokens) / len(output_tokens)
    if overlap < 0.2:
        return f"模型输出与知识原文重合度过低（{overlap:.0%}）"
    return None


def polish_body(
    config: LLMConfig,
    *,
    query: str,
    question: str,
    source_answer: str,
    action: str,
) -> PolishOutcome:
    """让模型改写 ``source_answer``；失败时返回 ``text=None`` 与可读原因。"""
    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "用户问题": query,
                        "知识标题": question,
                        "规则答案": source_answer,
                        "动作": action,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    try:
        data = _post_json(config.endpoint, payload, headers, config.timeout)
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 - 任何失败都必须回退到规则答案
        return PolishOutcome(None, f"调用失败（{type(exc).__name__}）")

    if not isinstance(content, str):
        return PolishOutcome(None, "模型返回格式错误（content 不是字符串）")

    reason = validate_polish(content, source_answer, question)
    if reason:
        return PolishOutcome(None, reason)
    return PolishOutcome(_clean(content), f"llm={config.model} 已润色，事实未改")
