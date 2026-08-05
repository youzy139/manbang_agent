"""LLM 版画像解析器：用 LLM 从自然语言偏好中提取结构化规则（无正则回退）。

当 LLM 不可用（无 api_key）或解析失败时返回 None，调用方应自行处理。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

_LOGGER = logging.getLogger(__name__)


def _read_prompt_or_empty(name: str) -> str:
    p = Path(__file__).resolve().parent / "prompts" / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


_PERSONA_EXTRACTOR_PROMPT = _read_prompt_or_empty("llm_persona_extractor_prompt.txt")


@dataclass(frozen=True)
class LLMPersonaConfig:
    """LLM 画像解析器配置，由 config.json 驱动。"""
    api_key: str = ""
    endpoint: str = ""
    model: str = "deepseek-chat"
    timeout: int = 60


def _build_messages(preferences: list[dict[str, Any]]) -> list[dict[str, str]]:
    """构造 LLM 对话消息：system prompt 外置到 prompts/llm_persona_extractor_prompt.txt，
    user 消息只携带动态偏好文本。"""
    pref_lines: list[str] = []
    for i, p in enumerate(preferences):
        content = p.get("content", "")
        penalty = p.get("penalty_amount", 0)
        cap = p.get("penalty_cap", "null")
        start = p.get("start_time", "")
        end = p.get("end_time", "")
        pref_lines.append("[%d] content: %s" % (i, content))
        pref_lines.append("    penalty: %s cap: %s" % (penalty, cap))
        pref_lines.append("    valid: %s ~ %s" % (start, end))

    preferences_text = "\n".join(pref_lines)

    return [
        {"role": "system", "content": _PERSONA_EXTRACTOR_PROMPT},
        {"role": "user", "content": "司机偏好：\n%s\n\n请按上面要求的格式输出JSON。" % preferences_text},
    ]


def _parse_llm_response(response_text: str) -> dict[str, Any] | None:
    """解析 LLM 返回的 JSON。"""
    text = response_text.strip()
    # 去除可能的 markdown 代码块标记
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    # 去除 JS 风格注释（LLM 可能模仿输出 schema 中的 // 注释）
    text = __import__("re").sub(r'^\s*//.*$', '', text, flags=__import__("re").MULTILINE)
    text = text.strip()
    try:
        result = json.loads(text)
        if not isinstance(result, dict):
            _LOGGER.warning("LLM 返回非 dict: %s", type(result))
            return None
        return result
    except json.JSONDecodeError as e:
        _LOGGER.warning("LLM JSON 解析失败: %s", e)
        return None


def extract_with_llm(
    preferences: list[Any],
    config: LLMPersonaConfig | None = None,
    chat_func: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    token_accumulator: dict[str, int] | None = None,
    custom_messages: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """用 LLM 解析司机偏好，返回结构化规则 dict。

    失败时返回 None（无 api_key / 网络异常 / 响应解析失败）。

    chat_func: 可选，用于替代 requests.post 的调用函数，
               接收 payload dict 返回响应 dict（如 api.model_chat_completion）。
               传入后忽略 config.endpoint / config.api_key / config.timeout，
               由 chat_func 自身管理地址与密钥。
    token_accumulator: 可选，可变 dict，提取成功时写入此次调用的 token 用量。
    custom_messages: 可选，自定义 LLM 消息列表。传入后替代默认的 _build_messages()，
                     由调用方自行构造 prompt，适用于需要更精细提取指令的场景。
    """
    if chat_func is None:
        cfg = config or LLMPersonaConfig()
        if not cfg.api_key:
            _LOGGER.info("LLM API Key 未配置，跳过 LLM 解析")
            return None
        if not cfg.endpoint:
            _LOGGER.info("LLM endpoint 未配置，跳过 LLM 解析")
            return None

    if custom_messages is not None:
        messages = custom_messages
    else:
        # 统一为 dict 格式
        try:
            pref_dicts: list[dict[str, Any]] = []
            for p in preferences:
                if isinstance(p, str):
                    pref_dicts.append({"content": p, "penalty_amount": 0, "penalty_cap": None, "start_time": "", "end_time": ""})
                elif isinstance(p, dict):
                    pref_dicts.append({
                        "content": p.get("content") or p.get("text", ""),
                        "penalty_amount": p.get("penalty_amount", 0),
                        "penalty_cap": p.get("penalty_cap"),
                        "start_time": p.get("start_time", ""),
                        "end_time": p.get("end_time", ""),
                    })
        except Exception as e:
            _LOGGER.warning("偏好预处理失败: %s", e)
            return None

        messages = _build_messages(pref_dicts)
    # 不设 model，让 chat_func 的底层用默认模型（如 config 中的 model_name）
    payload = {
        "messages": messages,
        "enable_thinking": False,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "max_tokens": 8192,
    }
    if config and config.model:
        payload["model"] = config.model

    try:
        if chat_func is not None:
            data = chat_func(payload)
        else:
            cfg = config or LLMPersonaConfig()
            resp = requests.post(
                cfg.endpoint,
                headers={
                    "Authorization": f"Bearer {cfg.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=cfg.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        # Fallback: some reasoning models put JSON in reasoning_content when content is empty.
        if not content:
            reasoning_content = data["choices"][0]["message"].get("reasoning_content", "")
            if reasoning_content:
                content = reasoning_content
        if not content:
            _LOGGER.warning("LLM 返回内容为空，完整响应: %s", json.dumps(data, ensure_ascii=False, default=str)[:2000])
        usage = data.get("usage", {})
        reasoning_tokens = int(usage.get("reasoning_tokens", 0)
            or (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
        if token_accumulator is not None:
            token_accumulator["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            token_accumulator["completion_tokens"] += int(usage.get("completion_tokens", 0))
            token_accumulator["reasoning_tokens"] += reasoning_tokens
            token_accumulator["total_tokens"] += int(usage.get("total_tokens", 0))
        _LOGGER.info(
            "llm_persona_extractor usage prompt=%s completion=%s reasoning=%s total=%s enable_thinking=%s",
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            reasoning_tokens,
            int(usage.get("total_tokens", 0)),
            payload.get("enable_thinking"),
        )
        result = _parse_llm_response(content)
        if result is None:
            _LOGGER.warning("LLM 返回解析失败，原始内容: %s", content[:200])
            _LOGGER.warning("LLM 返回解析失败，完整响应: %s", json.dumps(data, ensure_ascii=False, default=str)[:2000])
        return result
    except requests.RequestException as e:
        _LOGGER.warning("LLM API 调用失败: %s", e)
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        _LOGGER.warning("LLM 响应解析失败: %s", e)
        try:
            _LOGGER.warning("LLM 响应数据: %s", json.dumps(data, ensure_ascii=False, default=str)[:2000])
        except NameError:
            pass
        return None
