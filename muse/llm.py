"""Direct completions with caller-controlled formatting retries; task and gate never issue concurrent model calls."""

import json, time
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class Completion:
    text: str
    usage: dict
    reasoning: str = ""
    elapsed_seconds: float = 0
    finish_reason: str = "stop"


class DirectLLM:
    def __init__(
        self, api_key, base_url, model="glm-5.3-flash", effort="high", timeout=90
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.effort = effort
        self.timeout = timeout

    def complete(self, prompt, json_output=False):
        start = time.time()
        kwargs = {
            "model": self.model,
            "messages": (
                prompt
                if isinstance(prompt, list)
                else [{"role": "user", "content": prompt}]
            ),
            "reasoning_effort": self.effort,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        with OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
        ) as client:
            r = client.chat.completions.create(**kwargs)
        if len(r.choices) != 1:
            raise ValueError("Direct completion did not finish normally")
        m = r.choices[0].message
        return Completion(
            m.content,
            r.usage.model_dump() if r.usage else {},
            getattr(m, "reasoning_content", "") or "",
            time.time() - start,
            r.choices[0].finish_reason,
        )


def cost_cny(usage):
    prompt = usage.get("prompt_tokens", usage.get("inputTokens", 0)) or 0
    output = usage.get("completion_tokens", usage.get("outputTokens", 0)) or 0
    details = usage.get("prompt_tokens_details") or {}
    cache = details.get("cached_tokens", usage.get("cacheReadTokens", 0)) or 0
    if not 0 <= cache <= prompt:
        raise ValueError("Invalid token usage")
    return ((prompt - cache) * 0.8 + cache * 0.23 + output * 2.8) / 1e6
