"""Direct streaming LLM transport used by the actionable-review Gate."""

import time
from pathlib import Path
from urllib.parse import urlsplit
from openai import OpenAI
from .llm import Completion
from .io import append
from .review_runtime import normalize_review_output

REVISION = "actionable-review-pure-high-routes11110-v2"


class PureLLMGate:
    """No agent, no tools, no SDK retries; formatting retries remain caller-owned."""

    def __init__(self, original, out, client_factory=OpenAI):
        self.api_key, self.base_url = original.api_key, original.base_url
        self.model, self.effort = original.model, original.effort
        self.timeout = original.timeout
        self.out = Path(out)
        self.client_factory = client_factory
        self.request_count = 0

    def complete(self, prompt, json_output=False):
        started = time.time()
        self.request_count += 1
        request_id = "pure-gate-%06d" % self.request_count
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        kwargs = {
            "model": self.model,
            "messages": messages,
            "reasoning_effort": self.effort,
            "extra_body": {"thinking": {"type": "enabled"}},
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 128000,
        }
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        parsed = urlsplit(self.base_url)
        record = {
            "epoch": started,
            "kind": "model.request.status",
            "requestId": request_id,
            "attempt": 1,
            "transport": "pure_llm_single_call",
            "model": self.model,
            "reasoning_effort": self.effort,
            "endpoint": parsed.scheme
            + "://"
            + (parsed.hostname or "")
            + (":" + str(parsed.port) if parsed.port else "")
            + parsed.path,
            "roles": [m["role"] for m in messages],
            "tools": [],
            "sdk_max_retries": 0,
            "actual_model_request_count": 1,
            "status": "model_request_started",
            "timeout_seconds": self.timeout,
        }
        log = self.out / "pure-llm-requests.jsonl"
        append(log, record)
        content, reasoning, usage = [], [], {}
        finish, response_id, first_token = None, None, None
        try:
            with self.client_factory(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            ) as client:
                with client.chat.completions.create(**kwargs) as stream:
                    for chunk in stream:
                        response_id = chunk.id or response_id
                        if chunk.usage:
                            usage = chunk.usage.model_dump()
                        if not chunk.choices:
                            continue
                        if len(chunk.choices) != 1 or chunk.choices[0].index != 0:
                            raise ValueError("Expected one Gate completion")
                        choice = chunk.choices[0]
                        delta = choice.delta
                        if delta.content:
                            content.append(delta.content)
                        thought = getattr(delta, "reasoning_content", "") or ""
                        if thought:
                            reasoning.append(thought)
                        if first_token is None and (delta.content or thought):
                            first_token = time.time() - started
                        if choice.finish_reason is not None:
                            finish = choice.finish_reason
            if finish is None:
                raise ValueError("Gate stream ended without a finish reason")
            if not usage:
                raise ValueError("Gate stream did not return usage")
            elapsed = time.time() - started
            append(
                log,
                {
                    **record,
                    "epoch": time.time(),
                    "status": "model_request_completed",
                    "statusCode": 200,
                    "response_id": response_id,
                    "usage": usage,
                    "finish_reason": finish,
                    "elapsed_seconds": elapsed,
                    "first_token_seconds": first_token,
                },
            )
            raw_value = "".join(content)
            value, normalization = (
                normalize_review_output(raw_value)
                if json_output
                else (raw_value, "not_json")
            )
            append(
                self.out / "pure-llm-output-normalization.jsonl",
                {
                    "epoch": time.time(),
                    "requestId": request_id,
                    "raw_output": raw_value,
                    "normalized_output": value,
                    "normalization": normalization,
                    "choice_inferred": False,
                    "review_protocol_revision": REVISION,
                },
            )
            return Completion(value, usage, "".join(reasoning), elapsed, finish)
        except Exception as exc:
            provider_code = None
            try:
                body = exc.response.json()
                provider_code = str((body.get("error") or {}).get("code"))
            except Exception:
                pass
            append(
                log,
                {
                    **record,
                    "epoch": time.time(),
                    "status": "model_request_failed",
                    "statusCode": getattr(exc, "status_code", None),
                    "providerCode": provider_code,
                    "error_type": type(exc).__name__,
                    "elapsed_seconds": time.time() - started,
                },
            )
            raise
