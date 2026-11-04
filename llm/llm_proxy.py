import json
import re
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.observability import LangfuseTracer


class LLMProxyError(RuntimeError):
    pass


class LLMResponseError(LLMProxyError):
    pass


@dataclass(frozen=True)
class LLMProxyConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 90
    max_attempts: int = 3
    json_mode: bool = True
    langfuse_enabled: bool = False
    langfuse_capture_content: bool = False


def _chat_completions_url(base_url: str) -> str:
    """Chuẩn hóa URL endpoint Chat Completions."""
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _json_from_content(content) -> dict:
    """Trích xuất và kiểm tra JSON từ phản hồi LLM."""
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    if not isinstance(content, str):
        raise LLMResponseError("Proxy response message.content is not text")
    content = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.I | re.S)
    if fenced:
        content = fenced.group(1)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Proxy returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMResponseError("Proxy JSON response must be an object")
    return value


def _usage_details(envelope: dict) -> dict[str, int] | None:
    """Chuẩn hóa usage OpenAI-compatible thành bucket Langfuse độc quyền."""
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    except (TypeError, ValueError):
        return None
    return {"input": input_tokens, "output": output_tokens, "total": total_tokens}


def _cost_details(envelope: dict) -> dict[str, float] | None:
    """Lấy chi phí thực tế nếu proxy trả về ở top-level hoặc trong usage."""
    provider_cost = envelope.get("cost")
    if provider_cost is None and isinstance(envelope.get("usage"), dict):
        provider_cost = envelope["usage"].get("cost")
    if provider_cost is None:
        return None
    try:
        return {"total": float(provider_cost)}
    except (TypeError, ValueError):
        return None


def _telemetry_details(envelope: dict) -> dict:
    """Trích usage và cost độc lập cho từng lần gọi proxy."""
    update: dict = {}
    usage = _usage_details(envelope)
    if usage:
        update["usage_details"] = usage
    costs = _cost_details(envelope)
    if costs:
        update["cost_details"] = costs
    return update


class LLMProxyClient:
    """Client tối thiểu cho proxy tương thích OpenAI Chat Completions."""

    def __init__(self, config: LLMProxyConfig, opener=urlopen):
        """Khởi tạo LLM client từ cấu hình proxy."""
        if not config.base_url or not config.model:
            raise ValueError("LLM_PROXY_BASE_URL and LLM_MODEL are required")
        self.config = config
        self._opener = opener
        self.url = _chat_completions_url(config.base_url)
        self.tracer = LangfuseTracer(config.langfuse_enabled)

    def chat_json(
        self,
        system_prompt: str,
        user_payload: dict,
        operation_name: str = "llm.chat_json",
    ) -> dict:
        """Gửi prompt đến LLM proxy và trả về JSON đã kiểm tra."""
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        trace_input = (
            payload["messages"]
            if self.config.langfuse_capture_content
            else {
                "system_prompt_chars": len(system_prompt),
                "user_payload_chars": len(payload["messages"][1]["content"]),
            }
        )
        with self.tracer.observation(
            f"{operation_name}.request",
            as_type="span",
            input=trace_input,
            metadata={"json_mode": str(self.config.json_mode).lower()},
        ) as logical_request:
            last_error: Exception | None = None
            for attempt in range(1, self.config.max_attempts + 1):
                request = Request(self.url, data=body, headers=headers, method="POST")
                with self.tracer.observation(
                    operation_name,
                    as_type="generation",
                    input=trace_input,
                    model=self.config.model,
                    metadata={
                        "attempt": str(attempt),
                        "max_attempts": str(self.config.max_attempts),
                        "json_mode": str(self.config.json_mode).lower(),
                    },
                ) as generation:
                    try:
                        with self._opener(
                            request, timeout=self.config.timeout_seconds
                        ) as response:
                            envelope = json.loads(response.read().decode("utf-8"))
                        choices = envelope.get("choices") or []
                        if not choices:
                            raise LLMResponseError("Proxy response has no choices")
                        raw_content = choices[0].get("message", {}).get("content")
                        result = _json_from_content(raw_content)
                        update: dict = {
                            "output": result
                            if self.config.langfuse_capture_content
                            else {"response_chars": len(str(raw_content))},
                            "metadata": {"outcome": "success"},
                        }
                        update.update(_telemetry_details(envelope))
                        generation.update(**update)
                        logical_request.update(
                            output={"successful_attempt": attempt},
                            metadata={"attempts": str(attempt)},
                        )
                        return result
                    except HTTPError as exc:
                        raw_detail = exc.read().decode("utf-8", errors="replace")
                        detail = raw_detail[:1000]
                        last_error = LLMProxyError(
                            f"Proxy HTTP {exc.code}: {detail}"
                        )
                        error_update: dict = {
                            "level": "ERROR",
                            "status_message": f"Proxy HTTP {exc.code}",
                            "metadata": {
                                "outcome": "error",
                                "retryable": str(
                                    exc.code
                                    in {408, 409, 429, 500, 502, 503, 504}
                                ).lower(),
                            },
                        }
                        try:
                            error_envelope = json.loads(raw_detail)
                        except json.JSONDecodeError:
                            error_envelope = None
                        error_telemetry: dict = {}
                        if isinstance(error_envelope, dict):
                            error_telemetry = _telemetry_details(error_envelope)
                            error_update.update(error_telemetry)
                        error_update["metadata"].update(
                            {
                                "usage_reported": str(
                                    "usage_details" in error_telemetry
                                ).lower(),
                                "cost_reported": str(
                                    "cost_details" in error_telemetry
                                ).lower(),
                            }
                        )
                        generation.update(**error_update)
                        if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                            logical_request.update(
                                level="ERROR", status_message=f"Proxy HTTP {exc.code}"
                            )
                            raise last_error
                    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                        last_error = LLMProxyError(f"Proxy request failed: {exc}")
                        generation.update(
                            level="ERROR",
                            status_message=f"Proxy request failed: {type(exc).__name__}",
                            metadata={
                                "outcome": "error",
                                "retryable": "true",
                                "usage_reported": "false",
                                "cost_reported": "false",
                            },
                        )
                if attempt < self.config.max_attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
            logical_request.update(
                level="ERROR",
                status_message=str(last_error or "Proxy request failed"),
                metadata={"attempts": str(self.config.max_attempts)},
            )
            raise last_error or LLMProxyError("Proxy request failed")
