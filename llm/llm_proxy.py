import json
import re
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


class LLMProxyClient:
    """Client tối thiểu cho proxy tương thích OpenAI Chat Completions."""

    def __init__(self, config: LLMProxyConfig, opener=urlopen):
        """Khởi tạo LLM client từ cấu hình proxy."""
        if not config.base_url or not config.model:
            raise ValueError("LLM_PROXY_BASE_URL and LLM_MODEL are required")
        self.config = config
        self._opener = opener
        self.url = _chat_completions_url(config.base_url)

    def chat_json(self, system_prompt: str, user_payload: dict) -> dict:
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

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            request = Request(self.url, data=body, headers=headers, method="POST")
            try:
                with self._opener(request, timeout=self.config.timeout_seconds) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
                choices = envelope.get("choices") or []
                if not choices:
                    raise LLMResponseError("Proxy response has no choices")
                return _json_from_content(choices[0].get("message", {}).get("content"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = LLMProxyError(f"Proxy HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = LLMProxyError(f"Proxy request failed: {exc}")
            if attempt < self.config.max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        raise last_error or LLMProxyError("Proxy request failed")
