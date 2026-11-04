"""Tracing Langfuse an toàn cho agent, LLM, embedding và read tool."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator


logger = logging.getLogger(__name__)


class _SafeObservation:
    """Bọc observation để lỗi telemetry không làm hỏng luồng nghiệp vụ."""

    def __init__(self, observation: Any = None):
        self._observation = observation

    def update(self, **kwargs: Any) -> None:
        """Cập nhật observation nếu SDK đang hoạt động."""
        if self._observation is None:
            return
        try:
            self._observation.update(**kwargs)
        except Exception as exc:  # telemetry phải luôn fail-open
            logger.warning("Langfuse observation update failed: %s", exc)


class LangfuseTracer:
    """Adapter nhỏ dùng Langfuse SDK v5 nhưng vẫn chạy khi tracing bị tắt."""

    def __init__(self, enabled: bool = False):
        """Khởi tạo client từ các biến LANGFUSE_* chuẩn của SDK."""
        self.enabled = bool(enabled)
        self._client = None
        if not self.enabled:
            return
        try:
            from langfuse import get_client

            self._client = get_client()
        except (ImportError, RuntimeError, ValueError) as exc:
            logger.warning("Langfuse disabled because initialization failed: %s", exc)
            self.enabled = False

    @contextmanager
    def observation(
        self,
        name: str,
        *,
        as_type: str = "span",
        input: Any = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[_SafeObservation]:
        """Tạo observation hiện hành để các span con tự nối đúng trace."""
        if not self.enabled or self._client is None:
            yield _SafeObservation()
            return
        kwargs: dict[str, Any] = {
            "name": name,
            "as_type": as_type,
            "input": input,
            "metadata": metadata,
        }
        if model:
            kwargs["model"] = model
        try:
            context = self._client.start_as_current_observation(**kwargs)
        except Exception as exc:
            logger.warning("Langfuse observation start failed: %s", exc)
            yield _SafeObservation()
            return
        with context as observation:
            yield _SafeObservation(observation)

    def flush(self) -> None:
        """Đẩy batch telemetry còn lại; chủ yếu dùng khi worker sắp dừng."""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)
