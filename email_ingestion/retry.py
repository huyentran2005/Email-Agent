import json
import logging
import random
import time
from collections.abc import Callable
from typing import Any

from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

RETRIABLE_STATUSES = {429, 500, 502, 503, 504}
RETRIABLE_REASONS = {"rateLimitExceeded", "userRateLimitExceeded"}


def _error_reasons(error: HttpError) -> set[str]:
    """Trích xuất nguyên nhân lỗi Gmail có cấu trúc từ phản hồi HttpError."""
    try:
        payload = json.loads(error.content.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return set()

    details = payload.get("error", {}).get("errors", [])
    return {
        detail.get("reason")
        for detail in details
        if isinstance(detail, dict) and detail.get("reason")
    }


def is_retriable(error: HttpError) -> bool:
    """Kiểm tra phản hồi có thuộc nhóm lỗi Gmail cho phép thử lại hay không."""
    status = int(getattr(error.resp, "status", 0))
    return status in RETRIABLE_STATUSES or (
        status == 403 and bool(_error_reasons(error) & RETRIABLE_REASONS)
    )


class GmailRequestExecutor:
    """Giới hạn request theo quota và thử lại khi gặp lỗi tạm thời."""

    def __init__(
        self,
        quota_units_per_minute: float,
        max_attempts: int,
        base_delay_seconds: float,
        max_delay_seconds: float,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[float, float], float] = random.uniform,
    ):
        """Khởi tạo bộ thực thi request với quota và retry."""
        if quota_units_per_minute <= 0:
            raise ValueError("quota_units_per_minute must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.units_per_second = quota_units_per_minute / 60
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self._sleep = sleep
        self._monotonic = monotonic
        self._jitter = jitter
        self._next_request_at = 0.0

    def execute(self, request: Any, quota_cost: int, operation: str) -> Any:
        """Thực thi request Google API với giới hạn tốc độ và backoff hữu hạn."""
        for attempt in range(1, self.max_attempts + 1):
            self._throttle(quota_cost)
            try:
                return request.execute()
            except HttpError as error:
                if attempt == self.max_attempts or not is_retriable(error):
                    raise

                delay = min(
                    self.max_delay_seconds,
                    self.base_delay_seconds * (2 ** (attempt - 1)),
                )
                delay += self._jitter(0, min(1.0, delay * 0.1))
                logger.warning(
                    "Gmail %s hit a temporary API limit; retrying in %.1fs "
                    "(attempt %d/%d)",
                    operation,
                    delay,
                    attempt + 1,
                    self.max_attempts,
                )
                self._sleep(delay)

        raise RuntimeError("unreachable")

    def _throttle(self, quota_cost: int) -> None:
        """Chờ khi cần để không vượt quá quota request."""
        now = self._monotonic()
        wait_seconds = max(0.0, self._next_request_at - now)
        if wait_seconds:
            self._sleep(wait_seconds)
            now = self._monotonic()
        self._next_request_at = max(now, self._next_request_at) + (
            quota_cost / self.units_per_second
        )
