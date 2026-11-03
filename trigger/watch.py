import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from email_ingestion.retry import GmailRequestExecutor

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]


def resolve_project_id(settings: Settings) -> str:
    """Xác định Google Cloud project từ biến môi trường hoặc file OAuth client."""
    if settings.google_cloud_project_id:
        return settings.google_cloud_project_id

    credentials_path = Path(
        os.getenv("GMAIL_CREDENTIALS_PATH", BASE_DIR / "credentials.json")
    )
    try:
        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
        client_config = payload.get("installed") or payload.get("web") or {}
        project_id = client_config.get("project_id")
    except (OSError, json.JSONDecodeError):
        project_id = None

    if not project_id:
        raise RuntimeError(
            "Set GOOGLE_CLOUD_PROJECT in .env or provide project_id in "
            "credentials.json"
        )
    return project_id


def _request_executor(settings: Settings) -> GmailRequestExecutor:
    """Tạo bộ thực thi Gmail API có quota và retry."""
    return GmailRequestExecutor(
        quota_units_per_minute=settings.gmail_quota_units_per_minute,
        max_attempts=settings.gmail_retry_max_attempts,
        base_delay_seconds=settings.gmail_retry_base_delay_seconds,
        max_delay_seconds=settings.gmail_retry_max_delay_seconds,
    )


def register_gmail_watch(
    gmail_service,
    settings: Settings,
    executor: GmailRequestExecutor | None = None,
) -> dict[str, Any]:
    """Đăng ký hoặc gia hạn Gmail push notification cho nhãn Inbox."""
    project_id = resolve_project_id(settings)
    topic_name = f"projects/{project_id}/topics/{settings.gmail_pubsub_topic}"
    request = gmail_service.users().watch(
        userId="me",
        body={
            "topicName": topic_name,
            "labelIds": ["INBOX"],
            "labelFilterBehavior": "INCLUDE",
        },
    )
    response = (executor or _request_executor(settings)).execute(
        request,
        quota_cost=100,
        operation="users.watch",
    )
    expiration = response.get("expiration")
    expiration_text = "unknown"
    if expiration:
        expiration_text = datetime.fromtimestamp(
            int(expiration) / 1000, tz=timezone.utc
        ).isoformat()
    logger.info("Gmail watch active until %s", expiration_text)
    return response


def renew_watch_forever(
    settings: Settings,
    stop_event: threading.Event,
) -> None:
    """Gia hạn Gmail Watch hằng ngày đến khi tiến trình subscriber dừng."""
    from gmail_auth import get_gmail_service

    while not stop_event.wait(settings.gmail_watch_renewal_seconds):
        try:
            register_gmail_watch(get_gmail_service(), settings)
        except Exception:
            logger.exception("Could not renew Gmail watch; will retry next cycle")
