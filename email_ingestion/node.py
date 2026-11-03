from collections.abc import Callable, Mapping
from typing import Any

from gmail_auth import get_gmail_service

from core.config import Settings

from .collector import EmailCollector


def build_email_collector_node(
    gmail_service=None,
    settings: Settings | None = None,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Tạo node collector dùng lại được mà không phụ thuộc trực tiếp LangGraph."""
    collector = EmailCollector(gmail_service or get_gmail_service(), settings)
    collector.initialize()

    def email_collector_node(state: Mapping[str, Any]) -> dict[str, Any]:
        """Chạy collector từ workflow state và trả về phần state cần merge."""
        result = collector.sync(
            force_initial=bool(state.get("force_initial_sync", False))
        )
        return {
            "email_collection": result.to_dict(),
            "collected_email_ids": result.email_ids,
        }

    return email_collector_node
