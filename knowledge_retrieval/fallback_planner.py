"""Router rule-based dùng khi LLM tool planner không khả dụng."""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo


LOCAL_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_EMAIL_ID_PATTERN = re.compile(r"\bemail\s*#?\s*(\d+)\b", re.IGNORECASE)
_THREAD_ID_PATTERN = re.compile(r"\bthread\s*#?\s*(\d+)\b", re.IGNORECASE)
_DATE_RANGE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?"
    r"\s*(?:-|–|—|đến|den|tới|toi|to)\s*"
    r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?(?!\d)",
    re.IGNORECASE,
)
_THREAD_DETAIL_MARKERS = (
    "chuoi trao doi",
    "chuoi email",
    "chuoi mail",
    "toan bo trao doi",
    "lich su trao doi",
    "tom tat trao doi",
    "hoi thoai",
    "conversation",
    "thread",
    "luong email",
    "luong mail",
)
_EMAIL_DETAIL_MARKERS = (
    "doc chi tiet",
    "doc email",
    "doc mail",
    "noi dung day du",
    "toan bo noi dung",
    "chi tiet email",
    "chi tiet mail",
    "email noi gi",
    "mail noi gi",
    "mo email",
    "mo mail",
    "tom tat email",
    "tom tat mail",
    "tom tat noi dung",
)


def infer_detail_scope(query: str) -> str:
    """Suy ra scope chỉ cho đường fallback khi LLM planner không khả dụng."""
    normalized = unicodedata.normalize("NFKD", query.casefold()).replace("đ", "d")
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    normalized = " ".join(normalized.split())
    if any(marker in normalized for marker in _THREAD_DETAIL_MARKERS):
        return "thread"
    if any(marker in normalized for marker in _EMAIL_DETAIL_MARKERS):
        return "email"
    return "none"


def extract_date_range(query: str, today: date) -> tuple[date, date] | None:
    """Rút khoảng dd/mm[-yyyy] để fallback vẫn lọc đúng khi LLM lỗi."""
    match = _DATE_RANGE_PATTERN.search(query)
    if not match:
        return None
    start_day, start_month, start_year, end_day, end_month, end_year = match.groups()

    def full_year(raw: str | None, fallback: int) -> int:
        if raw is None:
            return fallback
        year = int(raw)
        return year + 2000 if year < 100 else year

    try:
        start_year_value = full_year(start_year, today.year)
        start = date(start_year_value, int(start_month), int(start_day))
        end_year_value = full_year(end_year, start_year_value)
        end = date(end_year_value, int(end_month), int(end_day))
        if end_year is None and end < start:
            end = date(start_year_value + 1, int(end_month), int(end_day))
    except ValueError:
        return None
    return start, end


def _search_query_without_dates(query: str) -> str:
    """Bỏ khoảng ngày khỏi semantic query vì ngày được truyền bằng filter riêng."""
    search_query = _DATE_RANGE_PATTERN.sub(" ", query)
    return re.sub(
        r"\b(?:từ|tu)(?:\s+(?:ngày|ngay))?\s*$",
        "",
        search_query,
        flags=re.IGNORECASE,
    ).strip() or query


def plan_read_tools(
    query: str, max_calls: int = 3, semantic_enabled: bool = False
) -> list[dict[str, Any]]:
    """Chọn tối đa vài read tool bằng quy tắc khi LLM planner không chạy được."""
    normalized = query.casefold()
    today = datetime.now(LOCAL_TIMEZONE).date()
    extracted_range = extract_date_range(query, today)
    search_query = _search_query_without_dates(query)
    date_arguments = (
        {
            "date_from": extracted_range[0].isoformat(),
            "date_to": extracted_range[1].isoformat(),
        }
        if extracted_range
        else {}
    )
    calls: list[dict[str, Any]] = []

    email_id = _EMAIL_ID_PATTERN.search(query)
    thread_id = _THREAD_ID_PATTERN.search(query)
    addresses = _EMAIL_PATTERN.findall(query)
    if email_id:
        calls.append(
            {"name": "get_email", "arguments": {"email_id": int(email_id.group(1))}}
        )
    if thread_id:
        calls.append(
            {
                "name": "get_thread",
                "arguments": {"thread_id": int(thread_id.group(1))},
            }
        )
    if addresses:
        calls.append(
            {
                "name": "get_sender_history",
                "arguments": {"email_address": addresses[0], "limit": 20},
            }
        )

    if any(word in normalized for word in ("deadline", "hạn", "quá hạn", "đến hạn")):
        calls.append(
            {
                "name": "get_deadlines",
                "arguments": {
                    "date_from": today.isoformat(),
                    "date_to": (today + timedelta(days=7)).isoformat(),
                    "include_overdue": True,
                },
            }
        )
    if any(word in normalized for word in ("họp", "cuộc họp", "meeting", "lịch hẹn")):
        calls.append(
            {
                "name": "get_meetings",
                "arguments": {
                    "date_from": today.isoformat(),
                    "date_to": (today + timedelta(days=30)).isoformat(),
                },
            }
        )
    if any(word in normalized for word in ("việc", "yêu cầu", "đang mở", "chưa làm")):
        calls.append({"name": "get_open_items", "arguments": {"limit": 30}})
    if any(word in normalized for word in ("hôm nay", "daily brief", "email mới")):
        calls.append(
            {"name": "get_daily_brief", "arguments": {"day": today.isoformat()}}
        )

    explicit_lookup = bool(email_id or thread_id or addresses)
    if semantic_enabled and not explicit_lookup:
        calls.append(
            {
                "name": "hybrid_search_subjects",
                "arguments": {"query": search_query, "limit": 12, **date_arguments},
            }
        )
    calls.append(
        {
            "name": "search_emails",
            "arguments": {
                "query": search_query,
                "search_terms": [search_query],
                "limit": 12,
                **date_arguments,
            },
        }
    )

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        if call["name"] in seen:
            continue
        seen.add(call["name"])
        unique.append(call)
        if len(unique) >= max(1, min(max_calls, 5)):
            break
    return unique
