"""Router rule-based dùng khi LLM tool planner không khả dụng."""

from __future__ import annotations

import re
import unicodedata
from calendar import monthrange
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo


LOCAL_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_EMAIL_ID_PATTERN = re.compile(r"\bemail\s*#?\s*(\d+)\b", re.IGNORECASE)
_THREAD_ID_PATTERN = re.compile(r"\bthread\s*#?\s*(\d+)\b", re.IGNORECASE)
_COMPOUND_SEPARATOR_PATTERN = re.compile(
    r"\s+(?:và|va|đồng thời|dong thoi|cùng với|cung voi)\s+|[;\n]+",
    re.IGNORECASE,
)
_DATE_RANGE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?"
    r"\s*(?:-|–|—|đến|den|tới|toi|to)\s*"
    r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?(?!\d)",
    re.IGNORECASE,
)
_MONTH_REFERENCE_PATTERN = re.compile(
    r"\b(?:tháng|thang)\s+(\d{1,2})"
    r"(?:\s*(?:/|-|năm|nam)\s*(\d{2,4}))?\b",
    re.IGNORECASE,
)
_DATE_REFERENCE_PATTERN = re.compile(
    r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)"
    r"|(?<!\d)\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?(?!\d)"
    r"|\b(?:hom nay|hom qua|hom kia|ngay mai)\b"
    r"|\b(?:tuan|thang|nam)\s+(?:nay|truoc|sau|qua|toi|den|\d{1,4})\b"
    r"|\b\d+\s*(?:ngay|tuan|thang|nam)\s*(?:qua|gan day|truoc|sau|toi|den)\b",
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


def _normalize_text(value: str) -> str:
    """Chuẩn hóa tiếng Việt để các rule fallback không phụ thuộc dấu."""
    normalized = unicodedata.normalize("NFKD", value.casefold()).replace("đ", "d")
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(normalized.split())


def has_date_reference(query: str) -> bool:
    """Chỉ cho phép date filter khi người dùng thực sự nêu mốc thời gian."""
    return bool(_DATE_REFERENCE_PATTERN.search(_normalize_text(query)))


def infer_detail_scope(query: str) -> str:
    """Suy ra scope chỉ cho đường fallback khi LLM planner không khả dụng."""
    normalized = _normalize_text(query)
    if any(marker in normalized for marker in _THREAD_DETAIL_MARKERS):
        return "thread"
    if any(marker in normalized for marker in _EMAIL_DETAIL_MARKERS):
        return "email"
    return "none"


def extract_date_range(query: str, today: date) -> tuple[date, date] | None:
    """Rút khoảng ngày hoặc tháng để fallback vẫn lọc đúng khi LLM lỗi."""
    match = _DATE_RANGE_PATTERN.search(query)
    if not match:
        month_match = _MONTH_REFERENCE_PATTERN.search(query)
        if not month_match:
            return None
        month_raw, year_raw = month_match.groups()
        try:
            month = int(month_raw)
            year = int(year_raw) if year_raw else today.year
            if year < 100:
                year += 2000
            return date(year, month, 1), date(
                year, month, monthrange(year, month)[1]
            )
        except ValueError:
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
    search_query = _MONTH_REFERENCE_PATTERN.sub(" ", search_query)
    return re.sub(
        r"\b(?:từ|tu)(?:\s+(?:ngày|ngay))?\s*$",
        "",
        search_query,
        flags=re.IGNORECASE,
    ).strip() or query


def is_generic_date_summary(query: str) -> bool:
    """Nhận diện yêu cầu tổng hợp mọi email trong khoảng ngày, không có chủ đề."""
    if not has_date_reference(query):
        return False
    remainder = _normalize_text(_search_query_without_dates(query))
    return remainder in {
        "tong hop",
        "tong hop thong tin",
        "tong hop email",
        "tong hop mail",
        "tom tat",
        "tom tat thong tin",
        "tom tat email",
        "tom tat mail",
        "cac email",
        "cac mail",
        "email",
        "mail",
    }


def is_deadline_query(query: str) -> bool:
    """Nhận diện intent deadline để không bị search email chung làm lệch kết quả."""
    normalized = _normalize_text(query)
    return bool(re.search(r"\b(?:deadline|han chot|den han|qua han)\b", normalized))


def split_compound_query(query: str) -> tuple[str, ...]:
    """Tách các vế độc lập khi câu hỏi ghép deadline với một chủ đề khác."""
    parts = tuple(
        part.strip(" ,.!?;:")
        for part in _COMPOUND_SEPARATOR_PATTERN.split(query)
        if part.strip(" ,.!?;:")
    )
    if len(parts) < 2:
        return (query.strip(),)
    has_deadline_part = any(is_deadline_query(part) for part in parts)
    has_other_part = any(not is_deadline_query(part) for part in parts)
    return parts if has_deadline_part and has_other_part else (query.strip(),)


def is_compound_query(query: str) -> bool:
    """Cho biết câu hỏi có nhiều intent cần trả lời riêng hay không."""
    return len(split_compound_query(query)) > 1


def requests_exhaustive_listing(query: str) -> bool:
    """Nhận diện yêu cầu cần liệt kê đủ, không chỉ chọn vài ví dụ tiêu biểu."""
    normalized = _normalize_text(query)
    return bool(
        re.search(
            r"\b(?:liet ke|tat ca|day du|toan bo|danh sach|cac|nhung|bao nhieu)\b",
            normalized,
        )
    )


def infer_answer_mode(query: str) -> str:
    """Tách tổng hợp nội dung email khỏi tổng hợp deadline."""
    normalized = _normalize_text(query)
    if is_compound_query(query):
        return "multi_intent"
    if is_deadline_query(query):
        return "deadline_summary"
    if any(
        marker in normalized
        for marker in (
            "tong hop thong tin",
            "tong hop email",
            "tong hop mail",
            "tom tat thong tin",
            "tom tat email",
            "tom tat mail",
        )
    ):
        return "information_summary"
    return "general"


def includes_overdue(query: str) -> bool:
    """Chỉ gồm deadline quá hạn khi người dùng yêu cầu rõ ràng."""
    return "qua han" in _normalize_text(query)


def is_memory_query(query: str) -> bool:
    """Nhận diện câu hỏi tham chiếu rõ ràng tới hội thoại trước."""
    normalized = _normalize_text(query)
    return any(
        marker in normalized
        for marker in (
            "ban con nho",
            "da trao doi",
            "da noi",
            "ghi nho",
            "hoi truoc",
            "lan truoc",
            "toi tung noi",
            "truoc day",
        )
    )


def plan_read_tools(
    query: str, max_calls: int = 3, semantic_enabled: bool = False
) -> list[dict[str, Any]]:
    """Chọn tối đa vài read tool bằng quy tắc khi LLM planner không chạy được."""
    normalized = query.casefold()
    today = datetime.now(LOCAL_TIMEZONE).date()
    extracted_range = extract_date_range(query, today)
    search_query = _search_query_without_dates(query)
    search_limit = 50 if requests_exhaustive_listing(query) else 12
    date_arguments = (
        {
            "date_from": extracted_range[0].isoformat(),
            "date_to": extracted_range[1].isoformat(),
        }
        if extracted_range
        else {}
    )
    calls: list[dict[str, Any]] = []

    if is_memory_query(query):
        return [
            {
                "name": "search_memory",
                "arguments": {"query": query, "limit": 5},
            }
        ]

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
                    "include_overdue": any(
                        marker in normalized for marker in ("quá hạn", "overdue")
                    ),
                    **date_arguments,
                },
            }
        )
    if any(word in normalized for word in ("họp", "cuộc họp", "meeting", "lịch hẹn")):
        calls.append(
            {
                "name": "get_meetings",
                "arguments": dict(date_arguments),
            }
        )
    if any(word in normalized for word in ("việc", "yêu cầu", "đang mở", "chưa làm")):
        calls.append({"name": "get_open_items", "arguments": {"limit": 30}})
    if "hôm nay" in normalized:
        calls.append(
            {"name": "get_daily_brief", "arguments": {"day": today.isoformat()}}
        )

    explicit_lookup = bool(email_id or thread_id or addresses)
    if semantic_enabled and not explicit_lookup:
        calls.append(
            {
                "name": "hybrid_search_subjects",
                "arguments": {
                    "query": search_query,
                    "limit": search_limit,
                    **date_arguments,
                },
            }
        )
    calls.append(
        {
            "name": "search_emails",
            "arguments": {
                "query": search_query,
                "search_terms": [search_query],
                "limit": search_limit,
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
