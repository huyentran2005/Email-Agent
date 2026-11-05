"""Các tool chỉ đọc dùng cho trợ lý hỏi đáp email."""

from __future__ import annotations

import unicodedata
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import or_, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from core.config import Settings
from core.observability import LangfuseTracer
from core.models import (
    Email,
    EmailEvent,
    EmailThread,
    ThreadEventState,
    ThreadSummary,
)

from .subject_embeddings import (
    EmbeddingError,
    QdrantError,
    SubjectHybridSearch,
)
from . import fallback_planner


LOCAL_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
OPEN_STATUSES = {"open", "tentative", "confirmed"}
ACTIONABLE_EVENT_TYPES = {
    "Request Meeting",
    "Request Data",
    "Request Action",
    "Request Action Data",
    "Request Meeting Data",
}


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": "search_emails",
        "description": "Tìm email theo từ khóa, người gửi và khoảng ngày tùy chọn.",
        "arguments": [
            "query",
            "search_terms",
            "from_address",
            "date_from",
            "date_to",
            "limit",
        ],
        "read_only": True,
    },
    {
        "name": "hybrid_search_subjects",
        "description": "Tìm subject bằng dense semantic + sparse keyword trên Qdrant; khoảng ngày là tùy chọn.",
        "arguments": ["query", "date_from", "date_to", "limit"],
        "read_only": True,
    },
    {
        "name": "get_email",
        "description": "Đọc một email theo ID nội bộ.",
        "arguments": ["email_id"],
        "read_only": True,
    },
    {
        "name": "get_thread",
        "description": "Đọc email, event, state và summary của một thread.",
        "arguments": ["thread_id"],
        "read_only": True,
    },
    {
        "name": "get_open_items",
        "description": "Lấy các yêu cầu hoặc công việc đang mở.",
        "arguments": ["limit"],
        "read_only": True,
    },
    {
        "name": "get_deadlines",
        "description": "Lấy deadline và việc quá hạn trong khoảng ngày.",
        "arguments": ["date_from", "date_to", "include_overdue", "limit"],
        "read_only": True,
    },
    {
        "name": "get_meetings",
        "description": "Lấy meeting request và lịch họp đã extract.",
        "arguments": ["date_from", "date_to", "limit"],
        "read_only": True,
    },
    {
        "name": "get_daily_brief",
        "description": "Tổng hợp email và công việc của một ngày từ PostgreSQL.",
        "arguments": ["day"],
        "read_only": True,
    },
    {
        "name": "get_sender_history",
        "description": "Lấy lịch sử trao đổi với một địa chỉ email.",
        "arguments": ["email_address", "limit"],
        "read_only": True,
    },
    {
        "name": "get_email_sources",
        "description": "Lấy metadata nguồn cho danh sách email ID.",
        "arguments": ["email_ids"],
        "read_only": True,
    },
)

SEARCH_TOOL_NAMES = frozenset({"search_emails", "hybrid_search_subjects"})
FUSED_SEARCH_TOOL_NAME = "fused_search_results"


def _validated_search_terms(value: Any, fallback_query: str) -> list[str]:
    """Kiểm tra term của planner và bổ sung alias nghiệp vụ cần thiết."""
    raw_terms = value if isinstance(value, list) else []
    terms: list[str] = []
    for item in raw_terms[:8]:
        if not isinstance(item, str):
            continue
        term = " ".join(item.split()).casefold()
        if 2 <= len(term) <= 100 and term not in terms:
            terms.append(term)
    fallback = " ".join(fallback_query.split()).casefold()[:200]
    if not terms and fallback:
        terms.append(fallback)

    normalized = unicodedata.normalize(
        "NFKD", " ".join([fallback, *terms])
    ).replace("đ", "d")
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    if "cap quyen" in normalized and (
        "du an" in normalized or "project" in normalized
    ):
        # Email GitLab dùng subject tiếng Anh dạng
        # "Access to ... project was granted", trong khi user thường hỏi tiếng Việt.
        for synonym in ("access to the", "was granted", "project"):
            if synonym not in terms and len(terms) < 8:
                terms.append(synonym)
    return terms


def _fuse_search_results(
    tool_results: list[dict[str, Any]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Hợp nhất keyword + hybrid theo RRF và loại email trùng."""
    ranked_results = [
        item
        for item in tool_results
        if item.get("tool") in SEARCH_TOOL_NAMES
        and isinstance(item.get("result"), list)
    ]
    if len(ranked_results) < 2:
        return tool_results
    if limit is None:
        requested_limits = [
            item.get("arguments", {}).get("limit")
            for item in ranked_results
            if isinstance(item.get("arguments"), dict)
        ]
        numeric_limits = [
            int(value) for value in requested_limits if str(value).isdigit()
        ]
        limit = max(numeric_limits, default=20)

    scores: dict[int, float] = {}
    rows: dict[int, dict[str, Any]] = {}
    methods: dict[int, list[str]] = {}
    for tool_result in ranked_results:
        method = str(tool_result["tool"])
        for rank, raw_row in enumerate(tool_result["result"], start=1):
            if not isinstance(raw_row, dict):
                continue
            try:
                email_id = int(raw_row["email_id"])
            except (KeyError, TypeError, ValueError):
                continue
            scores[email_id] = scores.get(email_id, 0.0) + 1.0 / (60 + rank)
            methods.setdefault(email_id, []).append(method)
            if email_id not in rows or method == "search_emails":
                rows[email_id] = dict(raw_row)

    fused_rows: list[dict[str, Any]] = []
    for email_id in sorted(scores, key=lambda item: (-scores[item], item))[
        : max(1, min(int(limit), 50))
    ]:
        row = dict(rows[email_id])
        row["fusion_score"] = round(scores[email_id], 8)
        row["retrieval_methods"] = list(dict.fromkeys(methods[email_id]))
        fused_rows.append(row)

    fused_item = {
        "tool": FUSED_SEARCH_TOOL_NAME,
        "arguments": {
            "methods": [item["tool"] for item in ranked_results],
            "limit": limit,
        },
        "result": fused_rows,
    }
    first_search_index = next(
        index
        for index, item in enumerate(tool_results)
        if item.get("tool") in SEARCH_TOOL_NAMES
    )
    preserved = [
        item for item in tool_results if item.get("tool") not in SEARCH_TOOL_NAMES
    ]
    preserved.insert(min(first_search_index, len(preserved)), fused_item)
    return preserved


def _is_suspicious(email: Email) -> bool:
    """Loại email đáng ngờ khỏi dữ liệu cung cấp cho agent."""
    safety = email.safety_analysis or {}
    spoof = email.spoof_analysis or {}
    authentication = spoof.get("authentication") or {}
    return safety.get("verdict") == "suspicious" or authentication.get(
        "dmarc"
    ) in {"fail", "permerror"}


def _iso(value: datetime | None) -> str | None:
    """Chuyển datetime thành chuỗi ISO hoặc giữ None."""
    return value.isoformat() if value else None


def _parse_date(value: Any) -> date | None:
    """Phân tích ngày từ đối tượng date, datetime hoặc chuỗi phổ biến."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    for parser in (
        lambda: datetime.fromisoformat(raw.replace("Z", "+00:00")).date(),
        lambda: datetime.strptime(raw, "%d/%m/%Y").date(),
        lambda: datetime.strptime(raw, "%d-%m-%Y").date(),
    ):
        try:
            return parser()
        except ValueError:
            continue
    return None


def _date_bounds(
    date_from: str | date | None,
    date_to: str | date | None,
) -> tuple[datetime | None, datetime | None]:
    """Đổi khoảng ngày local thành [start, end) UTC; date_to bao gồm cả ngày."""
    start_day = _parse_date(date_from)
    end_day = _parse_date(date_to)
    start = (
        datetime.combine(start_day, time.min, tzinfo=LOCAL_TIMEZONE).astimezone(
            timezone.utc
        )
        if start_day
        else None
    )
    end = (
        datetime.combine(
            end_day + timedelta(days=1), time.min, tzinfo=LOCAL_TIMEZONE
        ).astimezone(timezone.utc)
        if end_day
        else None
    )
    return start, end


def _email_payload(email: Email, body_limit: int = 1200) -> dict[str, Any]:
    """Chuyển email thành payload giới hạn nội dung cho read tool."""
    return {
        "email_id": email.id,
        "thread_id": email.thread_id,
        "subject": email.subject or "(Không có tiêu đề)",
        "from": email.from_address or "(Không rõ người gửi)",
        "to": email.to_addresses,
        "received_at": _iso(email.received_at or email.created_at),
        "labels": email.labels or [],
        "body_excerpt": (email.body_text or "")[:body_limit],
    }


class ReadOnlyEmailTools:
    """Tool gateway chỉ đọc; lớp này không gọi commit hoặc API ghi bên ngoài."""

    def __init__(self, settings: Settings | None = None):
        """Khởi tạo tool gateway và registry chỉ gồm các thao tác đọc."""
        self.settings = settings or Settings()
        self.tracer = LangfuseTracer(self.settings.langfuse_enabled)
        self.engine = create_engine(self.settings.database_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        self._hybrid_search: SubjectHybridSearch | None = None
        self._registry = {
            "search_emails": self.search_emails,
            "hybrid_search_subjects": self.hybrid_search_subjects,
            "get_email": self.get_email,
            "get_thread": self.get_thread,
            "get_open_items": self.get_open_items,
            "get_deadlines": self.get_deadlines,
            "get_meetings": self.get_meetings,
            "get_daily_brief": self.get_daily_brief,
            "get_sender_history": self.get_sender_history,
            "get_email_sources": self.get_email_sources,
        }

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Thực thi read tool theo tên và từ chối tool ngoài registry."""
        tool = self._registry.get(name)
        if tool is None:
            raise ValueError(f"Unknown or non-read-only tool: {name}")
        return tool(**(arguments or {}))

    def planner_schemas(self) -> list[dict[str, Any]]:
        """Trả schema read tool khả dụng cho LLM planner theo cấu hình hiện tại."""
        return [
            dict(schema)
            for schema in TOOL_SCHEMAS
            if schema["name"] != "get_email_sources"
            and (
                schema["name"] != "hybrid_search_subjects"
                or bool(self.settings.embedding_model)
            )
        ]

    def normalize_calls(
        self,
        raw_calls: Any,
        query: str,
        max_calls: int = 3,
    ) -> list[dict[str, Any]]:
        """Kiểm tra allowlist tool và tham số do LLM planner sinh ra."""
        schemas = {item["name"]: item for item in self.planner_schemas()}
        required_arguments = {
            "search_emails": {"query"},
            "hybrid_search_subjects": {"query"},
            "get_email": {"email_id"},
            "get_thread": {"thread_id"},
            "get_sender_history": {"email_address"},
        }
        if not isinstance(raw_calls, list):
            return []
        calls: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "")
            if name not in schemas or name in seen:
                continue
            raw_arguments = raw.get("arguments")
            if not isinstance(raw_arguments, dict):
                raw_arguments = {}
            allowed = set(schemas[name].get("arguments") or [])
            arguments = {
                key: value
                for key, value in raw_arguments.items()
                if key in allowed and value is not None
            }
            if name in {"search_emails", "hybrid_search_subjects"}:
                arguments["query"] = str(arguments.get("query") or query).strip()
            if name == "search_emails":
                arguments["search_terms"] = _validated_search_terms(
                    arguments.get("search_terms"), arguments["query"]
                )
            for key in ("date_from", "date_to"):
                if key in arguments:
                    parsed = _parse_date(arguments[key])
                    if parsed is None:
                        arguments.pop(key)
                    else:
                        arguments[key] = parsed.isoformat()
            for key in ("limit", "email_id", "thread_id"):
                if key in arguments:
                    try:
                        arguments[key] = int(arguments[key])
                    except (TypeError, ValueError):
                        arguments.pop(key)
            if "email_ids" in arguments:
                if not isinstance(arguments["email_ids"], list):
                    arguments.pop("email_ids")
                else:
                    arguments["email_ids"] = [
                        int(item)
                        for item in arguments["email_ids"][:100]
                        if str(item).isdigit()
                    ]
            if "include_overdue" in arguments:
                value = arguments["include_overdue"]
                arguments["include_overdue"] = (
                    value
                    if isinstance(value, bool)
                    else str(value).casefold() in {"1", "true", "yes"}
                )
            if not required_arguments.get(name, set()).issubset(arguments):
                continue
            calls.append({"name": name, "arguments": arguments})
            seen.add(name)
            if len(calls) >= max(1, min(int(max_calls), 5)):
                break

        extracted_range = fallback_planner.extract_date_range(
            query, datetime.now(LOCAL_TIMEZONE).date()
        )
        exhaustive_listing = fallback_planner.requests_exhaustive_listing(query)
        search_limit = 50 if exhaustive_listing else 12
        if (
            fallback_planner.is_deadline_query(query)
            and not fallback_planner.is_compound_query(query)
        ):
            deadline_arguments: dict[str, Any] = {
                "include_overdue": fallback_planner.includes_overdue(query),
                "limit": 100,
            }
            if extracted_range:
                deadline_arguments.update(
                    {
                        "date_from": extracted_range[0].isoformat(),
                        "date_to": extracted_range[1].isoformat(),
                    }
                )
            return [{"name": "get_deadlines", "arguments": deadline_arguments}]
        if fallback_planner.is_generic_date_summary(query) and extracted_range:
            return [
                {
                    "name": "search_emails",
                    "arguments": {
                        "query": "",
                        "search_terms": [],
                        "date_from": extracted_range[0].isoformat(),
                        "date_to": extracted_range[1].isoformat(),
                        "limit": 50,
                    },
                }
            ]

        direct_lookup = any(
            call["name"] in {"get_email", "get_thread", "get_sender_history"}
            for call in calls
        )
        if calls and self.settings.embedding_model and not direct_lookup:
            present = {call["name"] for call in calls}
            shared_arguments: dict[str, Any] = {}
            for call in calls:
                for key in ("date_from", "date_to"):
                    if call["arguments"].get(key):
                        shared_arguments[key] = call["arguments"][key]
            existing_search = next(
                (call for call in calls if call["name"] in SEARCH_TOOL_NAMES),
                None,
            )
            search_query = " ".join(
                str(
                    (existing_search or {}).get("arguments", {}).get("query")
                    or query
                ).split()
            )
            if "search_emails" not in present:
                calls.append(
                    {
                        "name": "search_emails",
                        "arguments": {
                            "query": search_query,
                            "search_terms": _validated_search_terms(None, search_query),
                            "limit": search_limit,
                            **shared_arguments,
                        },
                    }
                )
            if "hybrid_search_subjects" not in present:
                calls.append(
                    {
                        "name": "hybrid_search_subjects",
                        "arguments": {
                            "query": search_query,
                            "limit": search_limit,
                            **shared_arguments,
                        },
                    }
                )
            max_normalized_calls = max(2, min(int(max_calls), 5))
            if len(calls) > max_normalized_calls:
                search_pair = [
                    call for call in calls if call["name"] in SEARCH_TOOL_NAMES
                ]
                specialist_calls = [
                    call for call in calls if call["name"] not in SEARCH_TOOL_NAMES
                ]
                calls = (
                    specialist_calls[: max_normalized_calls - 2] + search_pair[:2]
                )

        if exhaustive_listing:
            for call in calls:
                if call["name"] in SEARCH_TOOL_NAMES:
                    call["arguments"]["limit"] = 50
        if not fallback_planner.has_date_reference(query):
            calls = [call for call in calls if call["name"] != "get_daily_brief"]
            for call in calls:
                for key in ("date_from", "date_to", "day"):
                    call["arguments"].pop(key, None)
            return calls
        shared_dates: dict[str, str] = {}
        if extracted_range:
            shared_dates = {
                "date_from": extracted_range[0].isoformat(),
                "date_to": extracted_range[1].isoformat(),
            }
        for call in calls:
            if call["name"] not in SEARCH_TOOL_NAMES:
                continue
            for key in ("date_from", "date_to"):
                value = call["arguments"].get(key)
                if value and key not in shared_dates:
                    shared_dates[key] = value
        if shared_dates:
            for call in calls:
                if call["name"] in SEARCH_TOOL_NAMES:
                    call["arguments"].update(shared_dates)
        return calls

    def run_calls(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Thực thi danh sách read tool đã được kiểm tra bởi normalize_calls."""
        results = []
        for call in calls:
            trace_input = (
                call["arguments"]
                if self.settings.langfuse_capture_content
                else {"argument_names": sorted(call["arguments"])}
            )
            with self.tracer.observation(
                f"email-tool.{call['name']}",
                as_type="tool",
                input=trace_input,
                metadata={"read_only": "true"},
            ) as observation:
                try:
                    result = self.execute(call["name"], call["arguments"])
                except (EmbeddingError, QdrantError) as exc:
                    result = {"error": str(exc), "fallback": "search_emails"}
                result_count = (
                    len(result)
                    if isinstance(result, (list, tuple, dict))
                    else int(result is not None)
                )
                observation.update(
                    output=result
                    if self.settings.langfuse_capture_content
                    else {
                        "result_count": result_count,
                        "has_error": isinstance(result, dict)
                        and "error" in result,
                    }
                )
            results.append(
                {
                    "tool": call["name"],
                    "arguments": call["arguments"],
                    "result": result,
                }
            )
        return _fuse_search_results(results)

    def run_for_query(self, query: str, max_calls: int = 3) -> list[dict[str, Any]]:
        """Dùng router quy tắc làm fallback rồi thực thi các read tool."""
        calls = self.normalize_calls(
            fallback_planner.plan_read_tools(
                query,
                max_calls=max_calls,
                semantic_enabled=bool(self.settings.embedding_model),
            ),
            query,
            max_calls=max_calls,
        )
        return self.run_calls(calls)

    def hybrid_search_subjects(
        self,
        query: str,
        date_from: str | date | None = None,
        date_to: str | date | None = None,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Tìm dense+sparse trên subject trong khoảng ngày được yêu cầu."""
        if not self.settings.embedding_model:
            return []
        received_at_from, received_at_to = _date_bounds(date_from, date_to)
        if (
            received_at_from is not None
            and received_at_to is not None
            and received_at_from >= received_at_to
        ):
            return []
        if self._hybrid_search is None:
            self._hybrid_search = SubjectHybridSearch(self.settings)
        return self._hybrid_search.search(
            query,
            limit=limit,
            received_at_from=received_at_from,
            received_at_to=received_at_to,
        )

    def search_emails(
        self,
        query: str,
        from_address: str | None = None,
        date_from: str | date | None = None,
        date_to: str | date | None = None,
        limit: int = 20,
        search_terms: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Tìm email bằng các term đã được LLM planner tách và kiểm tra."""
        terms = _validated_search_terms(search_terms, query)
        start, end = _date_bounds(date_from, date_to)
        limit = max(1, min(int(limit), 50))
        with self.session_factory() as session:
            statement = select(Email).where(Email.processing_status == "processed")
            if from_address:
                statement = statement.where(Email.from_address.ilike(f"%{from_address}%"))
            if start:
                statement = statement.where(Email.received_at >= start)
            if end:
                statement = statement.where(Email.received_at < end)
            if terms:
                conditions = []
                for term in terms:
                    pattern = f"%{term}%"
                    conditions.extend(
                        (
                            Email.subject.ilike(pattern),
                            Email.from_address.ilike(pattern),
                            Email.body_text.ilike(pattern),
                        )
                    )
                statement = statement.where(or_(*conditions))
            candidates = list(
                session.scalars(
                    statement.order_by(Email.received_at.desc().nulls_last(), Email.id.desc()).limit(200)
                )
            )
        candidates = [item for item in candidates if not _is_suspicious(item)]
        if terms:
            candidates.sort(
                key=lambda item: -sum(
                    " ".join((item.subject or "", item.from_address or "", item.body_text or ""))
                    .casefold()
                    .count(term)
                    for term in terms
                )
            )
        return [_email_payload(item) for item in candidates[:limit]]

    def get_email(self, email_id: int) -> dict[str, Any] | None:
        """Đọc một email an toàn cùng các event được trích xuất."""
        with self.session_factory() as session:
            email = session.get(Email, int(email_id))
            if email is None or _is_suspicious(email):
                return None
            payload = _email_payload(email, body_limit=5000)
            events = list(
                session.scalars(
                    select(EmailEvent)
                    .where(EmailEvent.source_email_id == email.id)
                    .order_by(EmailEvent.event_index)
                )
            )
            payload["events"] = [
                {
                    "event_type": item.event_type,
                    "trigger": item.trigger,
                    "arguments": item.arguments,
                    "evidence": item.evidence,
                    "confidence": item.confidence,
                }
                for item in events
            ]
            return payload

    def get_thread(self, thread_id: int) -> dict[str, Any] | None:
        """Đọc email, state và summary mới nhất của một thread."""
        with self.session_factory() as session:
            thread = session.get(EmailThread, int(thread_id))
            if thread is None:
                return None
            emails = list(
                session.scalars(
                    select(Email)
                    .where(Email.thread_id == thread.id)
                    .order_by(Email.received_at.asc().nulls_last(), Email.id)
                    .limit(50)
                )
            )
            states = list(
                session.scalars(
                    select(ThreadEventState)
                    .where(ThreadEventState.thread_id == thread.id)
                    .order_by(ThreadEventState.id)
                )
            )
            summaries = list(
                session.scalars(
                    select(ThreadSummary)
                    .where(ThreadSummary.thread_id == thread.id)
                    .order_by(ThreadSummary.version.desc(), ThreadSummary.id.desc())
                )
            )
        safe_emails = [item for item in emails if not _is_suspicious(item)]
        latest_summaries: dict[str, dict[str, Any]] = {}
        for item in summaries:
            latest_summaries.setdefault(
                item.summary_type,
                {
                    "summary_type": item.summary_type,
                    "summary_text": item.summary_text,
                    "supporting_email_ids": item.supporting_email_ids,
                },
            )
        return {
            "thread_id": thread.id,
            "subject": thread.subject,
            "emails": [_email_payload(item) for item in safe_emails],
            "states": [self._state_payload(item) for item in states],
            "summaries": list(latest_summaries.values()),
        }

    @staticmethod
    def _state_payload(state: ThreadEventState) -> dict[str, Any]:
        """Chuyển trạng thái event thành payload cho agent."""
        return {
            "state_id": state.id,
            "thread_id": state.thread_id,
            "event_type": state.event_type,
            "status": state.status,
            "current_value": state.current_value or {},
            "updated_by_email_id": state.updated_by_email_id,
            "updated_at": _iso(state.updated_at),
        }

    def get_open_items(self, limit: int = 50) -> list[dict[str, Any]]:
        """Lấy các công việc hoặc yêu cầu đang ở trạng thái mở."""
        with self.session_factory() as session:
            states = list(
                session.scalars(
                    select(ThreadEventState)
                    .where(
                        ThreadEventState.status.in_(sorted(OPEN_STATUSES)),
                        ThreadEventState.event_type.in_(sorted(ACTIONABLE_EVENT_TYPES)),
                    )
                    .order_by(ThreadEventState.updated_at.desc())
                    .limit(max(1, min(int(limit), 100)))
                )
            )
        return [self._state_payload(item) for item in states]

    def get_deadlines(
        self,
        date_from: str | date | None = None,
        date_to: str | date | None = None,
        include_overdue: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Lấy deadline, chỉ giới hạn ngày khi caller truyền mốc thời gian."""
        start = _parse_date(date_from)
        end = _parse_date(date_to)
        today = datetime.now(LOCAL_TIMEZONE).date()
        items = self.get_open_items(limit=limit)
        results = []
        for item in items:
            value = item["current_value"]
            raw_deadline = value.get("deadline") or value.get("meeting_time")
            parsed = _parse_date(raw_deadline)
            if parsed is None:
                continue
            if start is not None and parsed < start and not include_overdue:
                continue
            if end is not None and parsed > end:
                continue
            item = dict(item)
            item["deadline"] = str(raw_deadline)
            item["deadline_date"] = parsed.isoformat()
            item["deadline_status"] = (
                "overdue" if parsed < today else "due_today" if parsed == today else "upcoming"
            )
            results.append(item)
        results.sort(key=lambda item: (item["deadline_date"], item["state_id"]))
        return results

    def get_meetings(
        self,
        date_from: str | date | None = None,
        date_to: str | date | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Lấy event họp, chỉ giới hạn ngày khi caller truyền mốc thời gian."""
        return [
            item
            for item in self.get_deadlines(
                date_from, date_to, include_overdue=False, limit=limit
            )
            if "Meeting" in item["event_type"]
            or item["current_value"].get("meeting_time")
        ]

    def get_daily_brief(self, day: str | date | None = None) -> dict[str, Any]:
        """Tổng hợp email, deadline và việc mở của một ngày."""
        selected = _parse_date(day) or datetime.now(LOCAL_TIMEZONE).date()
        start = datetime.combine(selected, time.min, tzinfo=LOCAL_TIMEZONE).astimezone(timezone.utc)
        end = datetime.combine(selected + timedelta(days=1), time.min, tzinfo=LOCAL_TIMEZONE).astimezone(timezone.utc)
        with self.session_factory() as session:
            emails = list(
                session.scalars(
                    select(Email)
                    .where(Email.received_at >= start, Email.received_at < end)
                    .order_by(Email.received_at.desc())
                    .limit(50)
                )
            )
        safe_emails = [item for item in emails if not _is_suspicious(item)]
        deadlines = self.get_deadlines(selected, selected, include_overdue=True)
        open_items = self.get_open_items(limit=20)
        return {
            "day": selected.isoformat(),
            "email_count": len(safe_emails),
            "emails": [_email_payload(item, body_limit=500) for item in safe_emails[:20]],
            "deadlines": deadlines,
            "open_items": open_items,
            "limitations": [
                "Category, priority, VIP và requires_response chưa có model trong source hiện tại."
            ],
        }

    def get_sender_history(self, email_address: str, limit: int = 50) -> list[dict[str, Any]]:
        """Lấy lịch sử email nhận từ một địa chỉ người gửi."""
        return self.search_emails(
            query=email_address,
            from_address=email_address,
            limit=limit,
        )

    def get_email_sources(self, email_ids: list[int]) -> list[dict[str, Any]]:
        """Lấy metadata nguồn cho các email ID đã được agent sử dụng."""
        unique_ids = list(dict.fromkeys(int(item) for item in email_ids))[:100]
        if not unique_ids:
            return []
        with self.session_factory() as session:
            emails = list(
                session.scalars(
                    select(Email)
                    .where(Email.id.in_(unique_ids))
                    .order_by(Email.received_at.desc().nulls_last(), Email.id.desc())
                )
            )
        return [
            {
                "email_id": item.id,
                "thread_id": item.thread_id,
                "subject": item.subject or "(Không có tiêu đề)",
                "from_address": item.from_address or "(Không rõ người gửi)",
                "received_at": _iso(item.received_at or item.created_at),
            }
            for item in emails
            if not _is_suspicious(item)
        ]

    @staticmethod
    def collect_source_ids(tool_results: list[dict[str, Any]]) -> list[int]:
        """Thu thập và loại trùng email ID từ kết quả read tool."""
        ids: list[int] = []

        def visit(value: Any) -> None:
            """Duyệt đệ quy payload để tìm các trường chứa email ID."""
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"email_id", "source_email_id", "updated_by_email_id"} and isinstance(item, int):
                        ids.append(item)
                    elif key == "supporting_email_ids" and isinstance(item, list):
                        ids.extend(number for number in item if isinstance(number, int))
                    else:
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(tool_results)
        return list(dict.fromkeys(ids))
