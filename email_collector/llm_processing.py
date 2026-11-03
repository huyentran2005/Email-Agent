import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from .config import Settings
from .llm_proxy import LLMProxyClient, LLMProxyConfig, LLMResponseError
from .models import (
    Email,
    EmailEvent,
    EmailThread,
    EventChange,
    LLMRun,
    ThreadEventState,
    ThreadSummary,
)
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
)
from .repository import EmailRepository


EVENT_TYPES = {
    "Request Meeting",
    "Request Data",
    "Request Action",
    "Request Action Data",
    "Request Meeting Data",
    "Deliver Data",
    "Deliver Action Data",
    "Deliver Meeting Data",
    "Amend Data",
    "Amend Meeting Data",
}
EVENT_STATUSES = {"open", "tentative", "confirmed", "fulfilled", "declined", "cancelled", "superseded"}
CHANGE_TYPES = {"created", "updated", "confirmed", "fulfilled", "declined", "cancelled", "tentative", "no_change"}


def _request_hash(payload: dict) -> str:
    """Tạo mã băm ổn định cho payload gửi đến LLM."""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_key(value: str, event_type: str, index: int) -> str:
    """Chuẩn hóa và kiểm tra canonical key của một sự kiện."""
    value = re.sub(r"[^a-z0-9:_-]+", "-", value.lower()).strip("-")[:450]
    if not value:
        raise LLMResponseError(f"Event {index} has no canonical_key")
    return value


def _word_count(value: str) -> int:
    """Đếm số từ trong một chuỗi văn bản."""
    return len(value.split())


class ThreadLLMProcessor:
    def __init__(
        self,
        settings: Settings | None = None,
        client: LLMProxyClient | None = None,
    ):
        """Khởi tạo bộ xử lý thread cùng repository và LLM client."""
        self.settings = settings or Settings()
        self.repository = EmailRepository(self.settings.database_url)
        self.client = client or LLMProxyClient(
            LLMProxyConfig(
                base_url=self.settings.llm_proxy_base_url,
                api_key=self.settings.llm_proxy_api_key,
                model=self.settings.llm_model,
                timeout_seconds=self.settings.llm_timeout_seconds,
                max_attempts=self.settings.llm_max_attempts,
                json_mode=self.settings.llm_json_mode,
            )
        )

    def initialize(self) -> None:
        """Khởi tạo schema cơ sở dữ liệu cần thiết."""
        self.repository.create_schema()

    @staticmethod
    def _is_suspicious(email: Email) -> bool:
        """Xác định email có bị đánh dấu đáng ngờ hay không."""
        safety = email.safety_analysis or {}
        spoof = email.spoof_analysis or {}
        auth = spoof.get("authentication") or {}
        hard_auth_failure = auth.get("dmarc") in {"fail", "permerror"}
        return safety.get("verdict") == "suspicious" or hard_auth_failure

    @staticmethod
    def _states(session, thread_id: int) -> list[ThreadEventState]:
        """Lấy các trạng thái sự kiện của một thread."""
        return list(
            session.scalars(
                select(ThreadEventState)
                .where(ThreadEventState.thread_id == thread_id)
                .order_by(ThreadEventState.id)
            )
        )

    @staticmethod
    def _state_payload(states: list[ThreadEventState]) -> list[dict[str, Any]]:
        """Chuyển các trạng thái sự kiện thành payload gửi cho LLM."""
        return [
            {
                "id": state.id,
                "canonical_key": state.canonical_key,
                "event_type": state.event_type,
                "status": state.status,
                "current_value": state.current_value,
                "version": state.version,
            }
            for state in states
        ]

    def _start_run(self, session, thread_id: int, email_id: int, task: str, payload: dict) -> LLMRun:
        """Tạo mới hoặc đặt lại bản ghi theo dõi một lần chạy LLM."""
        run = session.scalar(
            select(LLMRun).where(
                LLMRun.source_email_id == email_id,
                LLMRun.task == task,
                LLMRun.prompt_version == PROMPT_VERSION,
            )
        )
        if run is None:
            run = LLMRun(
                thread_id=thread_id,
                source_email_id=email_id,
                task=task,
                prompt_version=PROMPT_VERSION,
                model=self.settings.llm_model,
                request_hash=_request_hash(payload),
                status="running",
            )
            session.add(run)
        else:
            run.request_hash = _request_hash(payload)
            run.status = "running"
            run.response_json = {}
            run.error = None
            run.started_at = datetime.now(timezone.utc)
            run.completed_at = None
        session.commit()
        return run

    @staticmethod
    def _finish_run(session, run: LLMRun, response: dict) -> None:
        """Đánh dấu lần chạy LLM đã hoàn thành thành công."""
        run.status = "success"
        run.response_json = response
        run.completed_at = datetime.now(timezone.utc)

    @staticmethod
    def _fail_run(session, run_id: int, error: Exception) -> None:
        """Rollback giao dịch và ghi nhận lỗi của lần chạy LLM."""
        session.rollback()
        run = session.get(LLMRun, run_id)
        if run:
            run.status = "failed"
            run.error = str(error)[:5000]
            run.completed_at = datetime.now(timezone.utc)
            session.commit()

    def _validate_extraction(self, response: dict, body_text: str) -> list[dict]:
        """Kiểm tra và chuẩn hóa các sự kiện do LLM trích xuất."""
        events = response.get("events")
        if not isinstance(events, list):
            raise LLMResponseError("events must be a list")
        validated: list[dict] = []
        for index, event in enumerate(events):
            if not isinstance(event, dict) or event.get("event_type") not in EVENT_TYPES:
                raise LLMResponseError(f"Invalid event_type at event {index}")
            evidence = event.get("evidence") or []
            if not isinstance(evidence, list):
                raise LLMResponseError("evidence must be a list")
            valid_evidence = [
                item for item in evidence
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
                and item["text"] in body_text
            ]
            if not valid_evidence:
                raise LLMResponseError(f"Event {index} has no exact evidence in current email")
            status = event.get("new_status", "open")
            change_type = event.get("change_type", "created")
            if status not in EVENT_STATUSES or change_type not in CHANGE_TYPES:
                raise LLMResponseError(f"Invalid state transition at event {index}")
            arguments = event.get("arguments") if isinstance(event.get("arguments"), dict) else {}
            state_after = event.get("state_after") if isinstance(event.get("state_after"), dict) else arguments
            validated.append(
                {
                    **event,
                    "arguments": arguments,
                    "state_after": state_after,
                    "evidence": valid_evidence,
                    "confidence": max(0.0, min(1.0, float(event.get("confidence", 0.0)))),
                    "new_status": status,
                    "change_type": change_type,
                    "canonical_key": _safe_key(str(event.get("canonical_key", "")), event["event_type"], index),
                }
            )
        return validated

    def _apply_events(self, session, email: Email, events: list[dict]) -> int:
        """Áp dụng sự kiện vào trạng thái và lịch sử thay đổi."""
        states = self._states(session, email.thread_id)
        by_id = {state.id: state for state in states}
        by_key = {state.canonical_key: state for state in states}
        created = 0
        for index, event in enumerate(events):
            target_id = event.get("target_state_id")
            state = by_id.get(target_id) if isinstance(target_id, int) else None
            state = state or by_key.get(event["canonical_key"])
            before: dict[str, Any] = {}
            if state is None:
                state = ThreadEventState(
                    thread_id=email.thread_id,
                    canonical_key=event["canonical_key"],
                    event_type=event["event_type"],
                    status=event["new_status"],
                    current_value=event["state_after"],
                    updated_by_email_id=email.id,
                    version=1,
                )
                session.add(state)
                session.flush()
                by_id[state.id] = state
                by_key[state.canonical_key] = state
            else:
                before = {"status": state.status, "current_value": copy.deepcopy(state.current_value)}
                if event["change_type"] != "no_change":
                    state.status = event["new_status"]
                    state.current_value = event["state_after"]
                    state.updated_by_email_id = email.id
                    state.version += 1

            email_event = EmailEvent(
                thread_id=email.thread_id,
                source_email_id=email.id,
                event_state_id=state.id,
                event_index=index,
                event_type=event["event_type"],
                trigger=str(event.get("trigger") or ""),
                arguments=event["arguments"],
                evidence=event["evidence"],
                confidence=event["confidence"],
                extraction_model=self.settings.llm_model,
                extractor_version=PROMPT_VERSION,
            )
            session.add(email_event)
            after = {"status": state.status, "current_value": copy.deepcopy(state.current_value)}
            if event["change_type"] != "no_change" or not before:
                session.add(
                    EventChange(
                        event_state_id=state.id,
                        source_email_id=email.id,
                        change_type=event["change_type"],
                        before_value=before,
                        after_value=after,
                    )
                )
            created += 1
        return created

    def _process_email(self, session, email: Email, recent_history: list[Email]) -> int:
        """Trích xuất và lưu các sự kiện của một email."""
        previous = session.scalar(
            select(LLMRun).where(
                LLMRun.source_email_id == email.id,
                LLMRun.task == "event_extraction",
                LLMRun.prompt_version == PROMPT_VERSION,
                LLMRun.status.in_(["success", "skipped"]),
            )
        )
        if previous:
            return 0
        payload = {
            "current_email": {
                "id": email.id,
                "subject": email.subject,
                "from": email.from_address,
                "to": email.to_addresses,
                "sent_at": email.sent_at.isoformat() if email.sent_at else None,
                "body": (email.body_text or "")[:30000],
            },
            "prior_event_states": self._state_payload(self._states(session, email.thread_id)),
            "recent_history": [
                {
                    "id": item.id,
                    "subject": item.subject,
                    "from": item.from_address,
                    "body_excerpt": (item.body_text or "")[:2000],
                }
                for item in recent_history[-3:]
            ],
        }
        run = self._start_run(session, email.thread_id, email.id, "event_extraction", payload)
        if self._is_suspicious(email) and not self.settings.llm_allow_suspicious_email:
            run.status = "skipped"
            run.error = "Email security analysis marked content suspicious"
            run.completed_at = datetime.now(timezone.utc)
            session.commit()
            return 0
        try:
            response = self.client.chat_json(EXTRACTION_SYSTEM_PROMPT, payload)
            events = self._validate_extraction(response, email.body_text or "")
            count = self._apply_events(session, email, events)
            delta = str(response.get("delta_summary") or "").strip()
            if delta:
                version = (session.scalar(
                    select(func.max(ThreadSummary.version)).where(
                        ThreadSummary.thread_id == email.thread_id,
                        ThreadSummary.summary_type == "delta",
                    )
                ) or 0) + 1
                session.add(
                    ThreadSummary(
                        thread_id=email.thread_id,
                        summary_type="delta",
                        summary_text=delta,
                        based_on_email_id=email.id,
                        supporting_email_ids=[email.id],
                        version=version,
                        model=self.settings.llm_model,
                    )
                )
            self._finish_run(session, run, response)
            session.commit()
            return count
        except Exception as exc:
            self._fail_run(session, run.id, exc)
            raise

    def _summary_payload(self, session, thread: EmailThread, emails: list[Email]) -> dict:
        """Tạo payload dùng để tóm tắt thread."""
        states = self._states(session, thread.id)
        changes = list(
            session.scalars(
                select(EventChange)
                .join(ThreadEventState, EventChange.event_state_id == ThreadEventState.id)
                .where(ThreadEventState.thread_id == thread.id)
                .order_by(EventChange.id)
            )
        )
        return {
            "thread": {"id": thread.id, "subject": thread.subject},
            "event_states": self._state_payload(states),
            "changes": [
                {
                    "source_email_id": change.source_email_id,
                    "change_type": change.change_type,
                    "before": change.before_value,
                    "after": change.after_value,
                }
                for change in changes
            ],
            "emails": [
                {"id": email.id, "from": email.from_address, "subject": email.subject}
                for email in emails
            ],
        }

    @staticmethod
    def _validate_summary(response: dict, valid_email_ids: set[int]) -> tuple[str, str, list[int]]:
        """Kiểm tra giới hạn và nguồn hỗ trợ của bản tóm tắt."""
        short = str(response.get("short_summary") or "").strip()
        long = str(response.get("long_summary") or "").strip()
        if not short or not long or _word_count(short) > 30 or _word_count(long) > 100:
            raise LLMResponseError("Summary is empty or exceeds EMAIL SUM limits")
        supporting = response.get("supporting_email_ids") or []
        supporting = [item for item in supporting if isinstance(item, int) and item in valid_email_ids]
        if not supporting:
            raise LLMResponseError("Summary has no valid supporting_email_ids")
        return short, long, list(dict.fromkeys(supporting))

    def _summarize(self, session, thread: EmailThread, emails: list[Email]) -> int:
        """Tạo, kiểm chứng và lưu các bản tóm tắt của thread."""
        if not emails:
            return 0
        payload = self._summary_payload(session, thread, emails)
        if not payload["event_states"]:
            return 0
        last_email = emails[-1]
        previous = session.scalar(
            select(LLMRun).where(
                LLMRun.source_email_id == last_email.id,
                LLMRun.task == "thread_summary",
                LLMRun.prompt_version == PROMPT_VERSION,
                LLMRun.status == "success",
            )
        )
        if previous:
            return 0
        run = self._start_run(session, thread.id, last_email.id, "thread_summary", payload)
        verify_run: LLMRun | None = None
        try:
            response = self.client.chat_json(SUMMARY_SYSTEM_PROMPT, payload)
            short, long, supporting = self._validate_summary(response, {email.id for email in emails})
            verification: dict[str, Any] = {"supported": None, "skipped": True}
            if self.settings.llm_verify_summary:
                verify_payload = {
                    **payload,
                    "candidate": {"short_summary": short, "long_summary": long},
                }
                verify_run = self._start_run(
                    session, thread.id, last_email.id, "summary_verification", verify_payload
                )
                verification = self.client.chat_json(VERIFY_SYSTEM_PROMPT, verify_payload)
                if verification.get("supported") is False:
                    corrected = {
                        "short_summary": verification.get("corrected_short_summary"),
                        "long_summary": verification.get("corrected_long_summary"),
                        "supporting_email_ids": supporting,
                    }
                    short, long, supporting = self._validate_summary(
                        corrected, {email.id for email in emails}
                    )
                self._finish_run(session, verify_run, verification)

            for summary_type, summary_text in (("short", short), ("long", long)):
                version = (session.scalar(
                    select(func.max(ThreadSummary.version)).where(
                        ThreadSummary.thread_id == thread.id,
                        ThreadSummary.summary_type == summary_type,
                    )
                ) or 0) + 1
                session.add(
                    ThreadSummary(
                        thread_id=thread.id,
                        summary_type=summary_type,
                        summary_text=summary_text,
                        based_on_email_id=last_email.id,
                        supporting_email_ids=supporting,
                        verification=verification,
                        version=version,
                        model=self.settings.llm_model,
                    )
                )
            self._finish_run(session, run, response)
            session.commit()
            return 2
        except Exception as exc:
            if verify_run is not None:
                self._fail_run(session, verify_run.id, exc)
            self._fail_run(session, run.id, exc)
            raise

    def process_thread(self, thread_id: int) -> dict[str, int]:
        """Xử lý sự kiện và tóm tắt cho một thread email."""
        with self.repository.session_factory() as session:
            thread = session.get(EmailThread, thread_id)
            if thread is None:
                raise ValueError(f"Thread {thread_id} not found")
            emails = list(
                session.scalars(
                    select(Email)
                    .where(Email.thread_id == thread_id)
                    .order_by(Email.received_at.asc().nulls_last(), Email.id)
                )
            )
            event_count = 0
            history: list[Email] = []
            for email in emails:
                event_count += self._process_email(session, email, history)
                history.append(email)
            summaries = self._summarize(session, thread, emails)
            return {"emails": len(emails), "events": event_count, "summaries": summaries}

    def process_all(self) -> dict[str, int]:
        """Xử lý toàn bộ thread email trong cơ sở dữ liệu."""
        with self.repository.session_factory() as session:
            thread_ids = list(session.scalars(select(EmailThread.id).order_by(EmailThread.id)))
        totals = {"threads": 0, "emails": 0, "events": 0, "summaries": 0}
        for thread_id in thread_ids:
            result = self.process_thread(thread_id)
            totals["threads"] += 1
            for key in ("emails", "events", "summaries"):
                totals[key] += result[key]
        return totals
