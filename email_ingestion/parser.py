import base64
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.utils import getaddresses, parsedate_to_datetime
from typing import Any

from email_filtering_understanding.content_processing import html_to_text, normalize_plain_text
from email_filtering_understanding.security import analyze_content_safety, analyze_spoofing


@dataclass
class ParsedAttachment:
    attachment_id: str
    filename: str
    mime_type: str
    size: int


@dataclass
class ParsedEmail:
    provider_message_id: str
    provider_thread_id: str
    internet_message_id: str | None
    from_address: str | None
    to_addresses: list[dict[str, str]]
    cc_addresses: list[dict[str, str]]
    bcc_addresses: list[dict[str, str]]
    subject: str | None
    body_text: str | None
    body_html: str | None
    labels: list[str]
    sent_at: datetime | None
    received_at: datetime | None
    raw_payload: dict[str, Any]
    headers: dict[str, list[str]]
    spoof_analysis: dict[str, Any] = field(default_factory=dict)
    safety_analysis: dict[str, Any] = field(default_factory=dict)
    attachments: list[ParsedAttachment] = field(default_factory=list)


def _decode_base64url(data: str | None) -> str:
    """Giải mã chuỗi base64url của Gmail thành văn bản UTF-8."""
    if not data:
        return ""
    padding = "=" * (-len(data) % 4)
    decoded = base64.urlsafe_b64decode(data + padding)
    return decoded.decode("utf-8", errors="replace")


def _decode_header(value: str | None) -> str | None:
    """Giải mã MIME encoded-word trong các header như Subject và filename."""
    if not value:
        return None
    try:
        return str(make_header(decode_header(value)))
    except (LookupError, UnicodeDecodeError):
        return value


def _parse_addresses(value: str | None) -> list[dict[str, str]]:
    """Tách một header địa chỉ thành danh sách tên và địa chỉ email chuẩn hóa."""
    if not value:
        return []
    return [
        {"name": _decode_header(name) or "", "address": address}
        for name, address in getaddresses([value])
        if address
    ]


def _parse_date(value: str | None) -> datetime | None:
    """Chuyển ngày trong header email thành datetime có timezone."""
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (TypeError, ValueError, OverflowError):
        return None


def _strip_body_data(message: dict[str, Any]) -> dict[str, Any]:
    """Tạo bản sao payload và loại dữ liệu body base64 trước khi lưu database."""
    raw = copy.deepcopy(message)

    def visit(part: dict[str, Any]) -> None:
        """Duyệt đệ quy từng MIME part để xóa trường dữ liệu body."""
        body = part.get("body", {})
        if "data" in body:
            body["data"] = "<removed>"
        for child in part.get("parts", []):
            visit(child)

    if "payload" in raw:
        visit(raw["payload"])
    return raw


def parse_gmail_message(message: dict[str, Any]) -> ParsedEmail:
    """Parse Gmail message thô thành cấu trúc email đã chuẩn hóa để lưu trữ."""
    payload = message.get("payload", {})
    headers: dict[str, list[str]] = {}
    for header in payload.get("headers", []):
        headers.setdefault(header["name"].lower(), []).append(header.get("value", ""))

    def first_header(name: str) -> str | None:
        """Lấy giá trị đầu tiên của một header email."""
        values = headers.get(name, [])
        return values[0] if values else None
    text_parts: list[str] = []
    html_parts: list[str] = []
    attachments: list[ParsedAttachment] = []

    def visit(part: dict[str, Any]) -> None:
        """Duyệt MIME tree để thu thập text, HTML và metadata attachment."""
        mime_type = part.get("mimeType", "application/octet-stream")
        filename = _decode_header(part.get("filename")) or ""
        body = part.get("body", {})
        attachment_id = body.get("attachmentId")

        if attachment_id:
            attachments.append(
                ParsedAttachment(
                    attachment_id=attachment_id,
                    filename=filename or "attachment",
                    mime_type=mime_type,
                    size=int(body.get("size", 0)),
                )
            )
        elif body.get("data"):
            decoded = _decode_base64url(body["data"])
            if mime_type == "text/plain":
                text_parts.append(decoded)
            elif mime_type == "text/html":
                html_parts.append(decoded)

        for child in part.get("parts", []):
            visit(child)

    visit(payload)
    internal_date = message.get("internalDate")
    received_at = (
        datetime.fromtimestamp(int(internal_date) / 1000, tz=timezone.utc)
        if internal_date
        else None
    )

    body_text = normalize_plain_text("\n".join(text_parts))
    body_html = "\n".join(html_parts).strip() or None
    if body_text is None:
        body_text, _ = html_to_text(body_html)
    parsed = ParsedEmail(
        provider_message_id=message["id"],
        provider_thread_id=message.get("threadId", message["id"]),
        internet_message_id=first_header("message-id"),
        from_address=first_header("from"),
        to_addresses=_parse_addresses(first_header("to")),
        cc_addresses=_parse_addresses(first_header("cc")),
        bcc_addresses=_parse_addresses(first_header("bcc")),
        subject=_decode_header(first_header("subject")),
        body_text=body_text,
        body_html=body_html,
        labels=message.get("labelIds", []),
        sent_at=_parse_date(first_header("date")),
        received_at=received_at,
        raw_payload=_strip_body_data(message),
        headers=headers,
        attachments=attachments,
    )
    parsed.spoof_analysis = analyze_spoofing(
        parsed.headers, parsed.from_address, parsed.body_text, parsed.body_html
    )
    parsed.safety_analysis = analyze_content_safety(
        parsed.body_text, parsed.attachments, parsed.body_html
    )
    return parsed
