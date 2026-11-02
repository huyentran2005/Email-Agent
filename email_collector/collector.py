import base64
import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from googleapiclient.errors import HttpError

from .config import Settings
from .object_storage import AttachmentStorage
from .parser import ParsedEmail, parse_gmail_message
from .repository import EmailRepository

logger = logging.getLogger(__name__)


@dataclass
class CollectorResult:
    status: str
    mode: str
    account: str
    collected: int
    inserted: int
    updated: int
    skipped: int
    email_ids: list[int]
    next_history_id: str

    def to_dict(self) -> dict[str, Any]:
        """Chuyển kết quả collector từ dataclass thành dictionary."""
        return asdict(self)


class EmailCollector:
    def __init__(self, gmail_service, settings: Settings | None = None):
        """Khởi tạo collector cùng Gmail API, repository và MinIO storage."""
        self.gmail = gmail_service
        self.settings = settings or Settings()
        self.repository = EmailRepository(self.settings.database_url)
        self.storage = AttachmentStorage(self.settings)

    def initialize(self) -> None:
        """Chuẩn bị schema PostgreSQL và bucket MinIO trước khi đồng bộ."""
        self.repository.create_schema()
        self.storage.ensure_bucket()

    def sync(self, force_initial: bool = False) -> CollectorResult:
        """Đồng bộ Gmail theo chế độ ban đầu hoặc tăng dần rồi trả về thống kê."""
        profile = self.gmail.users().getProfile(userId="me").execute()
        email_address = profile["emailAddress"]
        current_history_id = profile["historyId"]

        with self.repository.session_factory() as session:
            account = self.repository.get_or_create_account(session, email_address)
            session.commit()
            previous_history_id = account.latest_history_id

        mode = "initial" if force_initial or not previous_history_id else "incremental"
        message_ids = (
            self._list_initial_message_ids()
            if mode == "initial"
            else self._list_changed_message_ids(previous_history_id)
        )

        inserted = 0
        updated = 0
        skipped = 0
        stored_ids: list[int] = []

        with self.repository.session_factory() as session:
            account = self.repository.get_or_create_account(session, email_address)
            sync_run = self.repository.create_sync_run(session, account.id, mode)
            session.commit()

            try:
                for message_id in message_ids:
                    parsed = self._get_and_parse(message_id)
                    if parsed is None:
                        skipped += 1
                        continue
                    save_result = self.repository.save_email(session, account, parsed)
                    self._store_attachments(session, save_result.email_id, parsed)
                    session.commit()

                    stored_ids.append(save_result.email_id)
                    if save_result.status == "inserted":
                        inserted += 1
                    else:
                        updated += 1

                # Keep the snapshot taken before collection. Changes that happen
                # during this run are therefore picked up by the next history sync.
                next_history_id = current_history_id
                self.repository.complete_sync(
                    sync_run,
                    account,
                    next_history_id,
                    len(message_ids),
                    inserted,
                    updated,
                    skipped,
                )
                session.commit()
            except Exception as exc:
                session.rollback()
                sync_run = session.get(type(sync_run), sync_run.id)
                if sync_run is not None:
                    self.repository.fail_sync(sync_run, exc)
                    session.commit()
                raise

        return CollectorResult(
            status="success",
            mode=mode,
            account=email_address,
            collected=len(message_ids),
            inserted=inserted,
            updated=updated,
            skipped=skipped,
            email_ids=stored_ids,
            next_history_id=next_history_id,
        )

    def _list_initial_message_ids(self) -> list[str]:
        """Lấy ID email Inbox trong số ngày cấu hình cho lần đồng bộ ban đầu."""
        query = f"newer_than:{self.settings.initial_sync_days}d"
        return self._paginate_messages(query=query)

    def _paginate_messages(self, query: str) -> list[str]:
        """Duyệt toàn bộ trang kết quả messages.list và loại ID bị trùng."""
        message_ids: set[str] = set()
        page_token = None
        while True:
            response = (
                self.gmail.users()
                .messages()
                .list(
                    userId="me",
                    q=query,
                    labelIds=["INBOX"],
                    maxResults=self.settings.sync_page_size,
                    pageToken=page_token,
                )
                .execute()
            )
            message_ids.update(item["id"] for item in response.get("messages", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return sorted(message_ids)

    def _list_changed_message_ids(self, history_id: str) -> list[str]:
        """Lấy ID email thay đổi kể từ checkpoint Gmail History đã lưu."""
        message_ids: set[str] = set()
        page_token = None
        try:
            while True:
                response = (
                    self.gmail.users()
                    .history()
                    .list(
                        userId="me",
                        startHistoryId=history_id,
                        labelId="INBOX",
                        maxResults=self.settings.sync_page_size,
                        pageToken=page_token,
                    )
                    .execute()
                )
                for history in response.get("history", []):
                    for message in history.get("messages", []):
                        message_ids.add(message["id"])
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as exc:
            if exc.resp.status == 404:
                logger.warning("Gmail history expired; falling back to initial sync")
                return self._list_initial_message_ids()
            raise
        return sorted(message_ids)

    def _get_and_parse(self, message_id: str) -> ParsedEmail | None:
        """Tải một Gmail message và parse; trả về None nếu email đã bị xóa."""
        try:
            message = (
                self.gmail.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except HttpError as exc:
            if exc.resp.status == 404:
                logger.warning("Skipping Gmail message that no longer exists: %s", message_id)
                return None
            raise
        return parse_gmail_message(message)

    def _store_attachments(self, session, email_id: int, parsed: ParsedEmail) -> None:
        """Tải attachment, lưu vào MinIO và ghi metadata vào PostgreSQL."""
        for attachment in parsed.attachments:
            response = (
                self.gmail.users()
                .messages()
                .attachments()
                .get(
                    userId="me",
                    messageId=parsed.provider_message_id,
                    id=attachment.attachment_id,
                )
                .execute()
            )
            encoded = response.get("data", "")
            data = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            digest = hashlib.sha256(data).hexdigest()
            safe_name = PurePosixPath(attachment.filename).name or "attachment"
            storage_key = (
                f"gmail/{parsed.provider_message_id}/"
                f"{attachment.attachment_id}-{safe_name}"
            )
            self.storage.put(storage_key, data, attachment.mime_type)
            self.repository.save_attachment(
                session=session,
                email_id=email_id,
                attachment_id=attachment.attachment_id,
                filename=attachment.filename,
                mime_type=attachment.mime_type,
                size=len(data),
                bucket=self.storage.bucket,
                storage_key=storage_key,
                content_hash=digest,
            )
