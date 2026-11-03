from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from core.models import (
    Base,
    Email,
    EmailAccount,
    EmailAttachment,
    EmailSyncRun,
    EmailThread,
)
from .parser import ParsedEmail


@dataclass
class SaveResult:
    email_id: int
    status: str


class EmailRepository:
    def __init__(self, database_url: str):
        """Khởi tạo SQLAlchemy engine và session factory cho PostgreSQL."""
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        """Tạo schema trực tiếp từ SQLAlchemy models."""
        Base.metadata.create_all(self.engine)

    def get_or_create_account(
        self, session: Session, email_address: str, history_id: str | None = None
    ) -> EmailAccount:
        """Lấy tài khoản theo email hoặc tạo bản ghi tài khoản Gmail mới."""
        account = session.scalar(
            select(EmailAccount).where(EmailAccount.email_address == email_address)
        )
        if account is None:
            account = EmailAccount(
                provider="gmail",
                email_address=email_address,
                latest_history_id=history_id,
            )
            session.add(account)
            session.flush()
        return account

    @staticmethod
    def create_sync_run(
        session: Session, account_id: int, mode: str
    ) -> EmailSyncRun:
        """Tạo bản ghi audit cho một đợt đồng bộ đang bắt đầu."""
        run = EmailSyncRun(account_id=account_id, mode=mode, status="running")
        session.add(run)
        session.flush()
        return run

    @staticmethod
    def save_email(
        session: Session, account: EmailAccount, parsed: ParsedEmail
    ) -> SaveResult:
        """Upsert thread và email đã parse, sau đó trả về ID cùng trạng thái lưu."""
        thread = session.scalar(
            select(EmailThread).where(
                EmailThread.account_id == account.id,
                EmailThread.provider_thread_id == parsed.provider_thread_id,
            )
        )
        if thread is None:
            thread = EmailThread(
                account_id=account.id,
                provider_thread_id=parsed.provider_thread_id,
            )
            session.add(thread)
            session.flush()

        thread.subject = parsed.subject
        if parsed.received_at and (
            thread.last_message_at is None
            or parsed.received_at > thread.last_message_at
        ):
            thread.last_message_at = parsed.received_at

        email = session.scalar(
            select(Email).where(
                Email.account_id == account.id,
                Email.provider_message_id == parsed.provider_message_id,
            )
        )
        status = "inserted" if email is None else "updated"
        if email is None:
            email = Email(
                account_id=account.id,
                provider_message_id=parsed.provider_message_id,
            )
            session.add(email)

        email.thread_id = thread.id
        email.internet_message_id = parsed.internet_message_id
        email.from_address = parsed.from_address
        email.to_addresses = parsed.to_addresses
        email.cc_addresses = parsed.cc_addresses
        email.bcc_addresses = parsed.bcc_addresses
        email.subject = parsed.subject
        email.body_text = parsed.body_text
        email.body_html = parsed.body_html
        email.labels = parsed.labels
        email.sent_at = parsed.sent_at
        email.received_at = parsed.received_at
        email.raw_payload = parsed.raw_payload
        email.spoof_analysis = parsed.spoof_analysis
        email.safety_analysis = parsed.safety_analysis
        email.processing_status = "security_checked"
        session.flush()
        return SaveResult(email_id=email.id, status=status)

    @staticmethod
    def save_attachment(
        session: Session,
        email_id: int,
        attachment_id: str,
        filename: str,
        mime_type: str,
        size: int,
        bucket: str,
        storage_key: str,
        content_hash: str,
        security_status: str,
        security_report: dict,
    ) -> None:
        """Upsert metadata attachment và vị trí object tương ứng trong MinIO."""
        attachment = session.scalar(
            select(EmailAttachment).where(
                EmailAttachment.email_id == email_id,
                EmailAttachment.provider_attachment_id == attachment_id,
            )
        )
        if attachment is None:
            attachment = EmailAttachment(
                email_id=email_id,
                provider_attachment_id=attachment_id,
            )
            session.add(attachment)
        attachment.filename = filename
        attachment.mime_type = mime_type
        attachment.size = size
        attachment.storage_bucket = bucket
        attachment.storage_key = storage_key
        attachment.content_hash = content_hash
        attachment.security_status = security_status
        attachment.security_report = security_report

    @staticmethod
    def complete_sync(
        run: EmailSyncRun,
        account: EmailAccount,
        history_id: str,
        collected: int,
        inserted: int,
        updated: int,
        skipped: int,
    ) -> None:
        """Đánh dấu đợt đồng bộ thành công và cập nhật checkpoint của tài khoản."""
        now = datetime.now(timezone.utc)
        run.status = "success"
        run.collected_count = collected
        run.inserted_count = inserted
        run.updated_count = updated
        run.skipped_count = skipped
        run.completed_at = now
        account.latest_history_id = history_id
        account.last_synced_at = now

    @staticmethod
    def fail_sync(run: EmailSyncRun, error: Exception) -> None:
        """Đánh dấu đợt đồng bộ thất bại và lưu thông tin lỗi rút gọn."""
        run.status = "failed"
        run.error = str(error)[:5000]
        run.completed_at = datetime.now(timezone.utc)
