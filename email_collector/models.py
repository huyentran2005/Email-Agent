from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class EmailAccount(Base):
    __tablename__ = "email_accounts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    provider: Mapped[str] = mapped_column(String(30), default="gmail")
    email_address: Mapped[str] = mapped_column(String(320), unique=True)
    latest_history_id: Mapped[str | None] = mapped_column(String(255))
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    threads: Mapped[list["EmailThread"]] = relationship(back_populates="account")
    emails: Mapped[list["Email"]] = relationship(back_populates="account")


class EmailThread(Base):
    __tablename__ = "email_threads"
    __table_args__ = (
        UniqueConstraint("account_id", "provider_thread_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("email_accounts.id"))
    provider_thread_id: Mapped[str] = mapped_column(String(255))
    subject: Mapped[str | None] = mapped_column(Text)
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    account: Mapped[EmailAccount] = relationship(back_populates="threads")
    emails: Mapped[list["Email"]] = relationship(back_populates="thread")


class Email(Base):
    __tablename__ = "emails"
    __table_args__ = (
        UniqueConstraint("account_id", "provider_message_id"),
        Index("ix_emails_received_at", "received_at"),
        Index("ix_emails_processing_status", "processing_status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("email_accounts.id"))
    thread_id: Mapped[int | None] = mapped_column(ForeignKey("email_threads.id"))
    provider_message_id: Mapped[str] = mapped_column(String(255))
    internet_message_id: Mapped[str | None] = mapped_column(String(998))
    from_address: Mapped[str | None] = mapped_column(Text)
    to_addresses: Mapped[list[dict[str, str]]] = mapped_column(JSONB, default=list)
    cc_addresses: Mapped[list[dict[str, str]]] = mapped_column(JSONB, default=list)
    bcc_addresses: Mapped[list[dict[str, str]]] = mapped_column(JSONB, default=list)
    subject: Mapped[str | None] = mapped_column(Text)
    body_text: Mapped[str | None] = mapped_column(Text)
    body_html: Mapped[str | None] = mapped_column(Text)
    labels: Mapped[list[str]] = mapped_column(JSONB, default=list)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    spoof_analysis: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    safety_analysis: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    processing_status: Mapped[str] = mapped_column(String(30), default="security_checked")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    account: Mapped[EmailAccount] = relationship(back_populates="emails")
    thread: Mapped[EmailThread | None] = relationship(back_populates="emails")
    attachments: Mapped[list["EmailAttachment"]] = relationship(
        back_populates="email", cascade="all, delete-orphan"
    )


class EmailAttachment(Base):
    __tablename__ = "email_attachments"
    __table_args__ = (
        UniqueConstraint("email_id", "provider_attachment_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"))
    provider_attachment_id: Mapped[str] = mapped_column(Text)
    filename: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(String(255))
    size: Mapped[int] = mapped_column(BigInteger, default=0)
    storage_bucket: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    security_status: Mapped[str] = mapped_column(String(30), default="unscanned")
    security_report: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    email: Mapped[Email] = relationship(back_populates="attachments")


class EmailSyncRun(Base):
    __tablename__ = "email_sync_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("email_accounts.id"))
    mode: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(30), default="running")
    collected_count: Mapped[int] = mapped_column(Integer, default=0)
    inserted_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
