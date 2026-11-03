from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
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

class ThreadEventState(Base):
    __tablename__ = "thread_event_states"
    __table_args__ = (
        UniqueConstraint("thread_id", "canonical_key"),
        Index("ix_thread_event_states_status", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    thread_id: Mapped[int] = mapped_column(ForeignKey("email_threads.id"))
    canonical_key: Mapped[str] = mapped_column(String(500))
    event_type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(30), default="open")
    current_value: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)
    updated_by_email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class EmailEvent(Base):
    __tablename__ = "email_events"
    __table_args__ = (
        UniqueConstraint("source_email_id", "event_index"),
        Index("ix_email_events_type", "event_type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    thread_id: Mapped[int] = mapped_column(ForeignKey("email_threads.id"))
    source_email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"))
    event_state_id: Mapped[int | None] = mapped_column(ForeignKey("thread_event_states.id"))
    event_index: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(50))
    trigger: Mapped[str | None] = mapped_column(Text)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    extraction_model: Mapped[str] = mapped_column(String(255))
    extractor_version: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EventChange(Base):
    __tablename__ = "event_changes"
    __table_args__ = (Index("ix_event_changes_source_email", "source_email_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    event_state_id: Mapped[int] = mapped_column(ForeignKey("thread_event_states.id"))
    source_email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"))
    change_type: Mapped[str] = mapped_column(String(30))
    before_value: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    after_value: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ThreadSummary(Base):
    __tablename__ = "thread_summaries"
    __table_args__ = (
        UniqueConstraint("thread_id", "summary_type", "version"),
        Index("ix_thread_summaries_thread_type", "thread_id", "summary_type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    thread_id: Mapped[int] = mapped_column(ForeignKey("email_threads.id"))
    summary_type: Mapped[str] = mapped_column(String(20))
    summary_text: Mapped[str] = mapped_column(Text)
    based_on_email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"))
    supporting_email_ids: Mapped[list[int]] = mapped_column(JSONB, default=list)
    verification: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(Integer, default=1)
    model: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class LLMRun(Base):
    __tablename__ = "llm_runs"
    __table_args__ = (
        UniqueConstraint("source_email_id", "task", "prompt_version"),
        Index("ix_llm_runs_status", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    thread_id: Mapped[int] = mapped_column(ForeignKey("email_threads.id"))
    source_email_id: Mapped[int | None] = mapped_column(ForeignKey("emails.id"))
    task: Mapped[str] = mapped_column(String(40))
    prompt_version: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(255))
    request_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="running")
    response_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
