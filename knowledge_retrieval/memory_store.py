"""Bộ nhớ hội thoại bền vững dùng PostgreSQL và truy xuất từ khóa cục bộ."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from core.models import ConversationMemory


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold()).replace("đ", "d")
    return " ".join(
        "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        ).split()
    )


def _search_terms(query: str) -> list[str]:
    """Lấy các term đủ ý nghĩa để xếp hạng ký ức mà không cần vector DB."""
    stop_words = {
        "ban",
        "biet",
        "con",
        "da",
        "gi",
        "hoi",
        "khong",
        "lan",
        "nho",
        "noi",
        "toi",
        "truoc",
        "tung",
    }
    return list(
        dict.fromkeys(
            token
            for token in re.findall(r"[a-z0-9@._/-]{2,}", _normalized_text(query))
            if token not in stop_words
        )
    )[:12]


class ConversationMemoryStore:
    """Tạo schema, ghi lượt hội thoại và tìm ký ức trong một namespace."""

    def __init__(
        self,
        engine: Engine,
        session_factory: sessionmaker,
        namespace: str = "default",
    ):
        self.engine = engine
        self.session_factory = session_factory
        self.namespace = namespace.strip()[:100] or "default"

    def initialize(self) -> None:
        """Chỉ tạo bảng bộ nhớ, không tác động các bảng nghiệp vụ khác."""
        ConversationMemory.__table__.create(bind=self.engine, checkfirst=True)

    def remember(
        self,
        user_message: str,
        assistant_message: str,
        source_email_ids: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Ghi một lượt hoàn tất; cùng nội dung trong namespace chỉ ghi một lần."""
        user_message = user_message.strip()
        assistant_message = assistant_message.strip()
        if not user_message or not assistant_message:
            return False
        digest = hashlib.sha256(
            f"{self.namespace}\0{user_message}\0{assistant_message}".encode("utf-8")
        ).hexdigest()
        memory = ConversationMemory(
            namespace=self.namespace,
            memory_hash=digest,
            user_message=user_message,
            assistant_message=assistant_message,
            source_email_ids=list(dict.fromkeys(source_email_ids or [])),
            memory_metadata=metadata or {},
        )
        with self.session_factory() as session:
            session.add(memory)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return False
        return True

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Xếp hạng tối đa 200 ký ức gần nhất theo term và độ mới."""
        limit = max(1, min(int(limit), 20))
        with self.session_factory() as session:
            memories = list(
                session.scalars(
                    select(ConversationMemory)
                    .where(ConversationMemory.namespace == self.namespace)
                    .order_by(
                        ConversationMemory.created_at.desc(),
                        ConversationMemory.id.desc(),
                    )
                    .limit(200)
                )
            )
        terms = _search_terms(query)
        ranked: list[tuple[int, datetime, ConversationMemory]] = []
        for memory in memories:
            searchable = _normalized_text(
                f"{memory.user_message} {memory.assistant_message}"
            )
            score = sum(searchable.count(term) for term in terms)
            if terms and score == 0:
                continue
            ranked.append((score, memory.created_at, memory))
        ranked.sort(key=lambda item: (item[0], item[1], item[2].id), reverse=True)
        return [
            {
                "memory_id": memory.id,
                "user_message": memory.user_message[:2000],
                "assistant_message": memory.assistant_message[:4000],
                "source_email_ids": memory.source_email_ids or [],
                "created_at": memory.created_at.isoformat(),
            }
            for _, _, memory in ranked[:limit]
        ]
