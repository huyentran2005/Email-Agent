"""Hybrid search trên Qdrant, chỉ lập chỉ mục subject email."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from sqlalchemy import create_engine, select
from sqlalchemy.orm import load_only, sessionmaker

from core.config import Settings
from core.models import Email
from core.observability import LangfuseTracer


_TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)
DENSE_VECTOR_NAME = "subject_dense"
SPARSE_VECTOR_NAME = "subject_sparse"
logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class QdrantError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60
    max_attempts: int = 5
    retry_base_seconds: float = 2
    langfuse_enabled: bool = False
    langfuse_capture_content: bool = False


def _embeddings_url(base_url: str) -> str:
    """Chuẩn hóa base URL thành endpoint embeddings tương thích OpenAI."""
    base = base_url.strip().rstrip("/")
    if base.endswith("/embeddings"):
        return base
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")] + "/embeddings"
    if base.endswith(("/v1", "/openai")):
        return base + "/embeddings"
    return base + "/v1/embeddings"


class EmbeddingClient:
    """Client cho embeddings endpoint tương thích OpenAI."""

    def __init__(self, config: EmbeddingConfig, opener=urlopen, sleeper=time.sleep):
        """Khởi tạo client embedding và kiểm tra cấu hình bắt buộc."""
        if not config.base_url or not config.model:
            raise ValueError("EMBEDDING_BASE_URL and EMBEDDING_MODEL are required")
        self.config = config
        self.url = _embeddings_url(config.base_url)
        self._opener = opener
        self._sleeper = sleeper
        self.tracer = LangfuseTracer(config.langfuse_enabled)

    def _retry_delay(self, exc: HTTPError, detail: str, attempt: int) -> float:
        """Lấy thời gian chờ từ phản hồi 429 hoặc dùng exponential backoff."""
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
        match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", detail, re.I)
        if match:
            return min(60.0, max(0.0, float(match.group(1))))
        return min(
            60.0,
            self.config.retry_base_seconds * (2 ** max(0, attempt - 1)),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Chuyển danh sách văn bản đầu vào thành các dense vector."""
        if not texts:
            return []
        payload = {"model": self.config.model, "input": texts}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        trace_input = texts if self.config.langfuse_capture_content else {
            "text_count": len(texts),
            "total_chars": sum(len(text) for text in texts),
        }
        with self.tracer.observation(
            "email-subject.embedding",
            as_type="embedding",
            input=trace_input,
            model=self.config.model,
        ) as observation:
            request = Request(
                self.url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            envelope = None
            max_attempts = max(1, self.config.max_attempts)
            for attempt in range(1, max_attempts + 1):
                try:
                    with self._opener(
                        request, timeout=self.config.timeout_seconds
                    ) as response:
                        envelope = json.loads(response.read().decode("utf-8"))
                    break
                except HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")[:2000]
                    if exc.code != 429 or attempt >= max_attempts:
                        raise EmbeddingError(
                            f"Embedding HTTP {exc.code}: {detail}"
                        ) from exc
                    delay = self._retry_delay(exc, detail, attempt)
                    logger.warning(
                        "Embedding quota exceeded; retrying in %.1fs (%d/%d)",
                        delay,
                        attempt + 1,
                        max_attempts,
                    )
                    self._sleeper(delay)
                except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                    raise EmbeddingError(f"Embedding request failed: {exc}") from exc

            if not isinstance(envelope, dict):
                raise EmbeddingError("Embedding response is not a JSON object")

            usage = envelope.get("usage") if isinstance(envelope, dict) else None
            input_tokens = 0
            if isinstance(usage, dict):
                try:
                    input_tokens = int(
                        usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                    )
                except (TypeError, ValueError):
                    input_tokens = 0
            update: dict[str, Any] = {
                "output": {"vector_count": len(envelope.get("data") or [])}
            }
            if input_tokens:
                update["usage_details"] = {
                    "input": input_tokens,
                    "output": 0,
                    "total": input_tokens,
                }
            observation.update(**update)

        rows = envelope.get("data") if isinstance(envelope, dict) else None
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise EmbeddingError("Embedding response does not match input count")
        rows = sorted(rows, key=lambda item: item.get("index", 0))
        vectors: list[list[float]] = []
        for row in rows:
            vector = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError("Embedding response contains an invalid vector")
            vectors.append([float(value) for value in vector])
        return vectors


class QdrantRestClient:
    """REST client nhỏ gọn để không kéo thêm dependency runtime."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float = 30,
        opener=urlopen,
    ):
        """Khởi tạo REST client với URL, API key và thời gian chờ."""
        if not base_url.strip():
            raise ValueError("QDRANT_URL is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._opener = opener

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        """Gửi một REST request tới Qdrant và trả về trường result."""
        headers = {"Accept": "application/json"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.api_key:
            headers["api-key"] = self.api_key
        request = Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise QdrantError(f"Qdrant HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise QdrantError(f"Qdrant request failed: {exc}") from exc
        if not isinstance(envelope, dict) or "result" not in envelope:
            raise QdrantError("Qdrant returned an invalid response")
        return envelope["result"]

    def collection_info(self, collection: str) -> dict[str, Any] | None:
        """Đọc cấu hình collection hoặc trả về None nếu chưa tồn tại."""
        try:
            result = self._request("GET", f"/collections/{quote(collection, safe='')}")
        except QdrantError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        return result if isinstance(result, dict) else None

    def ensure_collection(self, collection: str, vector_size: int) -> None:
        """Tạo collection dense+sparse hoặc kiểm tra kích thước vector hiện có."""
        info = self.collection_info(collection)
        if info is None:
            self._request(
                "PUT",
                f"/collections/{quote(collection, safe='')}",
                {
                    "vectors": {
                        DENSE_VECTOR_NAME: {
                            "size": int(vector_size),
                            "distance": "Cosine",
                        }
                    },
                    "sparse_vectors": {
                        SPARSE_VECTOR_NAME: {"modifier": "idf"}
                    },
                },
            )
            return
        vectors = info.get("config", {}).get("params", {}).get("vectors", {})
        configured = vectors.get(DENSE_VECTOR_NAME, {}) if isinstance(vectors, dict) else {}
        if configured.get("size") != int(vector_size):
            raise QdrantError(
                "Qdrant collection has a different dense vector size; "
                "use a new QDRANT_COLLECTION when changing embedding dimensions"
            )

    def scroll_payloads(self, collection: str) -> dict[int, dict[str, Any]]:
        """Đọc toàn bộ point ID và payload để xác định dữ liệu cần cập nhật."""
        if self.collection_info(collection) is None:
            return {}
        results: dict[int, dict[str, Any]] = {}
        offset: Any = None
        while True:
            payload: dict[str, Any] = {
                "limit": 256,
                "with_payload": True,
                "with_vector": False,
            }
            if offset is not None:
                payload["offset"] = offset
            page = self._request(
                "POST",
                f"/collections/{quote(collection, safe='')}/points/scroll",
                payload,
            )
            for point in page.get("points", []):
                try:
                    point_id = int(point.get("id"))
                except (TypeError, ValueError):
                    continue
                results[point_id] = point.get("payload") or {}
            offset = page.get("next_page_offset")
            if offset is None:
                return results

    def point_payload(self, collection: str, point_id: int) -> dict[str, Any] | None:
        """Đọc payload của một point hoặc trả về None nếu chưa được index."""
        if self.collection_info(collection) is None:
            return None
        try:
            result = self._request(
                "GET",
                f"/collections/{quote(collection, safe='')}/points/{int(point_id)}",
            )
        except QdrantError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        return result.get("payload") if isinstance(result, dict) else None

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Thêm mới hoặc cập nhật một batch point trong Qdrant."""
        if points:
            self._request(
                "PUT",
                f"/collections/{quote(collection, safe='')}/points?wait=true",
                {"points": points},
            )

    def delete_points(self, collection: str, point_ids: list[int]) -> None:
        """Xóa các point không còn hợp lệ khỏi search index."""
        if point_ids:
            self._request(
                "POST",
                f"/collections/{quote(collection, safe='')}/points/delete?wait=true",
                {"points": point_ids},
            )

    def hybrid_query(
        self,
        collection: str,
        dense: list[float],
        sparse: dict[str, list[float] | list[int]],
        limit: int,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Tìm dense và sparse song song rồi hợp nhất kết quả bằng RRF."""
        candidate_limit = max(20, min(200, int(limit) * 5))
        prefetch = [
            {"query": dense, "using": DENSE_VECTOR_NAME, "limit": candidate_limit},
            {"query": sparse, "using": SPARSE_VECTOR_NAME, "limit": candidate_limit},
        ]
        if payload_filter:
            for request in prefetch:
                request["filter"] = payload_filter
        result = self._request(
            "POST",
            f"/collections/{quote(collection, safe='')}/points/query",
            {
                "prefetch": prefetch,
                "query": {"rrf": {}},
                "limit": max(1, min(int(limit), 50)),
                "with_payload": True,
                "with_vector": False,
            },
        )
        if isinstance(result, dict):
            return result.get("points", [])
        return result if isinstance(result, list) else []


def normalized_subject(subject: str | None) -> str:
    """Chuẩn hóa khoảng trắng trong subject mà không thay đổi nội dung."""
    return " ".join((subject or "").split())


def subject_hash(subject: str | None) -> str:
    """Tạo hash ổn định để phát hiện subject đã thay đổi."""
    return hashlib.sha256(normalized_subject(subject).encode("utf-8")).hexdigest()


def subjects_only(emails: Iterable[Email]) -> list[str]:
    """Tạo dense input chỉ từ subject; không truy cập body_text/body_html."""
    return [normalized_subject(email.subject) for email in emails]


def _token_id(token: str) -> int:
    """Chuyển token thành mã số 32-bit ổn định cho sparse vector."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big", signed=False)


def _lexical_token(token: str) -> str:
    """Chuẩn hóa token về chữ thường không dấu để tăng khả năng khớp."""
    normalized = unicodedata.normalize("NFKD", token.casefold()).replace("đ", "d")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def sparse_subject(text: str) -> dict[str, list[int] | list[float]]:
    """Sparse exact-term vector từ subject/query; Qdrant bổ sung IDF khi search."""
    counts = Counter(_lexical_token(token) for token in _TOKEN_PATTERN.findall(text))
    weighted = sorted(
        (_token_id(token), 1.0 + math.log(count)) for token, count in counts.items()
    )
    return {
        "indices": [item[0] for item in weighted],
        "values": [item[1] for item in weighted],
    }


def _is_suspicious(email: Email) -> bool:
    """Kiểm tra email có bị đánh dấu đáng ngờ hoặc DMARC thất bại không."""
    safety = email.safety_analysis or {}
    spoof = email.spoof_analysis or {}
    authentication = spoof.get("authentication") or {}
    return safety.get("verdict") == "suspicious" or authentication.get("dmarc") in {
        "fail",
        "permerror",
    }


def _embedding_client(settings: Settings) -> EmbeddingClient:
    """Tạo embedding client từ cấu hình ứng dụng."""
    return EmbeddingClient(
        EmbeddingConfig(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            timeout_seconds=settings.embedding_timeout_seconds,
            max_attempts=settings.embedding_max_attempts,
            retry_base_seconds=settings.embedding_retry_base_seconds,
            langfuse_enabled=settings.langfuse_enabled,
            langfuse_capture_content=settings.langfuse_capture_content,
        )
    )


def _qdrant_client(settings: Settings) -> QdrantRestClient:
    """Tạo Qdrant REST client từ cấu hình ứng dụng."""
    return QdrantRestClient(
        settings.qdrant_url,
        settings.qdrant_api_key,
        settings.qdrant_timeout_seconds,
    )


def _qdrant_point(email: Email, subject: str, dense: list[float], model: str) -> dict[str, Any]:
    """Tạo Qdrant point chỉ chứa vector subject và metadata định tuyến."""
    return {
        "id": email.id,
        "vector": {
            DENSE_VECTOR_NAME: dense,
            SPARSE_VECTOR_NAME: sparse_subject(subject),
        },
        "payload": {
            "email_id": email.id,
            "account_id": email.account_id,
            "thread_id": email.thread_id,
            "subject": subject,
            "from_address": email.from_address,
            "received_at": _iso(email.received_at or email.created_at),
            "labels": email.labels or [],
            "subject_hash": subject_hash(subject),
            "embedding_model": model,
        },
    }


class SubjectEmbeddingIndexer:
    """Đồng bộ dense+sparse subject index từ PostgreSQL sang Qdrant."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: EmbeddingClient | None = None,
        qdrant: QdrantRestClient | None = None,
    ):
        """Khởi tạo indexer cùng kết nối PostgreSQL, embedding và Qdrant."""
        self.settings = settings or Settings()
        self.client = client or _embedding_client(self.settings)
        self.qdrant = qdrant or _qdrant_client(self.settings)
        self.engine = create_engine(self.settings.database_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def index_email(self, email_id: int) -> str:
        """Index ngay một email sau khi PostgreSQL commit và tránh gọi lại khi subject chưa đổi."""
        with self.session_factory() as session:
            email = session.get(Email, int(email_id))
            if email is None:
                return "missing"
            subject = normalized_subject(email.subject)
            if not subject or _is_suspicious(email):
                if self.qdrant.collection_info(self.settings.qdrant_collection) is not None:
                    self.qdrant.delete_points(self.settings.qdrant_collection, [email.id])
                return "skipped"
            existing = self.qdrant.point_payload(
                self.settings.qdrant_collection, email.id
            )
            if (
                existing
                and existing.get("subject_hash") == subject_hash(subject)
                and existing.get("embedding_model") == self.settings.embedding_model
            ):
                return "unchanged"
            dense = self.client.embed([subject])[0]
            self.qdrant.ensure_collection(self.settings.qdrant_collection, len(dense))
            self.qdrant.upsert(
                self.settings.qdrant_collection,
                [_qdrant_point(email, subject, dense, self.settings.embedding_model)],
            )
            return "indexed"

    def index_pending(self, limit: int | None = None) -> dict[str, int]:
        """Đồng bộ các subject mới hoặc thay đổi từ PostgreSQL sang Qdrant."""
        batch_size = max(1, min(self.settings.embedding_batch_size, 256))
        existing = self.qdrant.scroll_payloads(self.settings.qdrant_collection)
        with self.session_factory() as session:
            emails = list(
                session.scalars(
                    select(Email)
                    .where(
                        Email.processing_status == "processed",
                        Email.subject.is_not(None),
                        Email.subject != "",
                    )
                    .order_by(Email.id)
                )
            )
            eligible = [email for email in emails if not _is_suspicious(email)]
            eligible_ids = {email.id for email in eligible}
            pending = [
                email
                for email in eligible
                if email.id not in existing
                or existing[email.id].get("subject_hash") != subject_hash(email.subject)
                or existing[email.id].get("embedding_model") != self.settings.embedding_model
            ]
            if limit is not None:
                pending = pending[: max(0, int(limit))]

            indexed = 0
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                subjects = subjects_only(batch)
                dense_vectors = self.client.embed(subjects)
                if dense_vectors:
                    self.qdrant.ensure_collection(
                        self.settings.qdrant_collection, len(dense_vectors[0])
                    )
                points = [
                    _qdrant_point(email, subject, dense, self.settings.embedding_model)
                    for email, subject, dense in zip(batch, subjects, dense_vectors)
                ]
                self.qdrant.upsert(self.settings.qdrant_collection, points)
                indexed += len(points)

        stale_ids = sorted(set(existing) - eligible_ids)
        if existing:
            self.qdrant.delete_points(self.settings.qdrant_collection, stale_ids)
        return {
            "eligible": len(eligible_ids),
            "pending": len(pending),
            "indexed": indexed,
            "removed": len(stale_ids),
        }


class SubjectHybridSearch:
    """Dense+sparse RRF trên subject; đọc body từ PostgreSQL sau ranking."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: EmbeddingClient | None = None,
        qdrant: QdrantRestClient | None = None,
    ):
        """Khởi tạo dịch vụ hybrid search và kết nối dữ liệu cần thiết."""
        self.settings = settings or Settings()
        self.client = client or _embedding_client(self.settings)
        self.qdrant = qdrant or _qdrant_client(self.settings)
        self.engine = create_engine(self.settings.database_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def search(
        self,
        query: str,
        limit: int = 12,
        received_at_from: datetime | None = None,
        received_at_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Tìm subject bằng RRF trong khoảng ngày rồi đọc email từ PostgreSQL."""
        query = query.strip()
        if not query:
            return []
        date_range = {}
        if received_at_from is not None:
            date_range["gte"] = received_at_from.isoformat()
        if received_at_to is not None:
            date_range["lt"] = received_at_to.isoformat()
        payload_filter = (
            {"must": [{"key": "received_at", "range": date_range}]}
            if date_range
            else None
        )
        dense = self.client.embed([query])[0]
        hits = self.qdrant.hybrid_query(
            self.settings.qdrant_collection,
            dense,
            sparse_subject(query),
            limit,
            payload_filter=payload_filter,
        )
        ranked_ids = []
        scores: dict[int, float] = {}
        for hit in hits:
            payload = hit.get("payload") or {}
            raw_id = payload.get("email_id", hit.get("id"))
            try:
                email_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            ranked_ids.append(email_id)
            scores[email_id] = float(hit.get("score", 0.0))
        if not ranked_ids:
            return []

        with self.session_factory() as session:
            statement = (
                select(Email)
                .options(
                    load_only(
                        Email.id,
                        Email.thread_id,
                        Email.subject,
                        Email.from_address,
                        Email.received_at,
                        Email.created_at,
                        Email.processing_status,
                        Email.safety_analysis,
                        Email.spoof_analysis,
                    )
                )
                .where(
                    Email.id.in_(ranked_ids),
                    Email.processing_status == "processed",
                )
            )
            if received_at_from is not None:
                statement = statement.where(Email.received_at >= received_at_from)
            if received_at_to is not None:
                statement = statement.where(Email.received_at < received_at_to)
            emails = list(
                session.scalars(statement)
            )
            by_id = {email.id: email for email in emails if not _is_suspicious(email)}
            return [
                {
                    "email_id": email.id,
                    "thread_id": email.thread_id,
                    "subject": email.subject or "(Không có tiêu đề)",
                    "from": email.from_address or "(Không rõ người gửi)",
                    "received_at": _iso(email.received_at or email.created_at),
                    "hybrid_score": round(scores[email.id], 6),
                    "body_excerpt": (email.body_text or "")[:1200],
                }
                for email_id in ranked_ids
                if (email := by_id.get(email_id)) is not None
            ]


def _iso(value: datetime | None) -> str | None:
    """Chuyển datetime thành chuỗi ISO hoặc giữ None."""
    return value.isoformat() if value else None
