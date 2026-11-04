"""Celery task tạo subject embedding và lưu dense+sparse vector vào Qdrant."""

import logging
from collections.abc import Iterable
from functools import lru_cache

from celery import Task
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from core.config import Settings
from core.models import Email
from knowledge_retrieval.subject_embeddings import SubjectEmbeddingIndexer

from .celery_app import celery_app

logger = logging.getLogger(__name__)
settings = Settings()


@lru_cache(maxsize=1)
def _indexer() -> SubjectEmbeddingIndexer:
    """Khởi tạo một subject indexer dùng lại trong tiến trình Celery worker."""
    return SubjectEmbeddingIndexer(settings)


@celery_app.task(
    bind=True,
    name="workers.index_email_subject",
    max_retries=settings.celery_task_max_retries,
    acks_late=True,
    reject_on_worker_lost=True,
)
def index_email_subject_task(task: Task, email_id: int) -> dict[str, int | str]:
    """Tạo embedding cho một subject và retry exponential backoff khi lỗi."""
    try:
        status = _indexer().index_email(int(email_id))
        result = {"email_id": int(email_id), "status": status}
        logger.info("Completed subject embedding task: %s", result)
        return result
    except Exception as exc:
        countdown = settings.celery_task_retry_backoff * (2 ** task.request.retries)
        logger.exception(
            "Subject embedding task failed email_id=%s; retry=%s/%s in %ss",
            email_id,
            task.request.retries + 1,
            settings.celery_task_max_retries,
            countdown,
        )
        raise task.retry(exc=exc, countdown=countdown) from exc


class EmbeddingJobQueue:
    """Publish job embedding theo email ID vào queue Redis riêng."""

    def __init__(
        self,
        app_settings: Settings | None = None,
        task: Task | None = None,
    ):
        """Khởi tạo publisher có thể thay task giả khi unit test."""
        self.settings = app_settings or Settings()
        self.task = task or index_email_subject_task
        self.engine = create_engine(self.settings.database_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def enqueue_email_ids(self, email_ids: Iterable[int | None]) -> int:
        """Loại ID trùng rồi publish một embedding job cho mỗi email."""
        if not self.settings.embedding_jobs_enabled:
            return 0
        unique_ids = list(
            dict.fromkeys(int(item) for item in email_ids if item is not None)
        )
        for email_id in unique_ids:
            self.task.apply_async(
                args=[email_id],
                queue=self.settings.celery_embedding_queue_name,
                retry=True,
                retry_policy={
                    "max_retries": 3,
                    "interval_start": 0,
                    "interval_step": 1,
                    "interval_max": 3,
                },
            )
        return len(unique_ids)

    def enqueue_all_emails(self) -> int:
        """Publish embedding job cho mọi email để backfill hoặc sửa index."""
        with self.session_factory() as session:
            email_ids = session.scalars(
                select(Email.id)
                .where(Email.subject.is_not(None), Email.subject != "")
                .order_by(Email.id)
            ).all()
        return self.enqueue_email_ids(email_ids)
