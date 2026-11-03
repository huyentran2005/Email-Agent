"""Khai báo Celery task và lớp gửi job xử lý thread vào Redis."""

import logging
from collections.abc import Iterable
from functools import lru_cache

from celery import Task
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from core.config import Settings
from core.models import Email
from email_filtering_understanding.llm_processing import ThreadLLMProcessor

from .celery_app import celery_app

logger = logging.getLogger(__name__)
settings = Settings()


@lru_cache(maxsize=1)
def _processor() -> ThreadLLMProcessor:
    """Khởi tạo một bộ xử lý LLM dùng lại trong mỗi tiến trình Celery worker."""
    processor = ThreadLLMProcessor(settings)
    processor.initialize()
    return processor


@celery_app.task(
    bind=True,
    name="workers.process_thread",
    max_retries=settings.celery_task_max_retries,
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_thread_task(task: Task, thread_id: int) -> dict[str, int]:
    """Xử lý một thread và retry bằng exponential backoff khi có lỗi."""
    try:
        result = _processor().process_thread(thread_id)
        logger.info("Completed Celery LLM task thread=%s result=%s", thread_id, result)
        return result
    except Exception as exc:
        countdown = settings.celery_task_retry_backoff * (2 ** task.request.retries)
        logger.exception(
            "Celery LLM task failed for thread=%s; retry=%s/%s in %ss",
            thread_id,
            task.request.retries + 1,
            settings.celery_task_max_retries,
            countdown,
        )
        raise task.retry(exc=exc, countdown=countdown) from exc


class CeleryJobQueue:
    """Tìm thread trong PostgreSQL và gửi task nhỏ gọn vào Celery/Redis."""

    def __init__(
        self,
        app_settings: Settings | None = None,
        task: Task | None = None,
    ):
        """Khởi tạo truy vấn email và task publisher có thể thay thế khi kiểm thử."""
        self.settings = app_settings or Settings()
        self.task = task or process_thread_task
        self.engine = create_engine(self.settings.database_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def enqueue_email_ids(self, email_ids: list[int]) -> int:
        """Tìm các thread của danh sách email rồi gửi mỗi thread một Celery task."""
        if not email_ids:
            return 0
        with self.session_factory() as session:
            thread_ids = session.scalars(
                select(Email.thread_id)
                .where(Email.id.in_(email_ids), Email.thread_id.is_not(None))
                .distinct()
                .order_by(Email.thread_id)
            ).all()
        return self.enqueue_thread_ids(thread_ids)

    def enqueue_all_threads(self) -> int:
        """Gửi một Celery task cho mỗi thread đã tồn tại để backfill dữ liệu."""
        with self.session_factory() as session:
            thread_ids = session.scalars(
                select(Email.thread_id)
                .where(Email.thread_id.is_not(None))
                .distinct()
                .order_by(Email.thread_id)
            ).all()
        return self.enqueue_thread_ids(thread_ids)

    def enqueue_thread_ids(self, thread_ids: Iterable[int | None]) -> int:
        """Loại ID trùng và publish các thread vào queue Redis đã cấu hình."""
        unique_ids = list(
            dict.fromkeys(int(item) for item in thread_ids if item is not None)
        )
        for thread_id in unique_ids:
            self.task.apply_async(
                args=[thread_id],
                queue=self.settings.celery_queue_name,
                retry=True,
                retry_policy={
                    "max_retries": 3,
                    "interval_start": 0,
                    "interval_step": 1,
                    "interval_max": 3,
                },
            )
        return len(unique_ids)

