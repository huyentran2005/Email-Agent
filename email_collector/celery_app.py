"""Cấu hình Celery dùng Redis cho các tác vụ xử lý LLM."""

from celery import Celery

from .config import Settings


settings = Settings()

celery_app = Celery(
    "email_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["email_collector.llm_queue"],
)

celery_app.conf.update(
    task_default_queue=settings.celery_queue_name,
    task_routes={
        "email_collector.process_thread": {"queue": settings.celery_queue_name}
    },
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    enable_utc=True,
    timezone="UTC",
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    result_expires=settings.celery_result_expires,
    visibility_timeout=settings.celery_visibility_timeout,
    broker_transport_options={
        "visibility_timeout": settings.celery_visibility_timeout,
        "global_keyprefix": settings.celery_key_prefix,
    },
    result_backend_transport_options={
        "visibility_timeout": settings.celery_visibility_timeout,
        "global_keyprefix": settings.celery_key_prefix,
        "retry_policy": {"timeout": 5.0},
    },
)

