from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str) -> bool:
    """Chuyển chuỗi cấu hình môi trường thành giá trị boolean."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://email_agent:email_agent_dev_password@localhost:5432/email_agent",
    )
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9100")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv(
        "MINIO_SECRET_KEY", "minioadmin_dev_password"
    )
    minio_bucket: str = os.getenv("MINIO_BUCKET", "email-attachments")
    minio_secure: bool = _as_bool(os.getenv("MINIO_SECURE", "false"))
    clamav_enabled: bool = _as_bool(os.getenv("CLAMAV_ENABLED", "true"))
    clamav_host: str = os.getenv("CLAMAV_HOST", "localhost")
    clamav_port: int = int(os.getenv("CLAMAV_PORT", "3310"))
    clamav_timeout_seconds: float = float(os.getenv("CLAMAV_TIMEOUT_SECONDS", "30"))
    llm_proxy_base_url: str = os.getenv("LLM_PROXY_BASE_URL", "")
    llm_proxy_api_key: str = os.getenv("LLM_PROXY_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    llm_max_attempts: int = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
    llm_json_mode: bool = _as_bool(os.getenv("LLM_JSON_MODE", "true"))
    llm_allow_suspicious_email: bool = _as_bool(
        os.getenv("LLM_ALLOW_SUSPICIOUS_EMAIL", "false")
    )
    llm_verify_summary: bool = _as_bool(os.getenv("LLM_VERIFY_SUMMARY", "true"))
    # Semantic search chỉ gửi subject email tới embeddings endpoint.
    embedding_base_url: str = os.getenv("EMBEDDING_BASE_URL") or os.getenv(
        "LLM_PROXY_BASE_URL", ""
    )
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY") or os.getenv(
        "LLM_PROXY_API_KEY", ""
    )
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "")
    embedding_timeout_seconds: float = float(
        os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60")
    )
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "email_subjects")
    qdrant_timeout_seconds: float = float(
        os.getenv("QDRANT_TIMEOUT_SECONDS", "30")
    )
    celery_broker_url: str = os.getenv(
        "CELERY_BROKER_URL", "redis://localhost:6379/0"
    )
    celery_result_backend: str = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/1"
    )
    celery_queue_name: str = os.getenv("CELERY_QUEUE_NAME", "email-llm")
    celery_key_prefix: str = os.getenv("CELERY_KEY_PREFIX", "email-agent:")
    celery_visibility_timeout: int = int(
        os.getenv("CELERY_VISIBILITY_TIMEOUT", "3600")
    )
    celery_result_expires: int = int(os.getenv("CELERY_RESULT_EXPIRES", "86400"))
    celery_task_max_retries: int = int(
        os.getenv("CELERY_TASK_MAX_RETRIES", "4")
    )
    celery_task_retry_backoff: int = int(
        os.getenv("CELERY_TASK_RETRY_BACKOFF", "30")
    )
    celery_worker_concurrency: int = int(
        os.getenv("CELERY_WORKER_CONCURRENCY", "1")
    )
    celery_worker_pool: str = os.getenv(
        "CELERY_WORKER_POOL", "solo" if os.name == "nt" else "prefork"
    )
    initial_sync_days: int = int(os.getenv("INITIAL_SYNC_DAYS", "30"))
    sync_page_size: int = int(os.getenv("SYNC_PAGE_SIZE", "100"))
    gmail_quota_units_per_minute: float = float(
        os.getenv("GMAIL_QUOTA_UNITS_PER_MINUTE", "4800")
    )
    gmail_retry_max_attempts: int = int(
        os.getenv("GMAIL_RETRY_MAX_ATTEMPTS", "8")
    )
    gmail_retry_base_delay_seconds: float = float(
        os.getenv("GMAIL_RETRY_BASE_DELAY_SECONDS", "1")
    )
    gmail_retry_max_delay_seconds: float = float(
        os.getenv("GMAIL_RETRY_MAX_DELAY_SECONDS", "60")
    )
    google_cloud_project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    gmail_pubsub_topic: str = os.getenv(
        "GMAIL_PUBSUB_TOPIC", "gmail-notifications"
    )
    gmail_pubsub_subscription: str = os.getenv(
        "GMAIL_PUBSUB_SUBSCRIPTION", "gmail-notifications-sub"
    )
    gmail_watch_renewal_seconds: int = int(
        os.getenv("GMAIL_WATCH_RENEWAL_SECONDS", "86400")
    )
