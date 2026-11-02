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
