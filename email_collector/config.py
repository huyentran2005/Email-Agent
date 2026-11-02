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
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv(
        "MINIO_SECRET_KEY", "minioadmin_dev_password"
    )
    minio_bucket: str = os.getenv("MINIO_BUCKET", "email-attachments")
    minio_secure: bool = _as_bool(os.getenv("MINIO_SECURE", "false"))
    initial_sync_days: int = int(os.getenv("INITIAL_SYNC_DAYS", "30"))
    sync_page_size: int = int(os.getenv("SYNC_PAGE_SIZE", "100"))
