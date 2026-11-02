import hashlib
import re
from io import BytesIO
from pathlib import PurePosixPath

from minio import Minio
from minio.error import S3Error

from .config import Settings


def build_attachment_key(
    message_id: str, attachment_id: str, filename: str
) -> str:
    """Tạo MinIO key ổn định với độ dài thành phần an toàn cho hệ thống file."""
    attachment_hash = hashlib.sha256(attachment_id.encode("utf-8")).hexdigest()
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ""
    return f"gmail/{message_id}/{attachment_hash}{suffix}"


class AttachmentStorage:
    def __init__(self, settings: Settings):
        """Khởi tạo MinIO client từ cấu hình ứng dụng."""
        self.bucket = settings.minio_bucket
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )

    def ensure_bucket(self) -> None:
        """Tạo bucket lưu attachment nếu bucket chưa tồn tại."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as exc:
            if exc.code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise

    def put(self, key: str, data: bytes, content_type: str) -> None:
        """Ghi nội dung attachment vào MinIO bằng object key xác định."""
        self.client.put_object(
            self.bucket,
            key,
            BytesIO(data),
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )
