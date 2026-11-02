from io import BytesIO

from minio import Minio
from minio.error import S3Error

from .config import Settings


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
