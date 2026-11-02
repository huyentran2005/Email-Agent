# Email Agent Collector

Collector đồng bộ Gmail vào PostgreSQL, lưu attachment trong MinIO và dùng
`historyId` của Gmail để các lần sau chỉ đọc những email đã thay đổi.

## Kiến trúc

```text
Gmail API
   -> parse + normalize
   -> upsert PostgreSQL
   -> upload attachment MinIO
   -> cập nhật Gmail history checkpoint
```

Checkpoint chỉ được cập nhật sau khi toàn bộ đợt đồng bộ thành công. Email được
chống trùng bằng cặp `(account_id, provider_message_id)`; object MinIO dùng key
xác định nên chạy lại không tạo thêm bản sao.

Collector giữ checkpoint chụp trước khi bắt đầu một đợt chạy. Vì vậy email đến
trong lúc collector đang xử lý sẽ được History API lấy ở lần chạy kế tiếp, thay
vì có nguy cơ bị bỏ qua.

## Chạy local

Yêu cầu Python 3.11+ và Docker Compose.

1. Tạo cấu hình môi trường:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Khởi động PostgreSQL và MinIO:

   ```powershell
   docker compose up -d
   ```

3. Cài dependency:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

4. Đặt OAuth client của Google tại `credentials.json`. Chạy xác thực nếu chưa
   có `token.json`:

   ```powershell
   python gmail_auth.py
   ```

5. Đồng bộ email:

   ```powershell
   python collect_emails.py
   ```

Lần chạy đầu tự động lấy email trong `INITIAL_SYNC_DAYS` ngày gần nhất. Những
lần sau sử dụng Gmail History API. Để chủ động đồng bộ lại khoảng thời gian ban
đầu:

```powershell
python collect_emails.py --initial
```

## Dịch vụ local

- PostgreSQL: `localhost:5432`
- MinIO S3 API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

Thông tin đăng nhập development nằm trong `.env.example`; hãy đổi mật khẩu khi
triển khai ngoài máy local. `credentials.json`, `token.json` và `.env` đã được
loại khỏi Git.

## Các bảng

- `email_accounts`: tài khoản và Gmail history checkpoint.
- `email_threads`: thread Gmail.
- `emails`: nội dung chuẩn hóa và payload gốc đã loại body base64.
- `email_attachments`: metadata và MinIO object key.
- `email_sync_runs`: audit mỗi lần đồng bộ.

Hiện schema được tạo tự động bằng SQLAlchemy khi collector khởi động. Khi dự án
bắt đầu có nhiều môi trường hoặc thay đổi schema, nên chuyển sang Alembic để
quản lý migration.

## Dùng như một workflow node

Node trả về phần state cần merge nên có thể dùng độc lập hoặc gắn vào LangGraph:

```python
from email_collector.node import build_email_collector_node

email_collector_node = build_email_collector_node()

# Với LangGraph:
workflow.add_node("email_collector", email_collector_node)
```

Input state tùy chọn:

```python
{"force_initial_sync": True}
```

Output state:

```python
{
    "email_collection": {
        "status": "success",
        "mode": "initial",
        "collected": 10,
        "inserted": 8,
        "updated": 2,
        "skipped": 0,
    },
    "collected_email_ids": [1, 2, 3],
}
```
