import json
import logging

from email_collector.config import Settings
from email_collector.watch import register_gmail_watch
from gmail_auth import get_gmail_service


def main() -> None:
    """Đăng ký Gmail Watch một lần để thiết lập và kiểm tra."""
    logging.basicConfig(level=logging.INFO)
    response = register_gmail_watch(get_gmail_service(), Settings())
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
