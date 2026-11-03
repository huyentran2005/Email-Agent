import argparse
import json
import logging

from email_ingestion.collector import EmailCollector
from gmail_auth import get_gmail_service


def main() -> None:
    """Đọc tham số CLI, khởi tạo collector và thực hiện một đợt đồng bộ Gmail."""
    parser = argparse.ArgumentParser(
        description="Collect Gmail messages into PostgreSQL and MinIO."
    )
    parser.add_argument(
        "--initial",
        action="store_true",
        help="Force a bounded initial sync instead of Gmail history sync.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    collector = EmailCollector(get_gmail_service())
    collector.initialize()
    result = collector.sync(force_initial=args.initial)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
