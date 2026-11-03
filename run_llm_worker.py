import argparse
import logging

from email_collector.celery_app import celery_app
from email_collector.config import Settings
from email_collector.llm_queue import CeleryJobQueue


def main() -> None:
    """Tùy chọn backfill rồi khởi động Celery worker xử lý queue LLM."""
    settings = Settings()
    parser = argparse.ArgumentParser(
        description="Process LLM jobs with a Celery worker backed by Redis."
    )
    parser.add_argument(
        "--enqueue-existing",
        action="store_true",
        help="Enqueue every existing email thread before starting the worker",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        help="Celery worker log level (default: INFO)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=settings.celery_worker_concurrency,
        help="Number of concurrent Celery execution slots",
    )
    parser.add_argument(
        "--pool",
        default=settings.celery_worker_pool,
        help="Celery pool, usually solo on Windows and prefork on Linux",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.enqueue_existing:
        enqueued = CeleryJobQueue(settings).enqueue_all_threads()
        logging.info("Enqueued %d existing email threads", enqueued)

    celery_app.worker_main(
        [
            "worker",
            f"--loglevel={args.loglevel}",
            f"--concurrency={args.concurrency}",
            f"--pool={args.pool}",
            f"--queues={settings.celery_queue_name}",
        ]
    )


if __name__ == "__main__":
    main()

