import argparse
import logging

from core.config import Settings
from workers.celery_app import celery_app
from workers.embedding_queue import EmbeddingJobQueue
from workers.llm_queue import CeleryJobQueue


def main() -> None:
    """Tùy chọn backfill rồi chạy worker xử lý LLM và subject embedding."""
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
        embedding_jobs = EmbeddingJobQueue(settings).enqueue_all_emails()
        logging.info("Enqueued %d existing email threads", enqueued)
        logging.info("Enqueued %d existing subject embedding jobs", embedding_jobs)

    celery_app.worker_main(
        [
            "worker",
            f"--loglevel={args.loglevel}",
            f"--concurrency={args.concurrency}",
            f"--pool={args.pool}",
            "--queues="
            f"{settings.celery_queue_name},{settings.celery_embedding_queue_name}",
        ]
    )


if __name__ == "__main__":
    main()

