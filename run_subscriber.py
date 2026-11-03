import json
import logging
import threading

from email_collector.collector import EmailCollector
from email_collector.config import Settings
from email_collector.llm_queue import CeleryJobQueue
from email_collector.subscriber import GmailPullSubscriber
from email_collector.watch import register_gmail_watch, renew_watch_forever
from gmail_auth import get_gmail_service


def main() -> None:
    """Khởi động Gmail Watch, đồng bộ bù rồi xử lý Pub/Sub liên tục."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings()
    gmail_service = get_gmail_service()
    collector = EmailCollector(gmail_service, settings)
    collector.initialize()
    job_queue = CeleryJobQueue(settings)

    register_gmail_watch(gmail_service, settings, collector.gmail_requests)
    startup_result = collector.sync()
    startup_jobs = job_queue.enqueue_email_ids(startup_result.email_ids)
    print(json.dumps(startup_result.to_dict(), indent=2, ensure_ascii=False))
    logging.info("Enqueued %d startup LLM thread jobs", startup_jobs)

    stop_event = threading.Event()
    renewal_thread = threading.Thread(
        target=renew_watch_forever,
        args=(settings, stop_event),
        name="gmail-watch-renewal",
        daemon=True,
    )
    renewal_thread.start()

    try:
        GmailPullSubscriber(
            collector=collector,
            settings=settings,
            expected_email=startup_result.account,
            job_queue=job_queue,
        ).run_forever()
    finally:
        stop_event.set()
        renewal_thread.join(timeout=5)


if __name__ == "__main__":
    main()
