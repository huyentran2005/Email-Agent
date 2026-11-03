import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from google.cloud import pubsub_v1
from google.cloud.pubsub_v1.subscriber.scheduler import ThreadScheduler

from core.config import Settings
from email_ingestion.collector import EmailCollector
from workers.llm_queue import CeleryJobQueue

from .watch import resolve_project_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GmailNotification:
    email_address: str
    history_id: str


def decode_notification(data: bytes) -> GmailNotification:
    """Giải mã payload JSON Gmail do Pub/Sub Python client trả về."""
    payload = json.loads(data.decode("utf-8"))
    return GmailNotification(
        email_address=payload["emailAddress"],
        history_id=str(payload["historyId"]),
    )


class GmailPullSubscriber:
    def __init__(
        self,
        collector: EmailCollector,
        settings: Settings,
        expected_email: str,
        job_queue: CeleryJobQueue,
        subscriber_client=None,
    ):
        """Khởi tạo subscriber và các phụ thuộc cần thiết."""
        self.collector = collector
        self.settings = settings
        self.expected_email = expected_email.casefold()
        self.job_queue = job_queue
        self.subscriber = subscriber_client or pubsub_v1.SubscriberClient()

    def handle_message(self, message) -> None:
        """Đồng bộ một notification hộp thư và chỉ ACK khi thành công."""
        try:
            notification = decode_notification(message.data)
            if notification.email_address.casefold() != self.expected_email:
                logger.warning(
                    "Ignoring Pub/Sub notification for unexpected account %s",
                    notification.email_address,
                )
                message.ack()
                return

            logger.info(
                "Received Gmail notification for %s at historyId=%s",
                notification.email_address,
                notification.history_id,
            )
            result = self.collector.sync()
            logger.info(
                "Gmail sync finished: collected=%d inserted=%d updated=%d",
                result.collected,
                result.inserted,
                result.updated,
            )
            enqueued = self.job_queue.enqueue_email_ids(result.email_ids)
            logger.info("Enqueued %d LLM thread jobs", enqueued)
            message.ack()
        except Exception:
            logger.exception("Gmail notification processing failed")
            message.nack()

    def run_forever(self) -> None:
        """Chạy tuần tự từng callback để Pub/Sub quản lý thời hạn message."""
        project_id = resolve_project_id(self.settings)
        subscription_path = self.subscriber.subscription_path(
            project_id, self.settings.gmail_pubsub_subscription
        )
        flow_control = pubsub_v1.types.FlowControl(max_messages=1)
        executor = ThreadPoolExecutor(max_workers=1)
        scheduler = ThreadScheduler(executor)
        future = self.subscriber.subscribe(
            subscription_path,
            callback=self.handle_message,
            flow_control=flow_control,
            scheduler=scheduler,
        )
        logger.info("Listening for Gmail notifications on %s", subscription_path)
        try:
            future.result()
        except KeyboardInterrupt:
            logger.info("Stopping Gmail Pub/Sub subscriber")
            future.cancel()
            future.result()
        finally:
            self.subscriber.close()
            executor.shutdown(wait=True, cancel_futures=True)
