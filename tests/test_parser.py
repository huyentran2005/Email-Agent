import base64
import unittest

from email_collector.parser import parse_gmail_message


def _encoded(value: str) -> str:
    """Mã hóa chuỗi thành base64url để mô phỏng payload Gmail."""
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


class GmailParserTest(unittest.TestCase):
    def test_parse_multipart_gmail_message(self):
        """Kiểm tra parser đọc đúng body, địa chỉ, attachment và raw payload."""
        message = {
            "id": "gmail-message-1",
            "threadId": "gmail-thread-1",
            "internalDate": "1762070400000",
            "labelIds": ["INBOX", "IMPORTANT"],
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [
                    {"name": "From", "value": "Sender <sender@example.com>"},
                    {"name": "To", "value": "User <user@example.com>"},
                    {"name": "Subject", "value": "Collector test"},
                    {"name": "Message-ID", "value": "<message-1@example.com>"},
                    {"name": "Date", "value": "Sun, 2 Nov 2025 08:00:00 +0000"},
                ],
                "body": {"size": 0},
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": _encoded("Hello from Gmail")},
                    },
                    {
                        "mimeType": "application/pdf",
                        "filename": "report.pdf",
                        "body": {"attachmentId": "attachment-1", "size": 42},
                    },
                ],
            },
        }

        parsed = parse_gmail_message(message)

        self.assertEqual(parsed.provider_message_id, "gmail-message-1")
        self.assertEqual(parsed.body_text, "Hello from Gmail")
        self.assertEqual(
            parsed.to_addresses,
            [{"name": "User", "address": "user@example.com"}],
        )
        self.assertEqual(parsed.attachments[0].filename, "report.pdf")
        self.assertEqual(
            parsed.raw_payload["payload"]["parts"][0]["body"]["data"],
            "<removed>",
        )


if __name__ == "__main__":
    unittest.main()
