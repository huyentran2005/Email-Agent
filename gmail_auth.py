import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
BASE_DIR = Path(__file__).resolve().parent


def get_gmail_service(
    credentials_path: str | Path | None = None,
    token_path: str | Path | None = None,
):
    """Xác thực OAuth cục bộ và trả về Gmail API service đã được cấp quyền."""
    credentials_path = Path(
        credentials_path
        or os.getenv("GMAIL_CREDENTIALS_PATH", BASE_DIR / "credentials.json")
    )
    token_path = Path(
        token_path or os.getenv("GMAIL_TOKEN_PATH", BASE_DIR / "token.json")
    )

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES
            )
            creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


if __name__ == "__main__":
    service = get_gmail_service()
    profile = service.users().getProfile(userId="me").execute()
    print("Connected Gmail:", profile["emailAddress"])
