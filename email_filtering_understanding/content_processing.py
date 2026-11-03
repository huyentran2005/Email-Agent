import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString


MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_LINKS = 200
_DROP_TAGS = {"head", "noscript", "script", "style", "svg", "template"}
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "div", "footer", "h1", "h2",
    "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p",
    "pre", "section", "table", "td", "th", "tr", "ul",
}
_HIDDEN_STYLE = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:[;\s]|$)|font-size\s*:\s*0(?:px|em|rem|%)?)",
    re.I,
)


def normalize_plain_text(value: str | None) -> str | None:
    """Chuẩn hóa whitespace và loại ký tự điều khiển khỏi plain text."""
    if not value:
        return None
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = value.strip()
    return value or None


def _bounded_html(value: str) -> str:
    """Giới hạn kích thước HTML theo số byte tối đa."""
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_HTML_BYTES:
        return value
    return encoded[:MAX_HTML_BYTES].decode("utf-8", errors="ignore")


def _is_hidden(tag) -> bool:
    """Kiểm tra một thẻ HTML có bị ẩn hay không."""
    if tag.has_attr("hidden") or str(tag.get("aria-hidden", "")).lower() == "true":
        return True
    return bool(_HIDDEN_STYLE.search(str(tag.get("style", ""))))


def html_source_text(value: str | None) -> str | None:
    """Lấy cả text ẩn để lớp an toàn có thể phát hiện instruction bị che giấu."""
    if not value:
        return None
    soup = BeautifulSoup(_bounded_html(value), "html5lib")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    return normalize_plain_text(soup.get_text(" "))


def html_to_text(value: str | None) -> tuple[str | None, list[dict[str, str]]]:
    """Parse HTML lỗi an toàn, bỏ nội dung ẩn/active và giữ đích liên kết."""
    if not value:
        return None, []
    soup = BeautifulSoup(_bounded_html(value), "html5lib")
    for tag in list(soup.find_all(True)):
        if tag.attrs is None:  # Parent đã bị decompose nên node này không còn hợp lệ.
            continue
        if tag.name in _DROP_TAGS or _is_hidden(tag):
            tag.decompose()

    links: list[dict[str, str]] = []
    for anchor in list(soup.find_all("a", href=True))[:MAX_LINKS]:
        url = str(anchor.get("href", "")).strip()
        label = normalize_plain_text(anchor.get_text(" ")) or url
        links.append({"text": label, "url": url})
        if url and url not in label:
            anchor.append(NavigableString(f" ({url})"))

    for image in list(soup.find_all("img")):
        alt = normalize_plain_text(str(image.get("alt", "")))
        image.replace_with(NavigableString(f" {alt} " if alt else ""))
    for br in list(soup.find_all("br")):
        br.replace_with(NavigableString("\n"))
    for tag in list(soup.find_all(_BLOCK_TAGS)):
        tag.insert_before(NavigableString("\n"))
        tag.insert_after(NavigableString("\n"))
    return normalize_plain_text(soup.get_text(" ")), links


def extract_plain_urls(value: str | None) -> list[str]:
    """Lấy URL HTTP(S) từ plain text và bỏ dấu câu thường bám cuối URL."""
    if not value:
        return []
    urls = re.findall(r"https?://[^\s<>\"']+", value, flags=re.IGNORECASE)
    return [url.rstrip(".,;:!?)]}") for url in urls[:MAX_LINKS]]


def displayed_domain(value: str) -> str | None:
    """Nhận diện khi nhãn liên kết tự hiển thị như một URL/domain."""
    candidate = value.strip().lower()
    if not candidate:
        return None
    has_scheme = bool(re.match(r"^[a-z][a-z0-9+.-]*://", candidate))
    if not has_scheme and not re.fullmatch(
        r"(?:[^\W_](?:[^\s./]*[^\W_])?\.)+[^\W_]{2,63}(?:/\S*)?",
        candidate,
        re.UNICODE,
    ):
        return None
    if not has_scheme:
        candidate = "https://" + candidate
    try:
        return (urlparse(candidate).hostname or "").lower() or None
    except ValueError:
        return None
