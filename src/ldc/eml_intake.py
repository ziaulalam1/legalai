import html as html_lib
import re
import subprocess
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import List, Tuple

ALLOWED_EXT = {".pdf", ".txt", ".html", ".htm"}


def parse_eml(eml_path: Path) -> dict:
    msg = BytesParser(policy=policy.default).parsebytes(eml_path.read_bytes())
    raw_mid = msg.get("Message-ID", "<unknown>")
    return {
        "message_id": raw_mid.strip().strip("<>"),
        "subject": msg.get("Subject", ""),
        "from": msg.get("From", ""),
        "to": [x.strip() for x in (msg.get("To", "") or "").split(",") if x.strip()],
        "raw": msg,
    }


def _pdf_to_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def _html_to_text(raw: str) -> str:
    # good enough for legal doc attachments, not worth bringing in BeautifulSoup
    text = re.sub(r"<[^>]+>", " ", raw)
    return html_lib.unescape(text)


def extract_docs(eml_path: Path, work_dir: Path) -> List[Tuple[Path, str]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    info = parse_eml(eml_path)
    msg = info["raw"]
    docs: List[Tuple[Path, str]] = []

    for part in msg.walk():
        ct = part.get_content_type()
        filename = part.get_filename()

        if filename:
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_EXT:
                continue
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            saved = work_dir / filename
            saved.write_bytes(payload)
            if ext == ".pdf":
                text = _pdf_to_text(saved)
            elif ext in {".html", ".htm"}:
                charset = part.get_content_charset() or "utf-8"
                text = _html_to_text(payload.decode(charset, errors="replace"))
            else:
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
            docs.append((saved, text))

        elif ct == "text/plain" and not filename:
            body = part.get_content()
            if body and body.strip():
                p = work_dir / "body.txt"
                p.write_text(body)
                docs.append((p, body))

        elif ct == "text/html" and not filename:
            raw = part.get_content()
            text = _html_to_text(raw)
            if text.strip():
                p = work_dir / "body.html"
                p.write_text(text)
                docs.append((p, text))

    return docs
