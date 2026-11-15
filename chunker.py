"""Splits company_policy.txt into chunks and tags each chunk with who may read it."""
import re
from pathlib import Path

# A chunk that contains this word is restricted to hr_admin users.
# Keep each policy section in ONE paragraph (no blank line inside a section),
# otherwise the second half of a confidential section would not carry the marker.
CONFIDENTIAL_MARKER = "CONFIDENTIAL"


def load_and_chunk(file_path):
    """Return a list of {"text": ..., "role_required": ...}, one dict per paragraph."""
    text = Path(file_path).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        restricted = CONFIDENTIAL_MARKER in paragraph.upper()
        chunks.append({
            "text": paragraph,
            "role_required": "hr_admin" if restricted else "employee",
        })
    return chunks


if __name__ == "__main__":
    policy = Path(__file__).resolve().parent / "company_policy.txt"
    for i, chunk in enumerate(load_and_chunk(policy), start=1):
        print(f"\nChunk {i} [{chunk['role_required']}]:\n{chunk['text']}")