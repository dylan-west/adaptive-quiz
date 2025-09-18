import fitz  # PyMuPDF
from typing import List, Tuple
import os
import base64

def extract_pages_text(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Returns list of (page_number_1_based, text) for each page.
    If the PDF has no text layer, returned texts may be empty.
    """
    out = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            out.append((i + 1, text.strip()))
    finally:
        doc.close()
    return out

def chunk_text_by_chars(
    text: str,
    max_chars: int = 1200,
    overlap: int = 150
) -> List[str]:
    """
    Simple, robust chunker: split on blank lines; pack paragraphs into ~max_chars chunks,
    with small overlap between chunks for context retention.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    cur = []
    cur_len = 0
    for p in paras:
        plen = len(p)
        if cur and cur_len + 1 + plen > max_chars:
            chunks.append("\n\n".join(cur))
            # overlap: take tail from previous chunk
            tail = chunks[-1][-overlap:]
            cur = [tail, p] if tail else [p]
            cur_len = len("\n\n".join(cur))
        else:
            cur.append(p)
            cur_len += plen + (2 if cur_len > 0 else 0)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks

def chunk_pages(
    pages: List[Tuple[int, str]],
    max_chars: int = 1200,
    overlap: int = 150
) -> List[Tuple[str, int, int]]:
    """
    From page texts -> list of (chunk_text, page_start, page_end).
    We chunk *within* each page (simpler + preserves page refs).
    """
    out: List[Tuple[str, int, int]] = []
    for pg, txt in pages:
        if not txt:
            continue
        pieces = chunk_text_by_chars(txt, max_chars=max_chars, overlap=overlap)
        for piece in pieces:
            out.append((piece, pg, pg))
    return out

# ------------- OCR (OpenAI Vision) -------------

def ocr_pages_with_openai(
    pdf_path: str,
    page_numbers: List[int],
    dpi: int = 150,
    model: str | None = None,
) -> List[Tuple[int, str]]:
    """
    OCR specific pages from a PDF using OpenAI Vision. Returns (page_number, text).
    Requires OPENAI_API_KEY in environment.
    """
    if not page_numbers:
        return []

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed or import failed") from e

    model = model or os.getenv("OCR_MODEL", "gpt-4o-mini")
    client = OpenAI()  # reads OPENAI_API_KEY

    out: List[Tuple[int, str]] = []
    doc = fitz.open(pdf_path)
    try:
        for pg_no in page_numbers:
            if pg_no < 1 or pg_no > len(doc):
                continue
            page = doc[pg_no - 1]
            zoom = float(dpi) / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            png_bytes = pix.tobytes("png")
            b64 = base64.b64encode(png_bytes).decode("utf-8")

            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all readable text from this page. Return plain UTF-8 text only."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip()
            out.append((pg_no, text))
    finally:
        doc.close()
    return out
