import fitz  # PyMuPDF
from typing import List, Tuple

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
