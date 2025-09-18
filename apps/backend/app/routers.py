# apps/backend/app/routers.py

from fastapi import APIRouter, Depends, Query, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import os
import tempfile
from uuid import UUID

from .db import get_session
from .embedder import embed_texts, DIM as EMB_DIM
from .pdf_ingest import extract_pages_text, chunk_pages
from .qgen import generate_mcqs

router = APIRouter()

# ---------- Helpers ----------

def _ensure_user(db: Session, email: str):
    row = db.execute(text("SELECT id FROM users WHERE email=:e"), {"e": email}).fetchone()
    if not row:
        row = db.execute(text("INSERT INTO users(email) VALUES(:e) RETURNING id"), {"e": email}).fetchone()
    return row[0]

# ---------- Health ----------

@router.get("/healthz")
def healthz(db: Session = Depends(get_session)):
    db.execute(text("SELECT 1"))
    return {"ok": True}

# ---------- Seed demo data ----------

@router.post("/dev/seed")
def seed_demo(db: Session = Depends(get_session)):
    """
    Insert a tiny doc with 3 chunks + embeddings so you can test vector search.
    Idempotent: re-calling won't duplicate.
    """
    # Ensure user
    user_row = db.execute(text("SELECT id FROM users LIMIT 1")).fetchone()
    if not user_row:
        user_row = db.execute(
            text("INSERT INTO users(email) VALUES(:e) RETURNING id"),
            {"e": "demo@example.com"},
        ).fetchone()
    user_id = user_row[0]

    # Ensure document
    doc_row = db.execute(
        text("SELECT id FROM documents WHERE title=:t"),
        {"t": "Demo Doc"},
    ).fetchone()
    if not doc_row:
        doc_row = db.execute(
            text("""
                INSERT INTO documents(owner_id, title, meta)
                VALUES (:owner, :title, '{}'::jsonb)
                RETURNING id
            """),
            {"owner": user_id, "title": "Demo Doc"},
        ).fetchone()
    doc_id = doc_row[0]

    samples = [
        ("Puppy training basics include positive reinforcement and short sessions.", 1, 1, ["Training"]),
        ("NVIDIA driver troubleshooting on Windows often involves clean installs and DDU.", 2, 2, ["GPUs"]),
        ("Tomatoes grow well in containers with 6+ hours of sun and regular watering.", 3, 3, ["Gardening"]),
    ]

    # Insert chunks if missing
    chunk_ids = []
    for txt, ps, pe, tags in samples:
        exists = db.execute(
            text("SELECT id FROM chunks WHERE doc_id=:d AND text=:t"),
            {"d": doc_id, "t": txt},
        ).fetchone()
        if exists:
            chunk_ids.append(exists[0])
            continue
        row = db.execute(
            text("""
                INSERT INTO chunks(doc_id, text, page_start, page_end, headings, concept_tags)
                VALUES (:doc_id, :text, :ps, :pe, :hd, :tags)
                RETURNING id
            """),
            {"doc_id": doc_id, "text": txt, "ps": ps, "pe": pe, "hd": ["Demo"], "tags": tags},
        ).fetchone()
        chunk_ids.append(row[0])

    # Create embeddings if missing
    texts_to_embed, ids_to_embed = [], []
    for cid, (txt, *_rest) in zip(chunk_ids, samples):
        has = db.execute(text("SELECT 1 FROM embeddings WHERE chunk_id=:cid"), {"cid": cid}).fetchone()
        if not has:
            texts_to_embed.append(txt)
            ids_to_embed.append(cid)

    if ids_to_embed:
        vecs = embed_texts(texts_to_embed)
        for cid, vec in zip(ids_to_embed, vecs):
            qvec = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            db.execute(
                text("""
                    INSERT INTO embeddings(chunk_id, embedding, model_name)
                    VALUES (:cid, (:qvec)::vector, :model)
                """),
                {"cid": cid, "qvec": qvec, "model": os.getenv("OPENAI_EMBEDDING_MODEL", "dev-fake")},
            )

    return {"seeded_chunks": [str(c) for c in chunk_ids], "provider": os.getenv("EMBEDDING_PROVIDER", "openai")}

# ---------- Vector search ----------

@router.get("/search")
def search(
    q: str = Query(..., description="Query string"),
    k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_session),
):
    """
    Vector search (cosine). Falls back to text search if embedding fails.
    """
    try:
        vec = embed_texts([q])[0]
        qvec = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        rows = db.execute(
            text("""
                SELECT c.id::text,
                       c.text,
                       1 - (e.embedding <=> (:qvec)::vector) AS cosine_similarity,
                       c.page_start, c.page_end
                FROM embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                ORDER BY e.embedding <=> (:qvec)::vector
                LIMIT :k
            """),
            {"qvec": qvec, "k": k},
        ).mappings().all()
        return {"mode": "vector", "query": q, "count": len(rows), "results": list(rows)}
    except Exception as e:
        rows = db.execute(
            text("""
                SELECT id::text, text, page_start, page_end
                FROM chunks
                WHERE text ILIKE '%' || :q || '%'
                ORDER BY created_at DESC
                LIMIT :k
            """),
            {"q": q, "k": k},
        ).mappings().all()
        return {"mode": "text", "query": q, "count": len(rows), "results": list(rows), "error": str(e)}

# ---------- Ingestion: text ----------

class IngestTextReq(BaseModel):
    title: str
    text: str
    owner_email: str = "demo@example.com"

@router.post("/ingest/text")
def ingest_text(payload: IngestTextReq, db: Session = Depends(get_session)):
    title = payload.title
    body_text = payload.text          # avoid shadowing sqlalchemy.text
    owner_email = payload.owner_email

    # ensure user
    owner_id = _ensure_user(db, owner_email)

    # document
    doc_id = db.execute(
        text("""
            INSERT INTO documents(owner_id, title, meta)
            VALUES (:owner, :title, '{}'::jsonb)
            RETURNING id
        """),
        {"owner": owner_id, "title": title},
    ).fetchone()[0]

    # chunk
    chunks = chunk_pages([(1, body_text)], max_chars=1200, overlap=150)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text to ingest.")

    # embed + insert
    BATCH = 64
    total = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        texts_to_embed = [c[0] for c in batch]
        vecs = embed_texts(texts_to_embed)
        for (chunk_text, ps, pe), vec in zip(batch, vecs):
            qvec = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            cid = db.execute(
                text("""
                    INSERT INTO chunks(doc_id, text, page_start, page_end, headings, concept_tags)
                    VALUES (:doc, :txt, :ps, :pe, :hd, :tags)
                    RETURNING id
                """),
                {"doc": doc_id, "txt": chunk_text, "ps": ps, "pe": pe, "hd": [], "tags": []},
            ).fetchone()[0]
            db.execute(
                text("""
                    INSERT INTO embeddings(chunk_id, embedding, model_name)
                    VALUES (:cid, (:qvec)::vector, :model)
                """),
                {"cid": cid, "qvec": qvec, "model": os.getenv("OPENAI_EMBEDDING_MODEL", "dev-fake")},
            )
            total += 1

    return {"doc_id": str(doc_id), "chunks": total, "embedding_dim": EMB_DIM}

# ---------- Ingestion: PDF (digital text) ----------

@router.post("/ingest/upload_pdf")
def ingest_pdf(
    file: UploadFile = File(...),
    title: str = Form(None),
    owner_email: str = Form("demo@example.com"),
    max_pages: int = Form(200),
    db: Session = Depends(get_session),
):
    """
    Extract text per page with PyMuPDF (digital PDFs), chunk, embed, store.
    If no text layer found (likely a scan), return 422 to suggest OCR.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        pages = extract_pages_text(tmp_path)
        if not pages:
            raise HTTPException(status_code=422, detail="Could not open PDF.")

        if max_pages and len(pages) > max_pages:
            pages = pages[:max_pages]

        if all(not t for _, t in pages):
            raise HTTPException(
                status_code=422,
                detail="No text layer found (likely a scanned PDF). Add OCR first or enable an OCR pipeline.",
            )

        # ensure user
        owner_id = _ensure_user(db, owner_email)

        # document
        doc_title = title or (file.filename or "Uploaded PDF")
        doc_id = db.execute(
            text("""
                INSERT INTO documents(owner_id, title, meta)
                VALUES (:owner, :title, '{}'::jsonb)
                RETURNING id
            """),
            {"owner": owner_id, "title": doc_title},
        ).fetchone()[0]

        # chunk per page
        triples = chunk_pages(pages, max_chars=1200, overlap=150)
        if not triples:
            raise HTTPException(status_code=422, detail="No usable text extracted from PDF.")

        # embed + insert
        BATCH = 48
        total = 0
        for i in range(0, len(triples), BATCH):
            batch = triples[i:i+BATCH]
            vecs = embed_texts([t for (t, _, _) in batch])
            for (chunk_text, ps, pe), vec in zip(batch, vecs):
                qvec = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
                cid = db.execute(
                    text("""
                        INSERT INTO chunks(doc_id, text, page_start, page_end, headings, concept_tags)
                        VALUES (:doc, :txt, :ps, :pe, :hd, :tags)
                        RETURNING id
                    """),
                    {"doc": doc_id, "txt": chunk_text, "ps": ps, "pe": pe, "hd": [], "tags": []},
                ).fetchone()[0]
                db.execute(
                    text("""
                        INSERT INTO embeddings(chunk_id, embedding, model_name)
                        VALUES (:cid, (:qvec)::vector, :model)
                    """),
                    {"cid": cid, "qvec": qvec, "model": os.getenv("OPENAI_EMBEDDING_MODEL", "dev-fake")},
                )
                total += 1

        return {"doc_id": str(doc_id), "chunks": total, "pages_processed": len(pages), "title": doc_title}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ---------- Item generation & listing ----------

class GenItemsReq(BaseModel):
    doc_id: str
    per_chunk: int = 1
    max_chunks: int = 20

@router.post("/items/generate_from_doc")
def items_generate_from_doc(payload: GenItemsReq, db: Session = Depends(get_session)):
    """
    For the given document, generate MCQs from up to max_chunks chunks (per_chunk each),
    store them in 'items'. Returns number of items created.
    """
    rows = db.execute(text("""
        SELECT id, text FROM chunks
        WHERE doc_id = :doc
        ORDER BY created_at ASC
        LIMIT :lim
    """), {"doc": payload.doc_id, "lim": payload.max_chunks}).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No chunks for doc_id")

    created = 0
    for cid, chunk_text in rows:
        for it in generate_mcqs(chunk_text, n=payload.per_chunk):
            db.execute(text("""
                INSERT INTO items(stem, choices, correct_index, concept_id, a, b, source_chunk_id)
                VALUES (:stem, :choices, :ci, NULL, 1.0, 0.0, :src)
            """), {"stem": it["stem"], "choices": it["choices"], "ci": it["correct_index"], "src": cid})
            created += 1

    return {"doc_id": payload.doc_id, "chunks_processed": len(rows), "items_created": created}

@router.get("/items/by_doc")
def items_by_doc(doc_id: str, limit: int = 10, db: Session = Depends(get_session)):
    rows = db.execute(text("""
        SELECT i.id::text, i.stem, i.choices, i.correct_index, c.page_start, c.page_end
        FROM items i
        LEFT JOIN chunks c ON c.id = i.source_chunk_id
        WHERE c.doc_id = :doc
        ORDER BY i.created_at DESC
        LIMIT :lim
    """), {"doc": doc_id, "lim": limit}).mappings().all()
    return {"doc_id": doc_id, "count": len(rows), "items": list(rows)}

# ---------- Quiz loop: next / answer / progress ----------

class QuizAnswerReq(BaseModel):
    user_email: str
    item_id: UUID          # validate UUID up front
    choice_index: int

@router.get("/quiz/next")
def quiz_next(doc_id: str, user_email: str, db: Session = Depends(get_session)):
    """
    Returns the next unanswered item for this user+doc, or 404 if none left.
    """
    user_id = _ensure_user(db, user_email)
    row = db.execute(text("""
        SELECT i.id::text AS item_id, i.stem, i.choices, i.correct_index
        FROM items i
        JOIN chunks c ON c.id = i.source_chunk_id
        LEFT JOIN interactions r
          ON r.item_id = i.id AND r.user_id = :uid
        WHERE c.doc_id = :doc AND r.id IS NULL
        ORDER BY random()
        LIMIT 1
    """), {"uid": user_id, "doc": doc_id}).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="No remaining items for this document")
    # Don’t leak the correct index
    return {"item_id": row["item_id"], "stem": row["stem"], "choices": row["choices"]}

@router.post("/quiz/answer")
def quiz_answer(payload: QuizAnswerReq, db: Session = Depends(get_session)):
    """
    Records the answer; returns correctness.
    Uses interactions.correct (boolean) and interactions.chosen (int).
    """
    user_id = _ensure_user(db, payload.user_email)

    # Fetch item & correct answer
    item = db.execute(text("""
        SELECT id, correct_index FROM items WHERE id = :iid
    """), {"iid": str(payload.item_id)}).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    is_correct = (payload.choice_index == item.correct_index)

    # Write interaction (schema column name is 'chosen'). Use an idempotent insert.
    db.execute(text("""
        INSERT INTO interactions(user_id, item_id, chosen, correct)
        SELECT :uid, :iid, :ci, :ok
        WHERE NOT EXISTS (
            SELECT 1 FROM interactions WHERE user_id = :uid AND item_id = :iid
        )
    """), {"uid": user_id, "iid": item.id, "ci": payload.choice_index, "ok": is_correct})

    return {"correct": bool(is_correct)}

@router.get("/quiz/progress")
def quiz_progress(doc_id: str, user_email: str, db: Session = Depends(get_session)):
    """
    Returns counts for answered/total and accuracy for this doc.
    """
    user_id = _ensure_user(db, user_email)

    total = db.execute(text("""
        SELECT count(*) FROM items i
        JOIN chunks c ON c.id = i.source_chunk_id
        WHERE c.doc_id = :doc
    """), {"doc": doc_id}).scalar_one()

    answered = db.execute(text("""
        SELECT count(*) FROM interactions r
        JOIN items i ON i.id = r.item_id
        JOIN chunks c ON c.id = i.source_chunk_id
        WHERE r.user_id = :uid AND c.doc_id = :doc
    """), {"uid": user_id, "doc": doc_id}).scalar_one()

    correct = db.execute(text("""
        SELECT count(*) FROM interactions r
        JOIN items i ON i.id = r.item_id
        JOIN chunks c ON c.id = i.source_chunk_id
        WHERE r.user_id = :uid AND c.doc_id = :doc AND r.correct = true
    """), {"uid": user_id, "doc": doc_id}).scalar_one()

    acc = (correct / answered) if answered else 0.0
    return {"total": total, "answered": answered, "correct": correct, "accuracy": round(acc, 3)}
