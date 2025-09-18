# apps/backend/app/routers.py

from fastapi import APIRouter, Depends, Query, HTTPException, File, UploadFile, Form
import json
from pydantic import BaseModel
from typing import Optional
import math
from sqlalchemy.orm import Session
from sqlalchemy import text
import os
import tempfile
from uuid import UUID

from .db import get_session, SessionLocal
from .embedder import embed_texts, DIM as EMB_DIM
from .pdf_ingest import extract_pages_text, chunk_pages, ocr_pages_with_openai
from .qgen import generate_mcqs, _sanitize_choices

# Cache whether items.rationale exists to avoid repeated information_schema queries
_HAS_ITEMS_RATIONALE: Optional[bool] = None

def _items_has_rationale(db: Session) -> bool:
    global _HAS_ITEMS_RATIONALE
    if _HAS_ITEMS_RATIONALE is not None:
        return _HAS_ITEMS_RATIONALE
    try:
        val = db.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'items' AND column_name = 'rationale'
            LIMIT 1
        """)).scalar()
        _HAS_ITEMS_RATIONALE = bool(val)
    except Exception:
        _HAS_ITEMS_RATIONALE = False
    return _HAS_ITEMS_RATIONALE

def _ensure_items_rationale(db: Session) -> bool:
    """Try to add items.rationale column if missing. Returns True if it exists after call."""
    if _items_has_rationale(db):
        return True
    try:
        db.execute(text("""
            ALTER TABLE IF EXISTS items
            ADD COLUMN IF NOT EXISTS rationale TEXT
        """))
        # Update cache
        global _HAS_ITEMS_RATIONALE
        _HAS_ITEMS_RATIONALE = True
        return True
    except Exception:
        return False

router = APIRouter()

# ---------- Helpers ----------

def _ensure_user(db: Session, email: str):
    row = db.execute(text("SELECT id FROM users WHERE email=:e"), {"e": email}).fetchone()
    if not row:
        row = db.execute(text("INSERT INTO users(email) VALUES(:e) RETURNING id"), {"e": email}).fetchone()
    return row[0]

def _heuristic_difficulty_b(text: str) -> float:
    """Estimate item difficulty from chunk text length/lexical variety.
    Returns a value roughly in [-2.5, 2.5]."""
    t = (text or "").strip()
    if not t:
        return 0.0
    tokens = [w for w in t.split() if w.isalpha() or w.isalnum()]
    uniq = len(set(w.lower() for w in tokens))
    # Center near ~80 unique tokens; scale by ~60
    b = (uniq - 80.0) / 60.0
    if b < -2.5: b = -2.5
    if b > 2.5: b = 2.5
    return float(b)

def _top_up_items_for_quiz(db: Session, doc_id: str, user_id: str, per_chunk: int = 1) -> dict:
    """Ensure the document has enough items for this user's session.
    Policy: initial 10 items; then grow pool to min(100, answered+10).
    Prefer generating from chunks associated with recent misses, then underused chunks.
    """
    # How many answered in this doc by this user?
    answered = db.execute(text("""
        SELECT count(*) FROM interactions r
        JOIN items i ON i.id = r.item_id
        JOIN chunks c ON c.id = i.source_chunk_id
        WHERE r.user_id = :uid AND c.doc_id::text = :doc
    """), {"uid": user_id, "doc": doc_id}).scalar_one()

    desired = min(100, max(10, int(answered) + 10))
    # Count only 'valid' items: at least 4 choices and correct_index within bounds
    have_valid = db.execute(text("""
        SELECT count(*)
        FROM items i
        JOIN chunks c ON c.id = i.source_chunk_id
        WHERE c.doc_id::text = :doc
          AND array_length(i.choices, 1) >= 4
          AND i.correct_index >= 0
          AND i.correct_index < COALESCE(array_length(i.choices, 1), 0)
    """), {"doc": doc_id}).scalar_one()
    to_add = int(desired) - int(have_valid)
    if to_add <= 0:
        return {"desired": desired, "have_valid": have_valid, "added": 0}

    # Candidate chunks from recent misses (prioritized)
    missed = db.execute(text("""
        SELECT DISTINCT c.id, c.text
        FROM interactions r
        JOIN items i ON i.id = r.item_id
        JOIN chunks c ON c.id = i.source_chunk_id
        WHERE r.user_id = :uid AND r.correct = false AND c.doc_id::text = :doc
        ORDER BY r.created_at DESC
        LIMIT 20
    """), {"uid": user_id, "doc": doc_id}).fetchall()

    # Underused chunks (fewest items), then random
    underused = db.execute(text("""
        SELECT c.id, c.text, COALESCE(COUNT(i.id),0)::int as item_count
        FROM chunks c
        LEFT JOIN items i ON i.source_chunk_id = c.id
        WHERE c.doc_id::text = :doc
        GROUP BY c.id, c.text
        ORDER BY item_count ASC, c.created_at ASC
        LIMIT 200
    """), {"doc": doc_id}).fetchall()

    # Merge with priority to missed
    seen = set()
    cand: list[tuple] = []
    for row in missed:
        if row[0] not in seen:
            cand.append((row[0], row[1]))
            seen.add(row[0])
    for row in underused:
        if row[0] not in seen:
            cand.append((row[0], row[1]))
            seen.add(row[0])

    added = 0
    has_rat = _ensure_items_rationale(db)
    for cid, ctext in cand:
        if added >= to_add:
            break
        try:
            # generate one per chunk per pass
            items = generate_mcqs(ctext, n=per_chunk)
        except Exception:
            items = []
        for it in items:
            if added >= to_add:
                break
            # Insert
            params = {
                "stem": it["stem"],
                "choices_json": json.dumps(it["choices"]),
                "ci": it["correct_index"],
                "rat": it.get("rationale", ""),
                "src": cid,
                "a": 1.0,
                "b": _heuristic_difficulty_b(ctext),
            }
            try:
                if has_rat:
                    db.execute(text("""
                        INSERT INTO items(stem, choices, correct_index, rationale, concept_id, a, b, source_chunk_id)
                        VALUES (
                          :stem,
                          (SELECT ARRAY(SELECT jsonb_array_elements_text(:choices_json::jsonb))),
                          :ci, :rat, NULL, :a, :b, :src
                        )
                    """), params)
                else:
                    db.execute(text("""
                        INSERT INTO items(stem, choices, correct_index, concept_id, a, b, source_chunk_id)
                        VALUES (
                          :stem,
                          (SELECT ARRAY(SELECT jsonb_array_elements_text(:choices_json::jsonb))),
                          :ci, NULL, :a, :b, :src
                        )
                    """), params)
                db.commit()
                added += 1
            except Exception:
                # If an insert fails (e.g., bad array binding), rollback and continue
                db.rollback()
                continue

    return {"desired": desired, "have_valid": have_valid, "added": added}

# ---------- Documents list ----------

@router.get("/docs/list")
def docs_list(owner_email: str = Query("demo@example.com"), limit: int = 20, db: Session = Depends(get_session)):
    """List recent documents for an owner with item counts and first/last activity."""
    owner_id = _ensure_user(db, owner_email)
    rows = db.execute(text(
        """
        WITH doc_items AS (
            SELECT c.doc_id, count(DISTINCT i.id)::int AS items
            FROM chunks c
            LEFT JOIN items i ON i.source_chunk_id = c.id
            GROUP BY c.doc_id
        )
        SELECT d.id::text AS doc_id, d.title, COALESCE(di.items, 0) AS items,
               d.created_at
        FROM documents d
        LEFT JOIN doc_items di ON di.doc_id = d.id
        WHERE d.owner_id = :oid
        ORDER BY d.created_at DESC
        LIMIT :lim
        """
    ), {"oid": owner_id, "lim": limit}).mappings().all()
    return {"owner_email": owner_email, "count": len(rows), "docs": list(rows)}

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
    If no text layer found (likely a scan), optionally OCR via OpenAI Vision when enabled.
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

        # OCR fallback for pages with no text
        empties = [pg for (pg, txt) in pages if not (txt and txt.strip())]
        use_ocr = os.getenv("OCR_PROVIDER", "").lower() == "openai" and bool(os.getenv("OPENAI_API_KEY"))
        if empties and use_ocr:
            try:
                ocred = ocr_pages_with_openai(
                    tmp_path,
                    empties,
                    dpi=int(os.getenv("OCR_DPI", "150")),
                    model=os.getenv("OCR_MODEL", "gpt-4o-mini"),
                )
                ocr_map = {pg: t for pg, t in ocred}
                pages = [(pg, (ocr_map.get(pg) or txt or "").strip()) for (pg, txt) in pages]
            except Exception as e:
                # Don't fail ingestion; surface helpful error
                raise HTTPException(status_code=502, detail=f"OCR failed: {e}")

        if all(not (t and t.strip()) for _, t in pages):
            raise HTTPException(
                status_code=422,
                detail="No text found. If this is a scanned PDF, set OCR_PROVIDER=openai and provide OPENAI_API_KEY.",
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
    has_rat = _ensure_items_rationale(db)
    for cid, chunk_text in rows:
        for it in generate_mcqs(chunk_text, n=payload.per_chunk):
            ch_json = json.dumps(it["choices"])  # ensure via JSON->text[] for consistency
            if has_rat:
                db.execute(text("""
                    INSERT INTO items(stem, choices, correct_index, rationale, concept_id, a, b, source_chunk_id)
                    VALUES (
                      :stem,
                      (SELECT ARRAY(SELECT jsonb_array_elements_text(:choices_json::jsonb))),
                      :ci, :rat, NULL, 1.0, 0.0, :src
                    )
                """), {"stem": it["stem"], "choices_json": ch_json, "ci": it["correct_index"], "rat": it.get("rationale", ""), "src": cid})
            else:
                db.execute(text("""
                    INSERT INTO items(stem, choices, correct_index, concept_id, a, b, source_chunk_id)
                    VALUES (
                      :stem,
                      (SELECT ARRAY(SELECT jsonb_array_elements_text(:choices_json::jsonb))),
                      :ci, NULL, 1.0, 0.0, :src
                    )
                """), {"stem": it["stem"], "choices_json": ch_json, "ci": it["correct_index"], "src": cid})
            created += 1

    return {"doc_id": payload.doc_id, "chunks_processed": len(rows), "items_created": created}

class GenItemsForDocQueryReq(BaseModel):
    doc_id: str
    query: str
    per_chunk: int = 1
    max_chunks: int = 10

@router.post("/items/generate_for_doc_query")
def items_generate_for_doc_query(payload: GenItemsForDocQueryReq, db: Session = Depends(get_session)):
    """
    Generate MCQ items for a specific document, focusing on chunks most similar to the query.
    Uses vector search within the doc; falls back to text match if embedding fails.
    """
    # Select candidate chunks within the doc by vector similarity
    try:
        vec = embed_texts([payload.query])[0]
        qvec = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        rows = db.execute(text(
            """
            SELECT c.id, c.text
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.id
            WHERE c.doc_id = :doc
            ORDER BY e.embedding <=> (:qvec)::vector
            LIMIT :lim
            """
        ), {"doc": payload.doc_id, "qvec": qvec, "lim": payload.max_chunks}).fetchall()
        mode = "vector"
    except Exception:
        rows = db.execute(text(
            """
            SELECT c.id, c.text
            FROM chunks c
            WHERE c.doc_id = :doc AND c.text ILIKE '%' || :q || '%'
            ORDER BY c.created_at DESC
            LIMIT :lim
            """
        ), {"doc": payload.doc_id, "q": payload.query, "lim": payload.max_chunks}).fetchall()
        mode = "text"

    if not rows:
        raise HTTPException(status_code=404, detail="No relevant chunks found for query in this document.")

    created = 0
    has_rat = _ensure_items_rationale(db)
    for cid, chunk_text in rows:
        for it in generate_mcqs(chunk_text, n=payload.per_chunk):
            ch_json = json.dumps(it["choices"])  # ensure via JSON->text[]
            if has_rat:
                db.execute(text(
                    """
                    INSERT INTO items(stem, choices, correct_index, rationale, concept_id, a, b, source_chunk_id)
                    VALUES (
                      :stem,
                      (SELECT ARRAY(SELECT jsonb_array_elements_text(:choices_json::jsonb))),
                      :ci, :rat, NULL, 1.0, 0.0, :src)
                    """
                ), {"stem": it["stem"], "choices_json": ch_json, "ci": it["correct_index"], "rat": it.get("rationale", ""), "src": cid})
            else:
                db.execute(text(
                    """
                    INSERT INTO items(stem, choices, correct_index, concept_id, a, b, source_chunk_id)
                    VALUES (
                      :stem,
                      (SELECT ARRAY(SELECT jsonb_array_elements_text(:choices_json::jsonb))),
                      :ci, NULL, 1.0, 0.0, :src)
                    """
                ), {"stem": it["stem"], "choices_json": ch_json, "ci": it["correct_index"], "src": cid})
            created += 1

    return {
        "doc_id": payload.doc_id,
        "query": payload.query,
        "mode": mode,
        "chunks_processed": len(rows),
        "items_created": created,
    }

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
    # Defensive: clear any aborted transaction state left by prior errors in this request
    try:
        db.rollback()
    except Exception:
        pass
    """
    Returns the next unanswered item for this user+doc, or 404 if none left.
    Adaptive selection: choose item whose difficulty b is closest to user's theta.
    """
    user_id = _ensure_user(db, user_email)
    # Ensure learner state exists; default theta=0.0
    theta = db.execute(text("SELECT theta FROM learner_state WHERE user_id = :uid"), {"uid": user_id}).scalar()
    if theta is None:
        db.execute(text("""
            INSERT INTO learner_state(user_id, theta, p_known)
            VALUES (:uid, 0.0, '{}'::jsonb)
            ON CONFLICT (user_id) DO NOTHING
        """), {"uid": user_id})
        theta = 0.0

    # Ensure we have enough items for this user's session (initial 10; then grow toward 100)
    try:
        topup = _top_up_items_for_quiz(db, doc_id, user_id)
    except Exception:
        # Ensure we clear any aborted transaction state
        try:
            db.rollback()
        except Exception:
            pass
        topup = None

    # Stop when user has answered 100 for this doc
    try:
        # Use a fresh short-lived session to avoid any aborted tx state here
        with SessionLocal() as tmp:
            answered_cnt = tmp.execute(text("""
                SELECT count(*) FROM interactions r
                JOIN items i ON i.id = r.item_id
                JOIN chunks c ON c.id = i.source_chunk_id
                WHERE r.user_id = :uid AND c.doc_id::text = :doc
            """), {"uid": user_id, "doc": doc_id}).scalar_one()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        answered_cnt = 0
    if int(answered_cnt) >= 100:
        raise HTTPException(status_code=404, detail="Quiz complete (100 questions answered)")

    # Pick item with minimal exposure for its concept, then closest difficulty to theta
    candidates = db.execute(text("""
        WITH exp AS (
            SELECT i2.concept_id, count(*)::int AS cnt
            FROM interactions r2
            JOIN items i2 ON i2.id = r2.item_id
            WHERE r2.user_id = :uid
            GROUP BY i2.concept_id
        )
        SELECT i.id::text AS item_id, i.stem, i.choices, i.correct_index
        FROM items i
        JOIN chunks c ON c.id = i.source_chunk_id
        LEFT JOIN interactions r ON r.item_id = i.id AND r.user_id = :uid
        LEFT JOIN exp ON exp.concept_id = i.concept_id
        WHERE c.doc_id::text = :doc AND r.id IS NULL
        ORDER BY COALESCE(exp.cnt, 0) ASC, abs(i.b - :theta) ASC, i.a DESC, random()
        LIMIT 10
    """), {"uid": user_id, "doc": doc_id, "theta": float(theta)}).mappings().all()

    if not candidates:
        # Try a small on-demand top-up to unblock if the pool is empty
        try:
            _top_up_items_for_quiz(db, doc_id, user_id, per_chunk=1)
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
        candidates = db.execute(text("""
            WITH exp AS (
                SELECT i2.concept_id, count(*)::int AS cnt
                FROM interactions r2
                JOIN items i2 ON i2.id = r2.item_id
                WHERE r2.user_id = :uid
                GROUP BY i2.concept_id
            )
            SELECT i.id::text AS item_id, i.stem, i.choices, i.correct_index
            FROM items i
            JOIN chunks c ON c.id = i.source_chunk_id
            LEFT JOIN interactions r ON r.item_id = i.id AND r.user_id = :uid
            LEFT JOIN exp ON exp.concept_id = i.concept_id
            WHERE c.doc_id::text = :doc AND r.id IS NULL
            ORDER BY COALESCE(exp.cnt, 0) ASC, abs(i.b - :theta) ASC, i.a DESC, random()
            LIMIT 10
        """), {"uid": user_id, "doc": doc_id, "theta": float(theta)}).mappings().all()
        if not candidates:
            raise HTTPException(status_code=404, detail="No remaining items for this document")

    # Pick first candidate with valid choices; sanitize to remove letter prefixes/placeholders
    for row in candidates:
        ch = row["choices"]
        cleaned = _sanitize_choices(ch) if ch is not None else None
        if not cleaned:
            continue
        # Optional: persist cleaned choices if they changed
        if cleaned != ch:
            try:
                db.execute(text("""
                    UPDATE items
                    SET choices = (SELECT ARRAY(SELECT jsonb_array_elements_text(:ch_json::jsonb)))
                    WHERE id = :iid
                """), {"ch_json": json.dumps(cleaned), "iid": row["item_id"]})
                db.commit()
            except Exception:
                # If this small update fails, rollback and keep serving the cleaned item
                try:
                    db.rollback()
                except Exception:
                    pass
        # Don’t leak the correct index
        return {"item_id": row["item_id"], "stem": row["stem"], "choices": cleaned}

    # If none valid, 404 so client can handle gracefully
    # One more attempt: quick top-up and retry once before failing
    try:
        _top_up_items_for_quiz(db, doc_id, user_id, per_chunk=1)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
    candidates = db.execute(text("""
        WITH exp AS (
            SELECT i2.concept_id, count(*)::int AS cnt
            FROM interactions r2
            JOIN items i2 ON i2.id = r2.item_id
            WHERE r2.user_id = :uid
            GROUP BY i2.concept_id
        )
        SELECT i.id::text AS item_id, i.stem, i.choices, i.correct_index
        FROM items i
        JOIN chunks c ON c.id = i.source_chunk_id
        LEFT JOIN interactions r ON r.item_id = i.id AND r.user_id = :uid
        LEFT JOIN exp ON exp.concept_id = i.concept_id
        WHERE c.doc_id::text = :doc AND r.id IS NULL
        ORDER BY COALESCE(exp.cnt, 0) ASC, abs(i.b - :theta) ASC, i.a DESC, random()
        LIMIT 10
    """), {"uid": user_id, "doc": doc_id, "theta": float(theta)}).mappings().all()
    for row in candidates:
        ch = row["choices"]
        cleaned = _sanitize_choices(ch) if ch is not None else None
        if not cleaned:
            continue
        if cleaned != ch:
            try:
                db.execute(text("""
                    UPDATE items
                    SET choices = (SELECT ARRAY(SELECT jsonb_array_elements_text(:ch_json::jsonb)))
                    WHERE id = :iid
                """), {"ch_json": json.dumps(cleaned), "iid": row["item_id"]})
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
        return {"item_id": row["item_id"], "stem": row["stem"], "choices": cleaned}
    raise HTTPException(status_code=404, detail="No valid items available (malformed choices)")

@router.post("/quiz/answer")
def quiz_answer(payload: QuizAnswerReq, db: Session = Depends(get_session)):
    """
    Records the answer; returns correctness.
    Adaptive update: update learner_state.theta via 2PL IRT gradient step.
    Uses interactions.correct (boolean) and interactions.chosen (int).
    """
    user_id = _ensure_user(db, payload.user_email)

    # Fetch item & correct answer
    # Select rationale only if column exists; always select source_chunk_id
    if _items_has_rationale(db):
        item = db.execute(text("""
            SELECT id, correct_index, a, b, rationale, source_chunk_id FROM items WHERE id = :iid
        """), {"iid": str(payload.item_id)}).first()
    else:
        item = db.execute(text("""
            SELECT id, correct_index, a, b, NULL::text AS rationale, source_chunk_id FROM items WHERE id = :iid
        """), {"iid": str(payload.item_id)}).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    # If already answered, do not double-update; return prior correctness
    prior = db.execute(text("""
        SELECT correct FROM interactions WHERE user_id = :uid AND item_id = :iid LIMIT 1
    """), {"uid": user_id, "iid": item.id}).scalar()
    if prior is not None:
        resp = {"correct": bool(prior), "correct_index": int(item.correct_index)}
        if getattr(item, 'rationale', None):
            resp["rationale"] = item.rationale
        else:
            # Fallback: derive a brief snippet from source chunk
            try:
                snip = db.execute(text("SELECT text FROM chunks WHERE id = :cid"), {"cid": getattr(item, 'source_chunk_id', None)}).scalar()
                if snip:
                    resp["rationale"] = ("Based on the source: " + snip[:200]).strip()
            except Exception:
                pass
        return resp

    # Current theta
    theta = db.execute(text("SELECT theta FROM learner_state WHERE user_id = :uid"), {"uid": user_id}).scalar()
    if theta is None:
        theta = 0.0

    is_correct = (payload.choice_index == item.correct_index)

    # 2PL model: P(correct) = sigmoid(a*(theta - b))
    a = float(getattr(item, 'a', 1.0) or 1.0)
    b = float(getattr(item, 'b', 0.0) or 0.0)
    p = 1.0 / (1.0 + math.exp(-a * (float(theta) - b)))
    y = 1.0 if is_correct else 0.0
    lr = float(os.getenv("ADAPT_LR", "0.2"))
    # Stabilize updates w.r.t discrimination: use gradient scaled independent of 'a'
    theta_new = float(theta) + lr * (y - p)
    # Clamp to typical IRT range
    theta_new = max(-3.0, min(3.0, theta_new))

    # Write interaction (schema column name is 'chosen'), with theta before/after.
    db.execute(text("""
        INSERT INTO interactions(user_id, item_id, chosen, correct, theta_before, theta_after)
        VALUES (:uid, :iid, :ci, :ok, :tb, :ta)
    """), {"uid": user_id, "iid": item.id, "ci": payload.choice_index, "ok": is_correct,
            "tb": float(theta), "ta": float(theta_new)})

    # Upsert learner state
    db.execute(text("""
        INSERT INTO learner_state(user_id, theta, p_known, updated_at)
        VALUES (:uid, :theta, '{}'::jsonb, now())
        ON CONFLICT (user_id) DO UPDATE SET theta = EXCLUDED.theta, updated_at = now()
    """), {"uid": user_id, "theta": float(theta_new)})

    resp = {"correct": bool(is_correct), "correct_index": int(item.correct_index)}
    if getattr(item, 'rationale', None):
        resp["rationale"] = item.rationale
    else:
        # Fallback
        try:
            snip = db.execute(text("SELECT text FROM chunks WHERE id = :cid"), {"cid": getattr(item, 'source_chunk_id', None)}).scalar()
            if snip:
                resp["rationale"] = ("Based on the source: " + snip[:200]).strip()
        except Exception:
            pass
    if os.getenv("ADAPT_DEBUG", "").lower() in ("1","true","yes"):
        resp.update({"p": round(float(p), 4), "theta_before": float(theta), "theta_after": float(theta_new)})
    return resp

@router.get("/learner/state")
def learner_state(user_email: str, doc_id: Optional[str] = None, db: Session = Depends(get_session)):
    """
    Returns the learner's current theta and a naive 95% CI estimate based on Fisher information
    accumulated over answered items. If doc_id is provided, also includes doc-specific accuracy.
    """
    user_id = _ensure_user(db, user_email)
    theta = db.execute(text("SELECT theta FROM learner_state WHERE user_id = :uid"), {"uid": user_id}).scalar()
    if theta is None:
        theta = 0.0

    # Total Fisher information from answered items: sum a^2 * p*(1-p)
    rows = db.execute(text("""
        SELECT i.a, i.b
        FROM interactions r
        JOIN items i ON i.id = r.item_id
        WHERE r.user_id = :uid
    """), {"uid": user_id}).fetchall()
    info = 0.0
    t = float(theta)
    for a, b in rows:
        a = float(a or 1.0)
        b = float(b or 0.0)
        p = 1.0 / (1.0 + math.exp(-a * (t - b)))
        info += a * a * p * (1.0 - p)
    ci = None
    if info > 1e-6:
        # Wald interval: theta ± 1.96 / sqrt(info)
        w = 1.96 / math.sqrt(info)
        ci = [round(t - w, 3), round(t + w, 3)]

    payload = {"theta": round(t, 3), "theta_ci95": ci}

    if doc_id:
        raw_total = db.execute(text("""
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
        desired = min(100, max(10, int(answered) + 10))
        total = min(int(raw_total), desired)
        correct = db.execute(text("""
            SELECT count(*) FROM interactions r
            JOIN items i ON i.id = r.item_id
            JOIN chunks c ON c.id = i.source_chunk_id
            WHERE r.user_id = :uid AND c.doc_id = :doc AND r.correct = true
        """), {"uid": user_id, "doc": doc_id}).scalar_one()
        acc = (correct / answered) if answered else 0.0
        payload.update({
            "doc_id": doc_id,
            "total": total,
            "answered": answered,
            "correct": correct,
            "accuracy": round(acc, 3),
        })

    return payload

@router.get("/quiz/progress")
def quiz_progress(doc_id: str, user_email: str, db: Session = Depends(get_session)):
    """
    Returns counts for answered/total and accuracy for this doc.
    """
    user_id = _ensure_user(db, user_email)

    raw_total = db.execute(text("""
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

    desired = min(100, max(10, int(answered) + 10))
    total = min(int(raw_total), desired)

    correct = db.execute(text("""
        SELECT count(*) FROM interactions r
        JOIN items i ON i.id = r.item_id
        JOIN chunks c ON c.id = i.source_chunk_id
        WHERE r.user_id = :uid AND c.doc_id = :doc AND r.correct = true
    """), {"uid": user_id, "doc": doc_id}).scalar_one()

    acc = (correct / answered) if answered else 0.0
    return {"total": total, "answered": answered, "correct": correct, "accuracy": round(acc, 3)}

@router.get("/quiz/recap")
def quiz_recap(doc_id: str, user_email: str, db: Session = Depends(get_session)):
    """Return recap including progress, theta, and simple study recommendations.

    Recommendations: top 3 chunks from the doc related to items the user missed (by recent misses).
    """
    # Base progress
    prog = quiz_progress(doc_id=doc_id, user_email=user_email, db=db)

    # Theta and CI
    state = learner_state(user_email=user_email, doc_id=doc_id, db=db)

    # Pull last 10 incorrect interactions for this doc and suggest related chunks
    user_id = _ensure_user(db, user_email)
    missed_chunk_rows = db.execute(text(
        """
        SELECT c.id, c.text
        FROM interactions r
        JOIN items i ON i.id = r.item_id
        JOIN chunks c ON c.id = i.source_chunk_id
        WHERE r.user_id = :uid AND r.correct = false AND c.doc_id = :doc
        ORDER BY r.created_at DESC
        LIMIT 10
        """
    ), {"uid": user_id, "doc": doc_id}).fetchall()

    recs = []
    for cid, ctext in missed_chunk_rows[:3]:
        recs.append({
            "chunk_id": str(cid),
            "snippet": (ctext or "").strip()[:280]
        })

    return {
        "progress": prog,
        "state": state,
        "recommendations": recs,
    }
