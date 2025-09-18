import os
import math
import hashlib
from typing import List
from .config import settings

PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def _normalize(vec: List[float]) -> List[float]:
    # cosine likes normalized vectors; pgvector also works without, but normalize for consistency
    s = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / s for v in vec]

def embed_texts(texts: List[str]) -> List[List[float]]:
    if PROVIDER == "fake":
        # Deterministic, non-semantic vectors for dev. Do NOT use in prod.
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # repeat hash to fill DIM
            raw = []
            while len(raw) < DIM:
                raw.extend(h)
                h = hashlib.sha256(h).digest()
            # map bytes to [-0.5, 0.5)
            vec = [(b / 255.0) - 0.5 for b in raw[:DIM]]
            out.append(_normalize(vec))
        return out

    # OpenAI provider
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env
    resp = client.embeddings.create(model=MODEL, input=texts)
    return [_normalize(d.embedding) for d in resp.data]
