-- Core entities
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  meta JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Text chunks from PDFs with metadata
CREATE TABLE IF NOT EXISTS chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  doc_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  text TEXT NOT NULL,
  page_start INT,
  page_end INT,
  headings TEXT[],
  concept_tags TEXT[],
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Embeddings table (one row per chunk)
-- Size uses env EMBEDDING_DIM in app code; here we default to 1536.
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  embedding vector(1536) NOT NULL,
  model_name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Concepts taxonomy
CREATE TABLE IF NOT EXISTS concepts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT UNIQUE NOT NULL,
  parent_id UUID REFERENCES concepts(id)
);

-- Items (questions)
CREATE TABLE IF NOT EXISTS items (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  stem TEXT NOT NULL,
  choices TEXT[] NOT NULL,
  correct_index INT NOT NULL,
  concept_id UUID REFERENCES concepts(id),
  a REAL DEFAULT 1.0,   -- IRT discrimination
  b REAL DEFAULT 0.0,   -- IRT difficulty
  source_chunk_id UUID REFERENCES chunks(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Learner state
CREATE TABLE IF NOT EXISTS learner_state (
  user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  theta REAL NOT NULL DEFAULT 0.0, -- IRT ability
  p_known JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Interactions (responses)
CREATE TABLE IF NOT EXISTS interactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  item_id UUID NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  chosen INT NOT NULL,
  correct BOOLEAN NOT NULL,
  latency_ms INT,
  theta_before REAL,
  theta_after REAL,
  p_known_before JSONB,
  p_known_after JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_concepts ON chunks USING GIN (concept_tags);
CREATE INDEX IF NOT EXISTS idx_items_concept ON items(concept_id);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);

-- Vector index (choose one; ivfflat requires list training)
-- HNSW works without training and is great for dev:
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops);
