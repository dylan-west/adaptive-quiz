# Adaptive Quiz (RAG + IRT/BKT)
End-to-end app: upload PDFs, OCR/parse → embed → semantic search → adaptive quizzes with grounded explanations.

## Structure
- apps/frontend  — Next.js UI (uploads, quiz player, progress)
- apps/backend   — FastAPI APIs (ingestion, embeddings, retrieval, IRT/BKT, sessions)
- infra/db       — Postgres + pgvector (docker-compose)
- data           — seeds, migrations, test PDFs
- notebooks      — experiments (chunking, tagging, IRT/BKT tuning)
- scripts        — dev utilities

## Quick Start
Run the whole stack with Docker (db + backend + frontend):

1) Create a .env file at repo root:

	Required database vars (also used by the db container):
	- POSTGRES_USER=adaptive
	- POSTGRES_PASSWORD=adaptive
	- POSTGRES_DB=adaptive_quiz
	- POSTGRES_PORT=5432

	Backend settings:
	- DATABASE_URL=postgresql+psycopg://adaptive:adaptive@db:5432/adaptive_quiz
	- QGEN_PROVIDER=openai
	- OPENAI_API_KEY=sk-... (optional, enables OCR + item generation)

	Frontend setting (public):
	- NEXT_PUBLIC_API_BASE=http://localhost:8000

2) Start everything:

```sh
docker compose up -d --build
```

3) Open services:
- API: http://localhost:8000 (GET / -> { status: running }, GET /healthz)
- Web: http://localhost:3000

Notes:
- The database init scripts in infra/db/init run only on first boot of a fresh volume.
- If your existing DB volume predates the items.rationale column, the backend auto-adds it at runtime. To apply the SQL migration via init, remove the volume and re-up, or run the SQL manually from infra/db/init/004_add_rationale.sql.
- For local (non-Docker) dev, you can still run: `uvicorn apps.backend.app.main:app --reload` and `npm run dev` in apps/frontend, provided DATABASE_URL points at localhost:5432.

