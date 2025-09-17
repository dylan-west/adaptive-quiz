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
- Step 2 will add Docker/Postgres.
- Step 3 will scaffold FastAPI.
- Step 4 will scaffold Next.js.
