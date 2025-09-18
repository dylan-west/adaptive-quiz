# Adaptive Quiz Frontend

A minimal Next.js (App Router) UI for the Adaptive Quiz backend.

## Setup

1. Create `.env.local` in `apps/frontend`:

```
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

2. Install deps and run the dev server:

```
npm install
npm run dev
```

## Pages
- `/` Home
- `/upload` Upload a PDF (uses OCR when backend configured)
- `/search` Query chunks
- `/quiz` Run adaptive quiz (next/answer/progress + learner/state)
