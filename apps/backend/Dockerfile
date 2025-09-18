# syntax=docker/dockerfile:1.7
FROM python:3.11-slim as base

# Faster, repeatable builds
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for psycopg and fitz (PyMuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libfreetype6 \
    libjpeg62-turbo \
    libpng16-16 \
    libxcb1 \
    libc6 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first for better cache
COPY apps/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY apps/backend/app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
