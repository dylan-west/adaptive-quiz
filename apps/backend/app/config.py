# apps/backend/app/config.py
import os
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env from the current working directory (run uvicorn from repo root)
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    # Optional; if missing, we'll construct from POSTGRES_* vars.
    database_url: str | None = None

def get_database_url() -> str:
    """Return a SQLAlchemy URL. Prefer DATABASE_URL; else build from POSTGRES_*.

    Handles special characters in password via URL encoding.
    """
    # 1) Direct DATABASE_URL wins
    direct = os.getenv("DATABASE_URL")
    if direct:
        return direct

    # 2) Construct from POSTGRES_* (works in Docker compose)
    user = os.getenv("POSTGRES_USER", "adaptive")
    pwd = os.getenv("POSTGRES_PASSWORD", "adaptive")
    host = os.getenv("POSTGRES_HOST") or os.getenv("DB_HOST") or "db"
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "adaptive_quiz")
    # URL-encode username and password to handle special characters
    user_enc = quote_plus(user)
    pwd_enc = quote_plus(pwd)
    return f"postgresql+psycopg://{user_enc}:{pwd_enc}@{host}:{port}/{db}"

settings = Settings()

