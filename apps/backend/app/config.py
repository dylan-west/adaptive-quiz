# apps/backend/app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env from the current working directory (run uvicorn from repo root)
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    # Reads the env var named DATABASE_URL automatically
    database_url: str

settings = Settings()

