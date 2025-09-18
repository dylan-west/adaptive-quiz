# apps/backend/app/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from typing import Generator
from .config import get_database_url

engine = create_engine(get_database_url(), pool_pre_ping=True)

class Base(DeclarativeBase):
    pass

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db           # handlers can use the session
        db.commit()        # <-- commit on success
    except:
        db.rollback()      # <-- rollback on error
        raise
    finally:
        db.close()

