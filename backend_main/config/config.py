import os
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
load_dotenv()

# ---------- CONFIG ----------
database_url=os.getenv("DATABASE_URL")



DATABASE_URL = database_url
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

STORAGE_ROOT = Path("storage")
STORAGE_ROOT.mkdir(exist_ok=True)

# ---------- LOGGING ----------
from backend_ai.core.logging_config import setup_logging
setup_logging()
logger = logging.getLogger("backend_main")
