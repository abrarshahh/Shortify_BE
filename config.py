import os
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ---------- CONFIG ----------
DATABASE_URL = "postgresql://admin: VIDEO_PASS@localhost:5432/shortify_ai"
SECRET_KEY = "supersecret"
ALGORITHM = "HS256"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

STORAGE_ROOT = Path("storage")
STORAGE_ROOT.mkdir(exist_ok=True)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ---------- LOGGING ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
