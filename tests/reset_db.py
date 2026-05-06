"""
One-time script: drops all existing tables and recreates them with the correct schema.
Run this from the project root: python reset_db.py
"""
from backend_main.config import Base, engine
from sqlalchemy import text

print("Dropping all existing tables...")
with engine.begin() as conn:
    # Drop in reverse dependency order to avoid FK constraint errors
    conn.execute(text("DROP TABLE IF EXISTS login_history CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS media_assets CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS projects CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))
    print("  All tables dropped.")

print("Recreating tables with updated schema...")
# Import models so they register themselves on Base.metadata
import backend_main.models  # noqa: F401
Base.metadata.create_all(bind=engine)
print("  Done! All tables created with correct UUID schema.")
