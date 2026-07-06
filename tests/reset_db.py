"""
One-time script: drops all existing tables and recreates them with the correct schema.
Run this from the project root: python reset_db.py
"""
from backend_main.config import Base, engine
from sqlalchemy import text

import os
import shutil

# Clean temporary storage, data, and cache directories
dirs_to_delete = [
    "cache",
    "storage",
    "data",
    ".pytest_cache",
]

print("Cleaning storage, data, and cache directories...")
for d in dirs_to_delete:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
            print(f"  Deleted directory: {d}")
        except Exception as e:
            print(f"  Error deleting {d}: {e}")

# Recreate the base directories and their expected structure
dirs_to_recreate = [
    "cache",
    "storage",
    "storage/users",
    "storage/exports",
    "data",
    "data/exports",
    "data/fonts",
    "data/local_effects",
    "data/local_stickers",
    "data/luts",
    "data/models",
]

print("Recreating base directories...")
for d in dirs_to_recreate:
    try:
        os.makedirs(d, exist_ok=True)
        print(f"  Recreated: {d}")
    except Exception as e:
        print(f"  Error recreating {d}: {e}")

# Clean log files
logs_dir = "logs"
if os.path.exists(logs_dir):
    print("Cleaning log files...")
    for f in os.listdir(logs_dir):
        file_path = os.path.join(logs_dir, f)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"  Removed log file: {file_path}")
            except Exception as e:
                try:
                    with open(file_path, "w") as out:
                        out.truncate(0)
                    print(f"  Truncated locked log file: {file_path}")
                except Exception as e2:
                    print(f"  Error cleaning log file {file_path}: {e2}")
else:
    os.makedirs(logs_dir, exist_ok=True)
    print(f"  Created logs directory: {logs_dir}")

print("Dropping all existing tables...")
with engine.begin() as conn:
    # Drop in reverse dependency order to avoid FK constraint errors
    conn.execute(text("DROP TABLE IF EXISTS reels CASCADE"))
    conn.execute(text("DROP TABLE IF EXISTS project_media_assets CASCADE"))
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
