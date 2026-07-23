"""
One-time script: drops all existing tables and recreates them with the correct schema.
Run this from the project root: python reset_db.py
"""
from backend_main.config import Base, engine
from sqlalchemy import text

import os
import shutil
from dotenv import load_dotenv

load_dotenv()

# Clean storage directories based on USE_SUPABASE environment variable
use_supabase = os.getenv("USE_SUPABASE", "false").strip().lower() == "true"

if use_supabase:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "shortify")
    
    if SUPABASE_URL and SUPABASE_KEY:
        print(f"USE_SUPABASE is true. Cleaning Supabase Storage bucket '{SUPABASE_BUCKET}'...")
        import httpx
        url_list = f"{SUPABASE_URL}/storage/v1/object/list/{SUPABASE_BUCKET}"
        url_delete = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        all_files = []
        
        def traverse(prefix):
            payload = {
                "prefix": prefix,
                "limit": 100,
                "sortBy": {"column": "name", "order": "asc"}
            }
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url_list, headers=headers, json=payload)
                    if resp.status_code != 200:
                        print(f"Error listing prefix '{prefix}': {resp.status_code} {resp.text}")
                        return
                    items = resp.json()
                    for item in items:
                        name = item.get("name")
                        if not name:
                            continue
                        item_path = f"{prefix}/{name}" if prefix else name
                        if item.get("metadata") is None and item.get("id") is None:
                            # Subfolder/Directory
                            traverse(item_path)
                        else:
                            # File
                            all_files.append(item_path)
            except Exception as exc:
                print(f"Exception listing prefix '{prefix}': {exc}")
                
        traverse("")
        
        if all_files:
            print(f"Found {len(all_files)} files in Supabase bucket to delete.")
            for i in range(0, len(all_files), 100):
                chunk = all_files[i:i+100]
                payload = {"prefixes": chunk}
                try:
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.request("DELETE", url_delete, headers=headers, json=payload)
                        if resp.status_code == 200:
                            print(f"  Successfully deleted files: {chunk}")
                        else:
                            print(f"  Failed to delete files {chunk}. Status: {resp.status_code}, Body: {resp.text}")
                except Exception as exc:
                    print(f"Exception deleting files {chunk}: {exc}")
        else:
            print("Supabase bucket is already empty.")
    else:
        print("USE_SUPABASE is true, but SUPABASE_URL or SUPABASE_KEY is missing. Skipping Supabase clean.")
else:
    print("USE_SUPABASE is false. Cleaning local storage, data, and cache directories...")
    dirs_to_delete = [
        "cache",
        "storage",
        "data",
        ".pytest_cache",
    ]
    for d in dirs_to_delete:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                print(f"  Deleted local directory: {d}")
            except Exception as e:
                print(f"  Error deleting local directory {d}: {e}")

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
