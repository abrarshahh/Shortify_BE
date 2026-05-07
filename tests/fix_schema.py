from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    print("Fixing schema: Dropping old project_id column from media_assets...")
    try:
        # Cascade will handle the foreign key constraint as well
        conn.execute(text("ALTER TABLE media_assets DROP COLUMN IF EXISTS project_id CASCADE;"))
        conn.commit()
        print("Success: project_id column dropped.")
    except Exception as e:
        print(f"Error: {e}")
