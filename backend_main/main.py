from fastapi import FastAPI
from backend_main.routers import (
    auth_router, projects_router, render_router, media_router, audio_router
)

from backend_main.config import Base, engine

from sqlalchemy import text

# ---------- FASTAPI APP ----------
Base.metadata.create_all(bind=engine)

# Dynamic self-healing database schema migrations
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS render_progress INTEGER DEFAULT 0;"))
        conn.execute(text("ALTER TABLE projects ADD COLUMN IF NOT EXISTS render_step VARCHAR;"))
        conn.commit()
    except Exception as e:
        print(f"Self-healing database schema migration warning: {e}")

app = FastAPI(title="Shortify AI", version="1.0.0")

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(media_router)
app.include_router(audio_router)
app.include_router(render_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
