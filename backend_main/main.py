import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)
from backend_main.routers import (
    auth_router, projects_router, render_router, media_router, audio_router
)

from backend_main.config import Base, engine

# ---------- FASTAPI APP ----------
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Shortify AI", version="1.0.0")

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(media_router)
app.include_router(audio_router)
app.include_router(render_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
