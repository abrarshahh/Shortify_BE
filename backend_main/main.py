import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
from backend_main.routers import (
    auth_router, projects_router, render_router, media_router, audio_router
)

from backend_main.config import Base, engine

# ---------- FASTAPI APP ----------
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Shortify AI", version="1.0.0")

# Enable CORS for Next.js dev server origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.29.91:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/storage", StaticFiles(directory="storage"), name="storage")

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(media_router)
app.include_router(audio_router)
app.include_router(render_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
