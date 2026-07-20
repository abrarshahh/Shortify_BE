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

# Enable CORS for Next.js dev server origin and public tunnel subdomains
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
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

@app.get("/debug/storage")
def debug_storage():
    import os
    files_list = []
    for root, dirs, files in os.walk("storage"):
        for f in files:
            full_path = os.path.join(root, f)
            files_list.append({
                "path": full_path,
                "size_bytes": os.path.getsize(full_path)
            })
    return {
        "cwd": os.getcwd(),
        "storage_exists": os.path.exists("storage"),
        "files_count": len(files_list),
        "files": files_list
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
