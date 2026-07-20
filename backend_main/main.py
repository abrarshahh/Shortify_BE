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

@app.get("/storage/{file_path:path}")
def get_storage_file(file_path: str):
    import os
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi import HTTPException
    
    local_file = os.path.join("storage", file_path)
    
    # 1. If the file exists locally, serve it directly
    if os.path.exists(local_file) and os.path.isfile(local_file):
        return FileResponse(local_file)
        
    # 2. Otherwise, if Supabase is configured, redirect to the permanent copy
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_bucket = os.getenv("SUPABASE_BUCKET", "shortify")
    if supabase_url:
        public_url = f"{supabase_url}/storage/v1/object/public/{supabase_bucket}/{file_path}"
        return RedirectResponse(url=public_url)
        
    # 3. Fallback
    raise HTTPException(status_code=404, detail="File not found")

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
