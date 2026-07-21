import logging
from fastapi import FastAPI, Request
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
async def get_storage_file(file_path: str, request: Request):
    import os
    import httpx
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi import HTTPException
    
    local_file = os.path.join("storage", file_path)
    
    # 1. If the file exists locally, serve it directly
    if os.path.exists(local_file) and os.path.isfile(local_file):
        return FileResponse(local_file)
        
    # 2. Otherwise, proxy the file stream from Supabase (avoids CORS redirect blocks)
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_bucket = os.getenv("SUPABASE_BUCKET", "shortify")
    if supabase_url:
        public_url = f"{supabase_url}/storage/v1/object/public/{supabase_bucket}/{file_path}"
        
        # Forward range headers to Supabase to support HTML5 video player scrubbing/seeking
        headers = {}
        range_header = request.headers.get("range")
        if range_header:
            headers["range"] = range_header
            
        async def stream_chunks():
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("GET", public_url, headers=headers) as r:
                    if r.status_code >= 400:
                        raise HTTPException(status_code=r.status_code, detail="Failed to fetch file from storage")
                    async for chunk in r.iter_bytes():
                        yield chunk
                        
        # Fetch the metadata/headers from the resource using a quick HEAD request
        try:
            async with httpx.AsyncClient() as client:
                head_resp = await client.head(public_url, headers=headers)
                content_type = head_resp.headers.get("content-type", "application/octet-stream")
                content_length = head_resp.headers.get("content-length")
                accept_ranges = head_resp.headers.get("accept-ranges")
                content_range = head_resp.headers.get("content-range")
                status_code = head_resp.status_code
        except Exception as e:
            logger.error(f"[Storage Proxy] Failed to HEAD public URL {public_url}: {e}")
            content_type = "application/octet-stream"
            content_length = None
            accept_ranges = "bytes"
            content_range = None
            status_code = 200
            
        resp_headers = {
            "Content-Type": content_type,
        }
        if content_length:
            resp_headers["Content-Length"] = content_length
        if accept_ranges:
            resp_headers["Accept-Ranges"] = accept_ranges
        if content_range:
            resp_headers["Content-Range"] = content_range
            
        return StreamingResponse(
            stream_chunks(),
            status_code=status_code,
            headers=resp_headers
        )
        
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
