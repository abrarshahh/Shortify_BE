from fastapi import FastAPI
from backend_main.routers import (
    auth_router, projects_router, render_router, media_router, audio_router
)

# ---------- FASTAPI APP ----------
app = FastAPI(title="Shortify AI", version="1.0.0")

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(media_router)
app.include_router(audio_router)
app.include_router(render_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
