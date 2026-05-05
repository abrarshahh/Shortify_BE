from fastapi import FastAPI
from backend_main.routers import auth, inputs, render

# ---------- FASTAPI APP ----------
app = FastAPI(title="Shortify AI", version="1.0.0")

app.include_router(auth.router)
app.include_router(inputs.router)
app.include_router(render.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
