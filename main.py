from fastapi import FastAPI
from routers import auth, inputs

# ---------- FASTAPI APP ----------
app = FastAPI()

app.include_router(auth.router)
app.include_router(inputs.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
