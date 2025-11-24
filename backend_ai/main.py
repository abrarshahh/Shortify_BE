from fastapi import FastAPI

app = FastAPI()

# Placeholder for future AI-related routers and logic
# from backend_ai.routers import ai_router

# app.include_router(ai_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # different port from backend_main
