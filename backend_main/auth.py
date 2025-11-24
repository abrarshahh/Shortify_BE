from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from backend_main.config import SessionLocal
from backend_main.models import User

# ---------- AUTH ----------
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    session_id = credentials.credentials
    db = SessionLocal()
    user = db.query(User).filter(User.session_id == session_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session_id")
    return user
