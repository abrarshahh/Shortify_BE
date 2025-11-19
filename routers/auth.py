import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from config import SessionLocal, logger
from models import User, LoginHistory
from passlib.hash import bcrypt

router = APIRouter(tags=["Account"])

# ----- Signup -----
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str

# ----- Login -----
class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/signup")
def signup(data: SignupRequest):
    db = SessionLocal()
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="User already exists")
    session_id = str(uuid.uuid4())
    hashed_pw = bcrypt.hash(data.password)
    user = User(username=data.username, email=data.email, password_hash=hashed_pw, session_id=session_id)
    db.add(user)
    db.commit()
    logger.info(f"User {data.username} signed up with session_id {session_id}")
    return {"message": "User created", "session_id": session_id}

# ----- Login -----
@router.post("/login")
def login(data: LoginRequest):
    db = SessionLocal()
    user = db.query(User).filter(User.username == data.username).first()
    if not user or not bcrypt.verify(data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Log login history
    login_entry = LoginHistory(user_id=user.id)
    db.add(login_entry)
    db.commit()
    logger.info(f"User {data.username} logged in")
    return {"status": "ok", "session_id": user.session_id}
