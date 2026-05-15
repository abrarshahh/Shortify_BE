import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from backend_main.config import Base

class User(Base):
    __tablename__ = "users"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String, unique=True)
    login_history = relationship("LoginHistory", back_populates="user")
    projects = relationship("Project", back_populates="user")
    media_assets = relationship("MediaAsset", back_populates="user")
    reels = relationship("ReelJob", back_populates="user")

class LoginHistory(Base):
    __tablename__ = "login_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    login_time = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="login_history")
