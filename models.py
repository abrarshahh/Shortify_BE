from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from config import Base, engine

# ---------- MODELS ----------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String, unique=True)
    login_history = relationship("LoginHistory", back_populates="user")
    media = relationship("Media", back_populates="user")
    audio = relationship("Audio", back_populates="user")
    moods = relationship("Mood", back_populates="user")

class Media(Base):
    __tablename__ = "media"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    type = Column(String)  # video/image
    filepath = Column(Text)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")

class Audio(Base):
    __tablename__ = "audio"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filepath = Column(Text)
    duration = Column(Integer)  # seconds
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")

class Mood(Base):
    __tablename__ = "moods"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mood_text = Column(String)
    user = relationship("User")

class LoginHistory(Base):
    __tablename__ = "login_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    login_time = Column(DateTime, default=datetime.utcnow)
    user = relationship("User")

Base.metadata.create_all(bind=engine)
