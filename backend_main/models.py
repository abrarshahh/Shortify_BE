import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, DateTime, ForeignKey, Boolean, Enum, JSON, BigInteger
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from backend_main.config import Base, engine

# ---------- MODELS ----------
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
    # Removed reels relationship to avoid coupling with backend_ai

class ProjectMediaAsset(Base):
    __tablename__ = "project_media_assets"
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), primary_key=True)
    media_asset_id = Column(PG_UUID(as_uuid=True), ForeignKey("media_assets.id"), primary_key=True)
    added_at = Column(DateTime, default=datetime.utcnow)

class Project(Base):
    __tablename__ = "projects"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    description = Column(String, nullable=True)
    target_duration = Column(Integer, nullable=False)  # seconds
    aspect_ratio = Column(String, nullable=False, default="9:16")
    style = Column(String, nullable=True)
    music_id = Column(PG_UUID(as_uuid=True), ForeignKey("media_assets.id"), nullable=True)  # FK to MediaAsset
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="projects")
    media_assets = relationship("MediaAsset", secondary="project_media_assets", back_populates="projects")
    # Removed reels relationship to avoid coupling with backend_ai

class MediaAsset(Base):
    __tablename__ = "media_assets"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    original_filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)  # relative path on disk
    mime_type = Column(String, nullable=False)
    file_size = Column(BigInteger, nullable=False)
    duration = Column(Integer, nullable=True)  # seconds, for videos
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    thumbnail_path = Column(String, nullable=True)
    extra_metadata = Column(JSON, nullable=True)  # e.g. faces, tags
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="media_assets")
    projects = relationship("Project", secondary="project_media_assets", back_populates="media_assets")

class LoginHistory(Base):
    __tablename__ = "login_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    login_time = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="login_history")

Base.metadata.create_all(bind=engine)
