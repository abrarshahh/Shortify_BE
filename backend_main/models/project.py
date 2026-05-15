import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from backend_main.config import Base

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
    last_output_path = Column(String, nullable=True) # Path to the last generated video
    is_rendering = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="projects")
    media_assets = relationship("MediaAsset", secondary="project_media_assets", back_populates="projects")
    reels = relationship("ReelJob", back_populates="project")
