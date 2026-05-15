import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from backend_main.config import Base

class ReelJob(Base):
    __tablename__ = "reels"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(PG_UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    target_duration = Column(Integer, nullable=False)
    aspect_ratio = Column(String, nullable=False, default="9:16")
    style = Column(String, nullable=True)
    music_asset_id = Column(PG_UUID(as_uuid=True), nullable=True)
    timeline_json = Column(JSON, nullable=True)
    status = Column(String, nullable=False, default="pending")
    render_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    error_message = Column(String, nullable=True)
    user = relationship("User", back_populates="reels")
    project = relationship("Project", back_populates="reels")
