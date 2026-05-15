import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON, BigInteger
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from backend_main.config import Base

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
