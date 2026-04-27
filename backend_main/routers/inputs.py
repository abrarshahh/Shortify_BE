import os, shutil, uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from backend_main.config import SessionLocal, STORAGE_ROOT, logger
from backend_main.models import Project, MediaAsset, User
from backend_main.auth import get_current_user
from backend_main.schemas import ProjectCreate, ProjectResponse, UploadResponse, MediaResponse
from pydantic import BaseModel
from typing import List

class AddMediaRequest(BaseModel):
    media_ids: List[str]

router = APIRouter(tags=["Inputs"])

def save_upload_file(user_id: str, upload_file: UploadFile) -> str:
    ext = Path(upload_file.filename).suffix
    media_id = uuid.uuid4()
    rel_dir = Path(f"users/{user_id}/media")
    full_dir = STORAGE_ROOT / rel_dir
    full_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{media_id}{ext}"
    full_path = full_dir / filename
    with full_path.open("wb") as f:
        f.write(upload_file.file.read())
    return str((rel_dir / filename))  # relative path to store in DB

def validate_file(upload_file: UploadFile) -> None:
    mime = upload_file.content_type
    if not mime.startswith("video/") and not mime.startswith("image/") and not mime.startswith("audio/"):
        raise HTTPException(400, "Invalid file type")
    # Limit max file size (e.g. 200MB)
    max_size = 200 * 1024 * 1024  # 200MB
    file_size = len(upload_file.file.read())
    upload_file.file.seek(0)  # Reset file pointer
    if file_size > max_size:
        raise HTTPException(400, "File too large")

@router.post("/projects", response_model=ProjectResponse, status_code=201)
def create_project(
    project_data: ProjectCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    proj = Project(
        user_id=user.id,
        title=project_data.title,
        description=project_data.description,
        target_duration=project_data.target_duration,
        aspect_ratio=project_data.aspect_ratio,
        style=project_data.style
    )
    db.add(proj)
    db.commit()
    db.refresh(proj)
    logger.info(f"User {user.username} created project: {proj.id}")
    return ProjectResponse(
        id=str(proj.id),
        title=proj.title,
        description=proj.description,
        target_duration=proj.target_duration,
        aspect_ratio=proj.aspect_ratio,
        style=proj.style,
        music_id=str(proj.music_id) if proj.music_id else None,
        created_at=proj.created_at
    )

@router.post("/projects/{project_id}/media", response_model=UploadResponse, status_code=201)
def upload_media(
    project_id: str,
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    # Basic checks: project exists and is owned by user
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Unauthorized")
    saved = []
    for upload in files:
        validate_file(upload)
        rel_path = save_upload_file(str(user.id), project_id, upload)  # helper from earlier
        # Get file size
        full_path = STORAGE_ROOT / rel_path
        file_size = full_path.stat().st_size
        asset = MediaAsset(
            project_id=project_id,
            user_id=str(user.id),
            original_filename=upload.filename,
            storage_path=rel_path,
            mime_type=upload.content_type,
            file_size=file_size
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
        saved.append({"id": str(asset.id), "path": asset.storage_path})
    logger.info(f"User {user.username} uploaded {len(saved)} media files to project {project_id}")
    return UploadResponse(uploaded=saved)

@router.post("/projects/{project_id}/music", response_model=UploadResponse, status_code=201)
def upload_music(
    project_id: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    # Basic checks: project exists and is owned by user
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Unauthorized")
    # Validate music file (audio only)
    if not file.content_type.startswith("audio/"):
        raise HTTPException(400, "Invalid file type for music")
    max_size = 50 * 1024 * 1024  # 50MB for music
    file_size = len(file.file.read())
    file.file.seek(0)
    if file_size > max_size:
        raise HTTPException(400, "Music file too large")
    rel_path = save_upload_file(str(user.id), file)
    full_path = STORAGE_ROOT / rel_path
    file_size = full_path.stat().st_size
    asset = MediaAsset(
        project_id=project_id,
        user_id=str(user.id),
        original_filename=file.filename,
        storage_path=rel_path,
        mime_type=file.content_type,
        file_size=file_size
    )
    db.add(asset)
    db.commit()
    db.refresh(asset)
    # Update project music_id
    project.music_id = asset.id
    db.commit()
    logger.info(f"User {user.username} uploaded music for project {project_id}: {asset.id}")
    return UploadResponse(uploaded=[{"id": str(asset.id), "path": asset.storage_path}])



@router.put("/projects/{project_id}/music")
def select_music(
    project_id: str,
    music_id: str = Form(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project or project.user_id != user.id:
        raise HTTPException(404, "Project not found")
    # Check if music_id exists and is owned by user
    music = db.query(MediaAsset).filter(MediaAsset.id == music_id, MediaAsset.user_id == str(user.id)).first()
    if not music:
        raise HTTPException(404, "Music not found")
    project.music_id = music_id
    db.commit()
    logger.info(f"User {user.username} selected music {music_id} for project {project_id}")
    return {"message": "Music selected"}

@router.put("/projects/{project_id}/media")
def add_media_to_project(
    project_id: str,
    request: AddMediaRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project or project.user_id != user.id:
        raise HTTPException(404, "Project not found")
    # Check if all media_ids exist and are owned by user
    media_assets = db.query(MediaAsset).filter(MediaAsset.id.in_(request.media_ids), MediaAsset.user_id == str(user.id)).all()
    if len(media_assets) != len(request.media_ids):
        raise HTTPException(404, "Some media not found or not owned by user")
    # Update project_id for these assets
    for asset in media_assets:
        asset.project_id = project_id
    db.commit()
    logger.info(f"User {user.username} added {len(media_assets)} media to project {project_id}")
    return {"message": f"Added {len(media_assets)} media to project"}

@router.get("/media", response_model=List[MediaResponse])
def list_user_media(
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    media = db.query(MediaAsset).filter(MediaAsset.user_id == str(user.id), ~MediaAsset.mime_type.startswith("audio/")).all()
    return [
        MediaResponse(
            id=str(m.id),
            original_filename=m.original_filename,
            storage_path=m.storage_path,
            mime_type=m.mime_type,
            file_size=m.file_size,
            duration=m.duration,
            thumbnail_path=m.thumbnail_path,
            uploaded_at=m.uploaded_at
        ) for m in media
    ]

@router.get("/music", response_model=List[MediaResponse])
def list_user_music(
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    music = db.query(MediaAsset).filter(MediaAsset.user_id == str(user.id), MediaAsset.mime_type.startswith("audio/")).all()
    return [
        MediaResponse(
            id=str(m.id),
            original_filename=m.original_filename,
            storage_path=m.storage_path,
            mime_type=m.mime_type,
            file_size=m.file_size,
            duration=m.duration,
            thumbnail_path=m.thumbnail_path,
            uploaded_at=m.uploaded_at
        ) for m in music
    ]
