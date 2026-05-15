import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from backend_main.config import SessionLocal, logger, STORAGE_ROOT
from backend_main.models import Project, MediaAsset, User, ProjectMediaAsset
from backend_main.auth import get_current_user
from backend_main.schemas import MediaResponse, UploadResponse, MediaLinkRequest, AudioLinkRequest
from backend_main.utils import save_upload_file, validate_file

router = APIRouter(prefix="/audio", tags=["Audio"])

@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_audio_to_library(
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    saved = []
    for upload in files:
        validate_file(upload)
        # Check if it IS audio
        if not upload.content_type.startswith("audio/"):
            continue
            
        rel_path = save_upload_file(str(user.id), upload)
        full_path = STORAGE_ROOT / rel_path
        file_size = full_path.stat().st_size
        
        asset = MediaAsset(
            user_id=user.id,
            original_filename=upload.filename,
            storage_path=rel_path,
            mime_type=upload.content_type,
            file_size=file_size
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
        saved.append({"id": str(asset.id), "path": asset.storage_path})
    
    return UploadResponse(uploaded=saved)

@router.post("/project/{project_id}/upload", response_model=UploadResponse, status_code=201)
def upload_audio_to_project(
    project_id: uuid.UUID,
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    saved = []
    for upload in files:
        validate_file(upload)
        if not upload.content_type.startswith("audio/"):
            continue
        rel_path = save_upload_file(str(user.id), upload)
        full_path = STORAGE_ROOT / rel_path
        file_size = full_path.stat().st_size
        
        asset = MediaAsset(
            user_id=user.id,
            original_filename=upload.filename,
            storage_path=rel_path,
            mime_type=upload.content_type,
            file_size=file_size
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
        
        # Set as the project's music_id (only one allowed)
        project.music_id = asset.id
        saved.append({"id": str(asset.id), "path": asset.storage_path})
    
    db.commit()
    return UploadResponse(uploaded=saved)

@router.post("/project/{project_id}/link", response_model=UploadResponse, status_code=201)
def link_audio_to_project(
    project_id: uuid.UUID,
    request_data: AudioLinkRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    saved = []
    a_id = request_data.audio_id
    asset = db.query(MediaAsset).filter(MediaAsset.id == a_id, MediaAsset.user_id == user.id).first()
    if asset and asset.mime_type.startswith("audio/"):
        project.music_id = asset.id
        saved.append({"id": str(asset.id), "path": asset.storage_path})
    else:
        raise HTTPException(404, "Audio asset not found")
    
    db.commit()
    return UploadResponse(uploaded=saved)

@router.put("/project/{project_id}/select/{audio_id}")
def select_project_music(
    project_id: uuid.UUID,
    audio_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    
    asset = db.query(MediaAsset).filter(
        MediaAsset.id == audio_id, 
        MediaAsset.user_id == user.id,
        MediaAsset.mime_type.startswith("audio/")
    ).first()
    
    if not asset:
        raise HTTPException(404, "Audio not found")
        
    project.music_id = audio_id
    db.commit()
    return {"message": "Music selected for project"}

@router.put("/project/{project_id}/replace/{old_audio_id}/{new_audio_id}")
def replace_project_audio(
    project_id: uuid.UUID,
    old_audio_id: uuid.UUID,
    new_audio_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    # Check if new audio exists and is owned by user
    new_asset = db.query(MediaAsset).filter(
        MediaAsset.id == new_audio_id, 
        MediaAsset.user_id == user.id,
        MediaAsset.mime_type.startswith("audio/")
    ).first()
    if not new_asset:
        raise HTTPException(404, "New audio asset not found")

    # Remove old link
    db.query(ProjectMediaAsset).filter(
        ProjectMediaAsset.project_id == project_id,
        ProjectMediaAsset.media_asset_id == old_audio_id
    ).delete()
    
    # Add new link
    existing_new_link = db.query(ProjectMediaAsset).filter(
        ProjectMediaAsset.project_id == project_id,
        ProjectMediaAsset.media_asset_id == new_audio_id
    ).first()
    if not existing_new_link:
        db.add(ProjectMediaAsset(project_id=project_id, media_asset_id=new_audio_id))
        
    # If old was music_id, update to new
    if str(project.music_id) == old_audio_id:
        project.music_id = new_audio_id
        
    db.commit()
    return {"message": "Audio replaced successfully"}

@router.get("", response_model=List[MediaResponse])
def list_audio(
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    audio = db.query(MediaAsset).filter(
        MediaAsset.user_id == user.id, 
        MediaAsset.mime_type.startswith("audio/")
    ).all()
    result = []
    for a in audio:
        # Find projects where this audio is set as music_id
        linked_projects = db.query(Project).filter(Project.music_id == a.id).all()
        p_ids = [str(p.id) for p in linked_projects]
        
        result.append(MediaResponse(
            id=str(a.id),
            original_filename=a.original_filename,
            storage_path=a.storage_path,
            mime_type=a.mime_type,
            file_size=a.file_size,
            duration=a.duration,
            thumbnail_path=a.thumbnail_path,
            uploaded_at=a.uploaded_at,
            project_ids=p_ids
        ))
    return result

@router.delete("/project/{project_id}/{audio_id}")
def remove_audio_from_project(
    project_id: uuid.UUID,
    audio_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    # Check project ownership
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    link = db.query(ProjectMediaAsset).filter(
        ProjectMediaAsset.project_id == project_id,
        ProjectMediaAsset.media_asset_id == audio_id
    ).first()
    
    if link:
        db.delete(link)
        # If this was the selected music, nullify it
        if project.music_id == audio_id:
            project.music_id = None
        db.commit()
        return {"message": "Relation broken"}
    raise HTTPException(404, "Relation not found")

@router.delete("/{audio_id}")
def delete_audio_entirely(
    audio_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    asset = db.query(MediaAsset).filter(MediaAsset.id == audio_id, MediaAsset.user_id == user.id).first()
    if not asset:
        raise HTTPException(404, "Audio not found")
        
    # Remove from storage
    full_path = STORAGE_ROOT / asset.storage_path
    if full_path.exists():
        full_path.unlink()
        
    # Remove all links to projects
    db.query(ProjectMediaAsset).filter(ProjectMediaAsset.media_asset_id == audio_id).delete()
    
    # Nullify music_id in projects where it was selected
    db.query(Project).filter(Project.music_id == audio_id).update({Project.music_id: None})
    
    db.delete(asset)
    db.commit()
    return {"message": "Audio deleted entirely"}
