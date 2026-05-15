import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from backend_main.config import SessionLocal, logger, STORAGE_ROOT
from backend_main.models import Project, MediaAsset, User, ProjectMediaAsset
from backend_main.auth import get_current_user
from backend_main.schemas import MediaResponse, UploadResponse, MediaLinkRequest
from backend_main.config import save_upload_file, validate_file

router = APIRouter(prefix="/media", tags=["Media"])

@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_media_to_library(
    files: List[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    saved = []
    for upload in files:
        validate_file(upload)
        # Check if it's NOT audio
        if upload.content_type.startswith("audio/"):
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
def upload_media_to_project(
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
        if upload.content_type.startswith("audio/"):
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
        
        # Link to project
        db.add(ProjectMediaAsset(project_id=project_id, media_asset_id=asset.id))
        saved.append({"id": str(asset.id), "path": asset.storage_path})
    
    db.commit()
    return UploadResponse(uploaded=saved)

@router.post("/project/{project_id}/link", response_model=UploadResponse, status_code=201)
def link_media_to_project(
    project_id: uuid.UUID,
    request_data: MediaLinkRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    saved = []
    if request_data.media_ids:
        for m_id in request_data.media_ids:
            asset = db.query(MediaAsset).filter(MediaAsset.id == m_id, MediaAsset.user_id == user.id).first()
            if asset and not asset.mime_type.startswith("audio/"):
                existing_link = db.query(ProjectMediaAsset).filter(
                    ProjectMediaAsset.project_id == project_id, 
                    ProjectMediaAsset.media_asset_id == m_id
                ).first()
                if not existing_link:
                    db.add(ProjectMediaAsset(project_id=project_id, media_asset_id=m_id))
                saved.append({"id": str(asset.id), "path": asset.storage_path})
    
    db.commit()
    return UploadResponse(uploaded=saved)

@router.get("", response_model=List[MediaResponse])
def list_media(
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    media = db.query(MediaAsset).filter(
        MediaAsset.user_id == user.id, 
        ~MediaAsset.mime_type.startswith("audio/")
    ).all()
    return [
        MediaResponse(
            id=str(m.id),
            original_filename=m.original_filename,
            storage_path=m.storage_path,
            mime_type=m.mime_type,
            file_size=m.file_size,
            duration=m.duration,
            thumbnail_path=m.thumbnail_path,
            uploaded_at=m.uploaded_at,
            project_ids=[str(p.id) for p in m.projects]
        ) for m in media
    ]

@router.delete("/project/{project_id}/{media_id}")
def remove_media_from_project(
    project_id: uuid.UUID,
    media_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    # Check project ownership
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    link = db.query(ProjectMediaAsset).filter(
        ProjectMediaAsset.project_id == project_id,
        ProjectMediaAsset.media_asset_id == media_id
    ).first()
    
    if link:
        db.delete(link)
        db.commit()
        return {"message": "Media removed from project"}
    raise HTTPException(404, "Relation not found")

@router.put("/project/{project_id}/replace/{old_media_id}/{new_media_id}")
def replace_project_media(
    project_id: uuid.UUID,
    old_media_id: uuid.UUID,
    new_media_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    # Check if new media exists and is owned by user
    new_asset = db.query(MediaAsset).filter(
        MediaAsset.id == new_media_id, 
        MediaAsset.user_id == user.id,
        ~MediaAsset.mime_type.startswith("audio/")
    ).first()
    if not new_asset:
        raise HTTPException(404, "New media asset not found")

    # Remove old link
    db.query(ProjectMediaAsset).filter(
        ProjectMediaAsset.project_id == project_id,
        ProjectMediaAsset.media_asset_id == old_media_id
    ).delete()
    
    # Add new link
    existing_new_link = db.query(ProjectMediaAsset).filter(
        ProjectMediaAsset.project_id == project_id,
        ProjectMediaAsset.media_asset_id == new_media_id
    ).first()
    if not existing_new_link:
        db.add(ProjectMediaAsset(project_id=project_id, media_asset_id=new_media_id))
        
    db.commit()
    return {"message": "Media replaced successfully"}

@router.delete("/{media_id}")
def delete_media_entirely(
    media_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    asset = db.query(MediaAsset).filter(MediaAsset.id == media_id, MediaAsset.user_id == user.id).first()
    if not asset:
        raise HTTPException(404, "Media not found")
        
    # Remove from storage
    full_path = STORAGE_ROOT / asset.storage_path
    if full_path.exists():
        full_path.unlink()
        
    # Remove all links to projects
    db.query(ProjectMediaAsset).filter(ProjectMediaAsset.media_asset_id == media_id).delete()
    
    db.delete(asset)
    db.commit()
    return {"message": "Media deleted entirely"}
