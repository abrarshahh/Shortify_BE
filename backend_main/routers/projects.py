import os, shutil, uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend_main.config import SessionLocal, logger, STORAGE_ROOT
from backend_main.models import Project, User, MediaAsset, ProjectMediaAsset
from backend_main.auth import get_current_user
from backend_main.schemas import ProjectCreate, ProjectResponse, ProjectDetailResponse, MediaResponse, ProjectListItem
from backend_main import worker_service
from typing import List

router = APIRouter(prefix="/projects", tags=["Projects"])

@router.get("", response_model=List[ProjectListItem])
def list_all_projects(
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal()),
):
    """
    Returns all projects belonging to the authenticated user,
    along with their current render status.
    """
    projects = (
        db.query(Project)
        .filter(Project.user_id == user.id)
        .order_by(Project.created_at.desc())
        .all()
    )

    result = []
    for proj in projects:
        job = worker_service.render_jobs.get(str(proj.id), {})
        result.append(
            ProjectListItem(
                id=str(proj.id),
                title=proj.title,
                description=proj.description,
                target_duration=proj.target_duration,
                aspect_ratio=proj.aspect_ratio,
                style=proj.style,
                created_at=proj.created_at,
                render_status=job.get("status", "not_started"),
            )
        )

    return result

@router.post("", response_model=ProjectResponse, status_code=201)
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

@router.get("/{project_id}", response_model=ProjectDetailResponse)
def get_project_details(
    project_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    media_list = []
    # project.media_assets contains all linked assets (video, photo, audio)
    for asset in project.media_assets:
        # We only show non-audio media in the 'media' list to keep it clean
        if not asset.mime_type.startswith("audio/"):
            media_list.append(MediaResponse(
                id=str(asset.id),
                original_filename=asset.original_filename,
                storage_path=asset.storage_path,
                mime_type=asset.mime_type,
                file_size=asset.file_size,
                duration=asset.duration,
                width=asset.width,
                height=asset.height,
                thumbnail_path=asset.thumbnail_path,
                extra_metadata=asset.extra_metadata,
                uploaded_at=asset.uploaded_at
            ))
            
    return ProjectDetailResponse(
        id=str(project.id),
        title=project.title,
        description=project.description,
        target_duration=project.target_duration,
        aspect_ratio=project.aspect_ratio,
        style=project.style,
        music_id=str(project.music_id) if project.music_id else None,
        created_at=project.created_at,
        media=media_list
    )

@router.delete("/{project_id}")
def delete_project_soft(
    project_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    """Deletes project and its relations, but keeps media files and output."""
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    # Remove relations
    db.query(ProjectMediaAsset).filter(ProjectMediaAsset.project_id == project_id).delete()
    
    db.delete(project)
    db.commit()
    return {"message": "Project deleted (files preserved)"}

@router.delete("/{project_id}/hard")
def delete_project_hard(
    project_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    """Deletes project, its relations, and all associated media/audio files."""
    project = db.query(Project).filter(Project.id == project_id, Project.user_id == user.id).first()
    if not project:
        raise HTTPException(404, "Project not found")
        
    # 1. Get all linked assets before removing links
    assets = list(project.media_assets)
    
    # 2. Remove all links for this project
    db.query(ProjectMediaAsset).filter(ProjectMediaAsset.project_id == project_id).delete()
    
    # 3. For each asset, check if it's orphaned (not used by any other project)
    for asset in assets:
        # Check if any links remain for this asset
        remaining_links = db.query(ProjectMediaAsset).filter(ProjectMediaAsset.media_asset_id == asset.id).count()
        
        # Also check if it's used as music_id in ANY project
        is_used_as_music = db.query(Project).filter(Project.music_id == asset.id).count()
        
        if remaining_links == 0 and is_used_as_music == 0:
            # Safe to delete file and record entirely
            full_path = STORAGE_ROOT / asset.storage_path
            if full_path.exists():
                full_path.unlink()
            # If there's a thumbnail, delete it too
            if asset.thumbnail_path:
                thumb_path = STORAGE_ROOT / asset.thumbnail_path
                if thumb_path.exists():
                    thumb_path.unlink()
            db.delete(asset)
        
    # 4. Remove export directory (output video)
    export_dir = STORAGE_ROOT / "exports" / project_id
    if export_dir.exists() and export_dir.is_dir():
        shutil.rmtree(export_dir)
        
    # 5. Delete the project itself
    db.delete(project)
    db.commit()
    return {"message": "Project and its exclusive assets deleted successfully"}
