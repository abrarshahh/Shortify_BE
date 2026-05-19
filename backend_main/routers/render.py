import os
import shutil
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from backend_main.config import SessionLocal, STORAGE_ROOT, logger
from backend_main.models import Project, MediaAsset, User, ProjectMediaAsset
from backend_main.auth import get_current_user
from backend_main.schemas import OutputVideoResponse, RenderResponse, RenderRequest
from backend_ai.orchestrator import ShortifyOrchestrator, AgentState
from backend_main import worker_service

router = APIRouter(prefix="/projects", tags=["Render"])


@router.post("/{project_id}/render", response_model=RenderResponse, status_code=202)
def trigger_render(
    project_id: uuid.UUID,
    body: RenderRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal()),
):
    """
    Triggers the full Shortify AI pipeline for a project.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Forbidden")

    media_assets = (
        db.query(MediaAsset)
        .join(ProjectMediaAsset, ProjectMediaAsset.media_asset_id == MediaAsset.id)
        .filter(
            ProjectMediaAsset.project_id == project_id,
            ~MediaAsset.mime_type.startswith("audio/")
        ).all()
    )

    if not media_assets:
        raise HTTPException(400, "No video media found for this project.")

    video_paths = [
        str(STORAGE_ROOT / asset.storage_path)
        for asset in media_assets
        if os.path.exists(STORAGE_ROOT / asset.storage_path)
    ]

    if not video_paths:
        raise HTTPException(400, "Video files not found on disk.")

    music_path = None
    if project.music_id:
        music_asset = db.query(MediaAsset).filter(MediaAsset.id == project.music_id).first()
        if music_asset:
            candidate = str(STORAGE_ROOT / music_asset.storage_path)
            if os.path.exists(candidate):
                music_path = candidate

    # Concurrency check via DB flag
    if project.is_rendering:
        raise HTTPException(409, f"Project '{project.title}' is already rendering. Please wait for it to complete.")

    # Set rendering flag
    project.is_rendering = True
    db.commit()

    worker_service.enqueue_job(
        project_id=str(project_id),
        prompt=body.prompt,
        video_paths=video_paths,
        music_path=music_path,
        output_filename=body.output_filename,
        target_duration=project.target_duration,
        aspect_ratio=project.aspect_ratio,
        style=project.style,
    )

    return RenderResponse(
        project_id=project_id,
        status="queued",
        message="Render pipeline queued in background worker.",
        progress_percentage=0,
        current_step="Queued in background worker..."
    )

@router.get("/{project_id}/render/status", response_model=RenderResponse)
def get_render_status(
    project_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal()),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Forbidden")

    job = worker_service.render_jobs.get(str(project_id))
    if not job:
        return RenderResponse(
            project_id=project_id,
            status="not_started",
            message="No render triggered.",
            progress_percentage=0,
            current_step="Not started"
        )

    return RenderResponse(
        project_id=project_id,
        status=job.get("status", "unknown"),
        message=job.get("message", ""),
        final_video_path=job.get("final_video_path"),
        safe_zone_verdict=job.get("safe_zone_verdict"),
        progress_percentage=job.get("progress_percentage", 0),
        current_step=job.get("current_step", "Initializing...")
    )

@router.get("/outputs", response_model=List[OutputVideoResponse])
def list_output_videos(
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal())
):
    projects = db.query(Project).filter(Project.user_id == user.id, Project.last_output_path != None).all()
    outputs = []
    for proj in projects:
        outputs.append(OutputVideoResponse(
            project_id=str(proj.id),
            output_video=proj.last_output_path
        ))
    return outputs
