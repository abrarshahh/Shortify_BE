import os
import shutil
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from backend_main.config import SessionLocal, STORAGE_ROOT, logger, get_db
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
    db: Session = Depends(get_db),
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
        logger.warning(f"[Render API] Project {project_id} has no media assets linked in the database.")
        raise HTTPException(400, "No video media found for this project.")

    from backend_main.supabase_storage import download_from_supabase

    # Download missing media assets from Supabase if they are not present on local disk
    for asset in media_assets:
        local_path = STORAGE_ROOT / asset.storage_path
        if not os.path.exists(local_path):
            logger.info(f"[Render API] Missing media asset: {asset.storage_path}. Attempting download from Supabase Storage...")
            download_from_supabase(asset.storage_path, str(local_path))

    video_paths = [
        str(STORAGE_ROOT / asset.storage_path)
        for asset in media_assets
        if os.path.exists(STORAGE_ROOT / asset.storage_path)
    ]

    if not video_paths:
        logger.warning(
            f"[Render API] Project {project_id} has linked media assets in DB, "
            f"but no corresponding physical files found in storage. "
            f"Paths checked: {[str(STORAGE_ROOT / asset.storage_path) for asset in media_assets]}"
        )
        raise HTTPException(400, "Video files not found on disk.")

    music_path = None
    if project.music_id:
        music_asset = db.query(MediaAsset).filter(MediaAsset.id == project.music_id).first()
        if music_asset:
            candidate = str(STORAGE_ROOT / music_asset.storage_path)
            if not os.path.exists(candidate):
                logger.info(f"[Render API] Missing music asset: {music_asset.storage_path}. Attempting download from Supabase Storage...")
                download_from_supabase(music_asset.storage_path, candidate)
            if os.path.exists(candidate):
                music_path = candidate

    # Concurrency check via DB flag
    if project.is_rendering:
        raise HTTPException(409, f"Project '{project.title}' is already rendering. Please wait for it to complete.")

    req_style = body.style.value if (body.style and hasattr(body.style, 'value')) else body.style
    req_aspect_ratio = body.aspect_ratio.value if hasattr(body.aspect_ratio, 'value') else body.aspect_ratio
    req_target_duration = body.target_duration.value if hasattr(body.target_duration, 'value') else body.target_duration
    req_caption_style = body.caption_style

    # Set rendering flag and update project database record with the latest realtime render parameters
    project.is_rendering = True
    project.target_duration = req_target_duration
    project.aspect_ratio = req_aspect_ratio
    project.style = req_style
    project.caption_style = req_caption_style
    db.commit()

    # Build a combined creative brief so all AI stages receive the full
    # project context, not just the one-line per-render prompt.
    project_context_parts = []
    if project.title:
        project_context_parts.append(f"Project Title: {project.title}")
    if project.description:
        project_context_parts.append(f"Project Description: {project.description}")
        
    project_context_parts.append(f"Project Style: {req_style or 'general'}")
    project_context_parts.append(f"Aspect Ratio: {req_aspect_ratio or '9:16'}")
    project_context_parts.append(f"Caption Style: {req_caption_style or 'none'}")
    project_context_parts.append(f"Target Duration: {req_target_duration} seconds")
    if body.prompt:
        project_context_parts.append(f"Render Instruction: {body.prompt}")
    enriched_prompt = "\n".join(project_context_parts)

    worker_service.enqueue_job(
        project_id=str(project_id),
        prompt=enriched_prompt,
        video_paths=video_paths,
        music_path=music_path,
        output_filename=body.output_filename,
        target_duration=req_target_duration,
        aspect_ratio=req_aspect_ratio,
        style=req_style,
        caption_style=req_caption_style or "none",
        add_subtitle=body.add_subtitle,
        add_stickers=body.add_stickers,
        add_textoverlay=body.add_textoverlay,
        audio_ducking=body.audio_ducking,
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
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Forbidden")

    job = worker_service.render_jobs.get(str(project_id))
    if not job:
        # Persisted DB state fallback if worker in-memory state is empty (e.g. server restarted)
        if project.is_rendering:
            return RenderResponse(
                project_id=project_id,
                status="running",
                message=f"Restored render task: {project.render_step or 'Processing'}",
                progress_percentage=project.render_progress or 0,
                current_step=project.render_step or "Rendering..."
            )
        elif project.last_output_path:
            return RenderResponse(
                project_id=project_id,
                status="done",
                message="Render completed.",
                progress_percentage=100,
                current_step="Complete!",
                final_video_path=str(STORAGE_ROOT / project.last_output_path)
            )
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
        current_step=job.get("current_step", "Initializing..."),
        skipped_clips=job.get("skipped_clips", [])
    )

@router.delete("/{project_id}/render", status_code=200)
def cancel_render(
    project_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Cancels a running or queued render job for the project.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Forbidden")

    # Request cancellation of the job in worker_service
    worker_service.cancel_job(str(project_id))

    # Reset rendering flag in DB immediately to prevent locking
    project.is_rendering = False
    db.commit()

    return {
        "project_id": str(project_id),
        "status": "cancelled",
        "message": "Render job cancellation requested successfully."
    }


@router.get("/outputs", response_model=List[OutputVideoResponse])
def list_output_videos(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    projects = db.query(Project).filter(Project.user_id == user.id, Project.last_output_path != None).all()
    outputs = []
    for proj in projects:
        outputs.append(OutputVideoResponse(
            project_id=str(proj.id),
            output_video=proj.last_output_path
        ))
    return outputs

@router.get("/{project_id}/download")
def download_project_video(
    project_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Downloads the final rendered video (.mp4) for the authenticated user.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Forbidden")
    if not project.last_output_path:
        raise HTTPException(400, "Project output video has not been rendered yet.")

    video_path = STORAGE_ROOT / project.last_output_path
    if not video_path.exists():
        raise HTTPException(404, "Video file not found on disk.")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=os.path.basename(project.last_output_path)
    )

@router.get("/{project_id}/thumbnail")
def get_project_thumbnail(
    project_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieves the generated cover thumbnail image (.jpg) for the project.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Forbidden")

    thumbnail_path = STORAGE_ROOT / "exports" / str(project_id) / "thumbnail.jpg"
    if not thumbnail_path.exists():
        raise HTTPException(404, "Thumbnail image not found for this project.")

    return FileResponse(
        path=str(thumbnail_path),
        media_type="image/jpeg",
        filename="thumbnail.jpg"
    )
