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

router = APIRouter(prefix="/projects", tags=["Render"])

# -------------------------------------------------------------------
# In-memory job tracker (replace with Redis / DB in production)
# -------------------------------------------------------------------
render_jobs: dict = {}

def run_pipeline(
    project_id: str,
    prompt: str,
    video_paths: list,
    music_path: Optional[str],
    output_filename: str,
    target_duration: int,
    aspect_ratio: str,
    style: Optional[str]
):
    """
    Background task: runs the full Shortify LangGraph pipeline.
    """
    try:
        logger.info(f"[{project_id}] Starting Shortify pipeline...")
        render_jobs[project_id] = {"status": "running", "message": "Pipeline started"}

        initial_state: AgentState = {
            "video_paths": video_paths,
            "music_path": music_path,
            "project_title": prompt,
            "output_filename": output_filename,
            "target_duration": target_duration,
            "aspect_ratio": aspect_ratio,
            "style": style or "general",
            "rhythm_data": {},
            "visual_data": [],
            "edl": {},
            "edl_feedback": "",
            "rendered_video_path": "",
            "safe_zone_report": {},
            "transcription": {},
            "final_video_path": "",
            "retry_count": 0,
        }

        orchestrator = ShortifyOrchestrator(
            exports_dir=str(STORAGE_ROOT / "exports" / project_id)
        )
        final_state = orchestrator.run(initial_state)

        final_video = final_state.get("final_video_path", "")
        verdict = final_state.get("safe_zone_report", {}).get("verdict", "N/A")

        # Update project in DB with last output path
        db = SessionLocal()
        try:
            proj = db.query(Project).filter(Project.id == project_id).first()
            if proj:
                # Save as relative path to STORAGE_ROOT
                rel_path = str(os.path.relpath(final_video, STORAGE_ROOT))
                proj.last_output_path = rel_path
                db.commit()
        finally:
            db.close()

        render_jobs[project_id] = {
            "status": "done",
            "message": "Render complete.",
            "final_video_path": final_video,
            "safe_zone_verdict": verdict,
        }
        logger.info(f"[{project_id}] Pipeline complete. Final: {final_video}")

    except Exception as e:
        logger.error(f"[{project_id}] Pipeline failed: {e}")
        render_jobs[project_id] = {
            "status": "error",
            "message": str(e),
        }

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

    existing = render_jobs.get(str(project_id), {})
    if existing.get("status") == "running":
        raise HTTPException(409, "A render is already in progress.")

    background_tasks.add_task(
        run_pipeline,
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
        message="Render pipeline started.",
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

    job = render_jobs.get(str(project_id))
    if not job:
        return RenderResponse(
            project_id=project_id,
            status="not_started",
            message="No render triggered.",
        )

    return RenderResponse(
        project_id=project_id,
        status=job.get("status", "unknown"),
        message=job.get("message", ""),
        final_video_path=job.get("final_video_path"),
        safe_zone_verdict=job.get("safe_zone_verdict"),
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
