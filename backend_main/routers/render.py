import os
import shutil
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from backend_main.config import SessionLocal, STORAGE_ROOT, logger
from backend_main.models import Project, MediaAsset, User
from backend_main.auth import get_current_user
from backend_ai.orchestrator import ShortifyOrchestrator, AgentState

router = APIRouter(prefix="/projects", tags=["Render"])

# -------------------------------------------------------------------
# Request / Response Schemas
# -------------------------------------------------------------------

class RenderRequest(BaseModel):
    prompt: str
    output_filename: Optional[str] = "final_output.mp4"


class RenderResponse(BaseModel):
    project_id: str
    status: str
    message: str
    final_video_path: Optional[str] = None
    safe_zone_verdict: Optional[str] = None


class ProjectListItem(BaseModel):
    id: str
    title: Optional[str]
    description: Optional[str]
    target_duration: int
    aspect_ratio: str
    style: Optional[str]
    created_at: Optional[datetime]
    render_status: str  # not_started | running | done | error


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


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------

@router.post("/{project_id}/render", response_model=RenderResponse, status_code=202)
def trigger_render(
    project_id: str,
    body: RenderRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal()),
):
    """
    Triggers the full Shortify AI pipeline for a project.

    - Resolves all uploaded media and music files from the project.
    - Runs the LangGraph pipeline as a background task.
    - Returns 202 Accepted immediately.
    - Poll GET /projects/{project_id}/render/status for progress.
    """
    # 1. Validate project ownership
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Forbidden")

    # 2. Resolve video file paths
    media_assets = db.query(MediaAsset).filter(
        MediaAsset.project_id == project_id,
        ~MediaAsset.mime_type.startswith("audio/")
    ).all()

    if not media_assets:
        raise HTTPException(400, "No video media found for this project. Please upload videos first.")

    video_paths = [
        str(STORAGE_ROOT / asset.storage_path)
        for asset in media_assets
        if os.path.exists(STORAGE_ROOT / asset.storage_path)
    ]

    if not video_paths:
        raise HTTPException(400, "Video files not found on disk. Please re-upload.")

    # 3. Resolve music path (optional)
    music_path = None
    if project.music_id:
        music_asset = db.query(MediaAsset).filter(MediaAsset.id == project.music_id).first()
        if music_asset:
            candidate = str(STORAGE_ROOT / music_asset.storage_path)
            if os.path.exists(candidate):
                music_path = candidate

    # 4. Guard against duplicate runs
    existing = render_jobs.get(project_id, {})
    if existing.get("status") == "running":
        raise HTTPException(409, "A render is already in progress for this project.")

    # 5. Kick off the background pipeline
    background_tasks.add_task(
        run_pipeline,
        project_id=project_id,
        prompt=body.prompt,
        video_paths=video_paths,
        music_path=music_path,
        output_filename=body.output_filename,
    )

    logger.info(f"User {user.username} triggered render for project {project_id}")
    return RenderResponse(
        project_id=project_id,
        status="queued",
        message="Render pipeline started. Poll /render/status for updates.",
    )


@router.get("/{project_id}/render/status", response_model=RenderResponse)
def get_render_status(
    project_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(lambda: SessionLocal()),
):
    """
    Returns the current status of the render pipeline for a project.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(404, "Project not found")
    if project.user_id != user.id:
        raise HTTPException(403, "Forbidden")

    job = render_jobs.get(project_id)
    if not job:
        return RenderResponse(
            project_id=project_id,
            status="not_started",
            message="No render has been triggered for this project yet.",
        )

    return RenderResponse(
        project_id=project_id,
        status=job.get("status", "unknown"),
        message=job.get("message", ""),
        final_video_path=job.get("final_video_path"),
        safe_zone_verdict=job.get("safe_zone_verdict"),
    )


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
        job = render_jobs.get(str(proj.id), {})
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
