import os
import uuid
import logging
from typing import Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor

from backend_main.config import SessionLocal, STORAGE_ROOT, logger
from backend_main.models import Project
from backend_ai.orchestrator import ShortifyOrchestrator, AgentState
from backend_ai.schemas.edl import EDLGenerationError

# Thread-safe job state tracker (in-memory, acts as database fallback)
render_jobs: Dict[str, Dict[str, Any]] = {}

# Single-worker thread pool to ensure renders run sequentially
# and don't overwhelm CPU/Memory resources with MoviePy/FFmpeg
_executor = ThreadPoolExecutor(max_workers=1)

def update_job(project_id: str, updates: Dict[str, Any]):
    """Helper to update job state fields in a thread-safe manner and persist to DB."""
    if project_id not in render_jobs:
        render_jobs[project_id] = {}
    render_jobs[project_id].update(updates)

    db = SessionLocal()
    try:
        pid_uuid = uuid.UUID(project_id)
        proj = db.query(Project).filter(Project.id == pid_uuid).first()
        if proj:
            status = updates.get("status")
            if status == "running":
                proj.is_rendering = True
            elif status in ("done", "error"):
                proj.is_rendering = False

            if "progress_percentage" in updates:
                proj.render_progress = updates["progress_percentage"]
            if "current_step" in updates:
                proj.render_step = updates["current_step"]
            db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist rendering progress to database: {e}")
    finally:
        db.close()

def _execute_render_task(
    project_id: str,
    prompt: str,
    video_paths: list,
    music_path: str,
    output_filename: str,
    target_duration: int,
    aspect_ratio: str,
    style: str,
):
    """
    Synchronous worker task running inside the ThreadPool thread.
    Ties the orchestrator lifecycle and updates state progress.
    """
    logger.info(f"[Worker] Starting job for project {project_id}")
    update_job(project_id, {
        "status": "running",
        "progress_percentage": 0,
        "current_step": "Initializing pipeline...",
        "message": "Starting Shortify AI engine."
    })

    def orchestrator_progress_callback(percentage: int, step: str):
        logger.info(f"[{project_id}] Progress: {percentage}% | {step}")
        messages_map = {
            "Initializing pipeline...": "Launching Shortify AI engine orchestrator and loading resources.",
            "Analyzing audio beats...": "RhythmEngineer is analyzing the soundtrack beats and energy levels to align transitions.",
            "AI Media Analysis...": "MediaAnalyst is identifying visual highlights, scenes, and media quality.",
            "Creating storyboard and timeline...": "CreativeDirector is compiling the narrative Edit Decision List story storyboard.",
            "Rendering video...": "VideoEditor is center-cropping, syncing cuts to beats, and rendering video clips.",
            "Color grading...": "ColorGradingAgent is applying rich cinematic look-up tables (LUTs) to balance color tones.",
            "Checking safe zone compliance...": "ValidatorAgent is scanning overlays to ensure text remains perfectly within UI safe zones.",
            "Transcribing speech...": "SubtitleAgent is transcribing spoken word audio with high-precision Whisper.",
            "Burning subtitles...": "SubtitleAgent is drawing dual-layered drop-shadow captions onto the video canvas.",
            "Creating click-through cover...": "ThumbnailAgent is rendering an eye-catching, high-impact thumbnail for social media."
        }
        friendly_message = messages_map.get(step, f"Processing stage: {step}")
        update_job(project_id, {
            "progress_percentage": percentage,
            "current_step": step,
            "message": friendly_message
        })

    try:
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
            "color_graded_path": "",
            "safe_zone_report": {},
            "transcription": {},
            "final_video_path": "",
            "retry_count": 0,
            "max_edl_retries": 0,
            "pre_flight_report": {},
            "progress_callback": orchestrator_progress_callback,
            "clip_scores": {},
        }

        # Instantiate orchestrator
        orchestrator = ShortifyOrchestrator(
            exports_dir=str(STORAGE_ROOT / "exports" / project_id)
        )
        final_state = orchestrator.run(initial_state)

        final_video = final_state.get("final_video_path", "")
        verdict = final_state.get("safe_zone_report", {}).get("verdict", "N/A")
        skipped_clips = final_state.get("skipped_clips", [])

        # Clean up intermediate render files to save disk space
        for path_key in ["rendered_video_path", "color_graded_path"]:
            p = final_state.get(path_key)
            if p and os.path.exists(p) and p != final_video:
                try:
                    os.unlink(p)
                    logger.info(f"[Worker] Cleaned up intermediate file: {p}")
                except Exception as e:
                    logger.warning(f"[Worker] Failed to delete intermediate file {p}: {e}")

        # Save result to DB
        db = SessionLocal()
        try:
            pid_uuid = uuid.UUID(project_id)
            proj = db.query(Project).filter(Project.id == pid_uuid).first()
            if proj:
                # Save as relative path to STORAGE_ROOT
                rel_path = str(os.path.relpath(final_video, STORAGE_ROOT))
                proj.last_output_path = rel_path
                db.commit()
        finally:
            db.close()

        update_job(project_id, {
            "status": "done",
            "progress_percentage": 100,
            "current_step": "Complete!",
            "final_video_path": final_video,
            "safe_zone_verdict": verdict,
            "skipped_clips": skipped_clips,
            "message": "Render completed successfully."
        })
        logger.info(f"[Worker] Job completed for project {project_id}")

    except EDLGenerationError as e:
        logger.error(f"[Worker] EDL generation failed for project {project_id}: {e}")
        update_job(project_id, {
            "status": "error",
            "progress_percentage": 0,
            "current_step": "EDL validation failed",
            "message": str(e),
            "skipped_clips": []
        })
    except Exception as e:
        logger.error(f"[Worker] Job failed for project {project_id}: {e}")
        update_job(project_id, {
            "status": "error",
            "progress_percentage": 0,
            "current_step": "Failed",
            "message": str(e),
            "skipped_clips": []
        })
    finally:
        # Always reset rendering flag in DB
        db = SessionLocal()
        try:
            pid_uuid = uuid.UUID(project_id)
            proj = db.query(Project).filter(Project.id == pid_uuid).first()
            if proj:
                proj.is_rendering = False
                db.commit()
        finally:
            db.close()

def enqueue_job(
    project_id: str,
    prompt: str,
    video_paths: list,
    music_path: str,
    output_filename: str,
    target_duration: int,
    aspect_ratio: str,
    style: str,
):
    """Enqueues a rendering task into the thread pool."""
    update_job(project_id, {
        "status": "queued",
        "progress_percentage": 0,
        "current_step": "Queued in background worker...",
        "message": "Waiting for worker thread."
    })
    
    _executor.submit(
        _execute_render_task,
        project_id=project_id,
        prompt=prompt,
        video_paths=video_paths,
        music_path=music_path,
        output_filename=output_filename,
        target_duration=target_duration,
        aspect_ratio=aspect_ratio,
        style=style,
    )
