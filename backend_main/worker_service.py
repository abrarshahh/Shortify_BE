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

class RenderCancelledError(Exception):
    """Raised when a render job is cancelled by the user."""
    pass

active_futures: Dict[str, Any] = {}

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
    caption_style: str = "none",
    add_subtitle: bool = True,
    add_stickers: bool = True,
    add_textoverlay: bool = True,
    audio_ducking: bool = True,
):
    """
    Synchronous worker task running inside the ThreadPool thread.
    Ties the orchestrator lifecycle and updates state progress.
    """
    is_successful = False
    logger.info(f"[Worker] Starting job for project {project_id}")
    update_job(project_id, {
        "status": "running",
        "progress_percentage": 0,
        "current_step": "Initializing pipeline...",
        "message": "Starting Shortify AI engine."
    })

    def orchestrator_progress_callback(percentage: int, step: str):
        job = render_jobs.get(project_id, {})
        if job.get("is_cancelled"):
            raise RenderCancelledError("Render task cancelled by user request.")

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
            "caption_style": caption_style,
            "add_subtitle": add_subtitle,
            "add_stickers": add_stickers,
            "add_textoverlay": add_textoverlay,
            "audio_ducking": audio_ducking,
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
            "has_cached_director": False,
        }

        # Query project user info for user-specific caching
        db = SessionLocal()
        user_folder = "unknown_user"
        try:
            proj = db.query(Project).filter(Project.id == uuid.UUID(project_id)).first()
            if proj and proj.user:
                user_folder = proj.user.username or str(proj.user.id)
        except Exception as e:
            logger.warning(f"[Worker] Failed to query user info for cache path: {e}")
        finally:
            db.close()

        # 1. Download existing project cache files from Supabase if they exist
        from backend_main.supabase_storage import download_from_supabase, upload_to_supabase
        
        cache_files = [
            "clip_scores/clip_scores.json",
            "media_analysis/media_analysis.json",
            "music_analysis/music_analysis.json",
            "director_analysis/director_analysis.json",
            "metadata/metadata.json"
        ]
        
        for f_subpath in cache_files:
            supabase_path = f"cache/{user_folder}/{project_id}/{f_subpath}"
            local_cache_path = os.path.join("cache", user_folder, project_id, f_subpath.replace('/', os.sep))
            if not os.path.exists(local_cache_path):
                download_from_supabase(supabase_path, local_cache_path)

        # Instantiate orchestrator
        orchestrator = ShortifyOrchestrator(
            exports_dir=str(STORAGE_ROOT / "exports" / project_id),
            project_id=project_id,
            user=user_folder
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

        # Save result to DB & Upload cache and logs to Supabase
        db = SessionLocal()
        try:
            pid_uuid = uuid.UUID(project_id)
            proj = db.query(Project).filter(Project.id == pid_uuid).first()
            if proj:
                # Save as relative path to STORAGE_ROOT
                rel_path = str(os.path.relpath(final_video, STORAGE_ROOT)).replace('\\', '/')
                proj.last_output_path = rel_path
                db.commit()
                
                # Upload cache files to Supabase
                for f_subpath in cache_files:
                    local_cache_path = os.path.join("cache", user_folder, project_id, f_subpath.replace('/', os.sep))
                    if os.path.exists(local_cache_path):
                        supabase_path = f"cache/{user_folder}/{project_id}/{f_subpath}"
                        upload_to_supabase(local_cache_path, supabase_path, mime_type="application/json")
                        
                # Upload run logs
                log_files = ["app.log", "error.log", "agents.log"]
                for log_file in log_files:
                    local_log = os.path.join("logs", log_file)
                    if os.path.exists(local_log):
                        upload_to_supabase(local_log, f"logs/{project_id}/{log_file}", mime_type="text/plain")
        finally:
            db.close()

        # Mark render task run as successful (prevents deleting export files during cleanup)
        is_successful = True

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

    except RenderCancelledError as e:
        logger.info(f"[Worker] Job for project {project_id} was cancelled by user.")
        update_job(project_id, {
            "status": "cancelled",
            "progress_percentage": 0,
            "current_step": "Cancelled",
            "message": str(e),
            "skipped_clips": []
        })
        # Clean up export directory if partial renders exist
        export_dir = STORAGE_ROOT / "exports" / project_id
        if export_dir.exists() and export_dir.is_dir():
            import shutil
            try:
                shutil.rmtree(export_dir)
                logger.info(f"[Worker] Cleaned up export directory for cancelled job: {export_dir}")
            except Exception as cleanup_err:
                logger.warning(f"[Worker] Failed to clean up cancelled job export dir: {cleanup_err}")
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
        active_futures.pop(project_id, None)
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
            
        # Clean up local disk files to prevent Render container from running out of disk space
        import shutil
        from backend_main.supabase_storage import is_supabase_configured
        
        if is_supabase_configured():
            try:
                # 1. Delete export directory ONLY if the job failed or was cancelled
                if not is_successful:
                    export_dir = os.path.join(STORAGE_ROOT, "exports", project_id)
                    if os.path.exists(export_dir):
                        shutil.rmtree(export_dir, ignore_errors=True)
                        logger.info(f"[Worker Cleanup] Deleted local export directory for failed/cancelled job: {export_dir}")
                    
                # 2. Delete cache directory
                project_cache_dir = os.path.join("cache", user_folder, project_id)
                if os.path.exists(project_cache_dir):
                    shutil.rmtree(project_cache_dir, ignore_errors=True)
                    logger.info(f"[Worker Cleanup] Deleted local cache directory: {project_cache_dir}")
                    
                # 3. Delete downloaded source videos/audios used in this run
                for path in video_paths:
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                            logger.info(f"[Worker Cleanup] Deleted local source video: {path}")
                        except Exception as unlink_err:
                            logger.warning(f"[Worker Cleanup] Failed to delete source video {path}: {unlink_err}")
                
                if music_path and os.path.exists(music_path):
                    try:
                        os.unlink(music_path)
                        logger.info(f"[Worker Cleanup] Deleted local music file: {music_path}")
                    except Exception as unlink_err:
                        logger.warning(f"[Worker Cleanup] Failed to delete music file {music_path}: {unlink_err}")
            except Exception as cleanup_err:
                logger.error(f"[Worker Cleanup] Error running cleanup: {cleanup_err}")
        else:
            logger.info("[Worker Cleanup] USE_SUPABASE is false. Retaining all local exports, cache, and media files.")

def cancel_job(project_id: str) -> bool:
    """
    Requests cancellation of a queued or running render job.
    Returns True if the job was found and cancellation was requested, False otherwise.
    """
    # 1. Check if the future is in active_futures and can be cancelled before starting
    future = active_futures.get(project_id)
    if future:
        cancelled = future.cancel()
        if cancelled:
            update_job(project_id, {
                "status": "cancelled",
                "progress_percentage": 0,
                "current_step": "Cancelled",
                "message": "Render task was cancelled before starting."
            })
            active_futures.pop(project_id, None)
            return True

    # 2. If it is already running, flag it for cooperative cancellation
    job = render_jobs.get(project_id)
    if job and job.get("status") in ("queued", "running"):
        job["is_cancelled"] = True
        job["status"] = "cancelled"
        job["current_step"] = "Cancelling..."
        job["message"] = "Render task cancellation requested by user."
        return True

    return False

def enqueue_job(
    project_id: str,
    prompt: str,
    video_paths: list,
    music_path: str,
    output_filename: str,
    target_duration: int,
    aspect_ratio: str,
    style: str,
    caption_style: str = "none",
    add_subtitle: bool = True,
    add_stickers: bool = True,
    add_textoverlay: bool = True,
    audio_ducking: bool = True,
):
    """Enqueues a rendering task into the thread pool."""
    update_job(project_id, {
        "status": "queued",
        "progress_percentage": 0,
        "current_step": "Queued in background worker...",
        "message": "Waiting for worker thread."
    })
    
    future = _executor.submit(
        _execute_render_task,
        project_id=project_id,
        prompt=prompt,
        video_paths=video_paths,
        music_path=music_path,
        output_filename=output_filename,
        target_duration=target_duration,
        aspect_ratio=aspect_ratio,
        style=style,
        caption_style=caption_style,
        add_subtitle=add_subtitle,
        add_stickers=add_stickers,
        add_textoverlay=add_textoverlay,
        audio_ducking=audio_ducking,
    )
    active_futures[project_id] = future
