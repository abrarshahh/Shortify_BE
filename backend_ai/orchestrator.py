import os
import logging
from typing import Dict, Any, List, Optional, Callable
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

# Import our agents
from backend_ai.services.rhythm_service import RhythmEngineer
from backend_ai.agents.media_agent import MediaAnalyst
from backend_ai.agents.director_agent import CreativeDirector
from backend_ai.services.editor_service import VideoEditor
from backend_ai.agents.subtitle_agent import SubtitleAgent
from backend_ai.services.color_service import ColorGradingAgent
from backend_ai.agents.project_analyst_agent import ProjectAnalystAgent
from backend_ai.agents.thumbnail_agent import ThumbnailAgent
from backend_ai.agents.clip_scoring_agent import ClipScoringAgent
from backend_ai.services.edl_validation_service import validate_edl
from backend_ai.schemas.edl import EDLGenerationError, EDLValidationError
from backend_ai.core.config_loader import AGENTS_CONFIG
from backend_ai.core.logging_config import setup_logging, start_new_agent_run

# Initialize logging config
setup_logging()
logger = logging.getLogger("orchestrator")

# -------------------------------------------------------------------
# 1. Define the State
# -------------------------------------------------------------------
class AgentState(TypedDict):
    video_paths: List[str]
    music_path: Optional[str]
    project_title: str
    output_filename: str
    target_duration: int
    aspect_ratio: str
    style: str
    
    # Internal data passed between nodes
    rhythm_data: Dict[str, Any]
    visual_data: List[Dict[str, Any]]
    edl: Dict[str, Any]
    edl_feedback: str
    
    rendered_video_path: str
    color_graded_path: str
    safe_zone_report: Dict[str, Any]
    transcription: Dict[str, Any]
    final_video_path: str
    retry_count: int
    max_edl_retries: int
    pre_flight_report: Dict[str, Any]
    progress_callback: Optional[Callable[[int, str], None]]
    skipped_clips: Optional[List[str]]
    clip_scores: Optional[Dict[str, Any]]
    dynamic_style: Optional[Dict[str, Any]]


class ShortifyOrchestrator:
    """
    Phase 7: Centralized LangGraph orchestrator that ties all agents together.
    """

    def __init__(self, exports_dir: str = "data/exports"):
        self.exports_dir = exports_dir
        os.makedirs(exports_dir, exist_ok=True)
        
        # Instantiate the agents
        self.rhythm_agent = RhythmEngineer()
        self.media_agent = MediaAnalyst()
        self.director_agent = CreativeDirector()
        
        self.color_grading_agent = ColorGradingAgent()
        self.analyst_agent = ProjectAnalystAgent()
        self.thumbnail_agent = ThumbnailAgent()
        self.clip_scoring_agent = ClipScoringAgent()
        
        # Subtitle config
        sub_config = AGENTS_CONFIG.get("subtitle_agent", {})
        self.subtitle_agent = SubtitleAgent(
            model_size=sub_config.get("model_size", "base"),
            device=sub_config.get("device", "cpu"),
            caption_style=sub_config.get("caption_style", "hormozi")
        )
        
        # Build and compile the graph
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_rhythm", self.node_analyze_rhythm)
        workflow.add_node("pre_flight_check", self.node_pre_flight_check)
        workflow.add_node("analyze_media", self.node_analyze_media)
        workflow.add_node("score_clips", self.node_score_clips)
        workflow.add_node("generate_edl", self.node_generate_edl)
        workflow.add_node("render_video", self.node_render_video)
        workflow.add_node("color_grade", self.node_color_grade)
        workflow.add_node("review_safety", self.node_review_safety)
        workflow.add_node("burn_subtitles", self.node_burn_subtitles)

        # Set edges
        workflow.set_entry_point("pre_flight_check")
        workflow.add_edge("pre_flight_check", "score_clips")
        workflow.add_edge("score_clips", "analyze_rhythm")
        workflow.add_edge("analyze_rhythm", "analyze_media")
        workflow.add_edge("analyze_media", "generate_edl")
        workflow.add_edge("generate_edl", "render_video")
        workflow.add_edge("render_video", "color_grade")
        workflow.add_edge("color_grade", "review_safety")
        
        # Conditional edge based on safety check
        workflow.add_conditional_edges(
            "review_safety",
            self.route_after_safety,
            {
                "pass": "burn_subtitles",
                "fail": "generate_edl"
            }
        )
        
        workflow.add_edge("burn_subtitles", END)

        return workflow.compile()

    # -------------------------------------------------------------------
    # 2. Node Functions
    # -------------------------------------------------------------------

    def node_pre_flight_check(self, state: AgentState) -> Dict:
        logger.info("NODE: pre_flight_check")
        callback = state.get("progress_callback")
        if callback:
            callback(10, "Evaluating video clip quality...")
            
        report = self.analyst_agent.analyze_inputs(state["video_paths"])
        
        # If all files failed validation, abort pipeline
        if report["valid_files"] == 0:
            raise ValueError(
                "Pipeline aborted: None of the input files passed pre-flight validation. "
                f"Rejected reasons: {[r['reason'] for r in report['rejected_files']]}"
            )
            
        return {
            "video_paths": report["recommended_order"],
            "pre_flight_report": report
        }

    def node_analyze_rhythm(self, state: AgentState) -> Dict:
        logger.info("NODE: analyze_rhythm")
        callback = state.get("progress_callback")
        if callback:
            callback(25, "Analyzing audio beats...")
            
        music_path = state.get("music_path")
        
        if not music_path or not os.path.exists(music_path):
            logger.info("No music path provided or found. Skipping rhythm analysis.")
            return {"rhythm_data": {}}
        
        logger.info(f"Analyzing beats for: {music_path}")
        rhythm_data = self.rhythm_agent.analyze_music(music_path, clip_scores=state.get("clip_scores"))
        return {"rhythm_data": rhythm_data}

    def node_analyze_media(self, state: AgentState) -> Dict:
        logger.info("NODE: analyze_media")
        callback = state.get("progress_callback")
        if callback:
            callback(50, "AI Media Analysis...")
            
        visual_data = []
        for path in state["video_paths"]:
            if os.path.exists(path):
                logger.info(f"Analyzing visual context for: {path}")
                analysis = self.media_agent.analyze_video(path)
                visual_data.append(analysis)
                # Small delay to avoid bursting the Gemini API rate limit
                import time
                time.sleep(1.5)
            else:
                logger.warning(f"Video not found at {path}")
                
        return {"visual_data": visual_data}

    def node_score_clips(self, state: AgentState) -> Dict:
        logger.info("NODE: score_clips")
        callback = state.get("progress_callback")
        if callback:
            callback(58, "Scoring and ranking video segments locally...")
            
        clip_scores = self.clip_scoring_agent.score_all_clips(
            video_paths=state["video_paths"],
            style=state.get("style", "cinematic")
        )
        
        # For backward compatibility, also scoring visual_data if it exists (e.g. during retries/custom flows)
        visual_data = state.get("visual_data", [])
        if visual_data:
            visual_data = self.clip_scoring_agent.score_visual_data(visual_data)
            
        return {
            "clip_scores": clip_scores,
            "visual_data": visual_data
        }

    def node_generate_edl(self, state: AgentState) -> Dict:
        logger.info("NODE: generate_edl")
        callback = state.get("progress_callback")
        if callback:
            callback(65, "Creating storyboard and timeline...")
            
        feedback = state.get("edl_feedback")
        max_edl_retries = state.get("max_edl_retries", 0)
        clips_dir = os.path.dirname(state["video_paths"][0]) if state.get("video_paths") else ""
        if not clips_dir:
            raise ValueError("No video paths provided for EDL validation.")

        prompt = state["project_title"]

        while True:
            logger.info(f"Generating EDL. Prompt: {prompt}")
            edl = self.director_agent.generate_edl(
                user_prompt=prompt,
                media_analyses=state["visual_data"],
                audio_analysis=state.get("rhythm_data", {}),
                target_duration=state["target_duration"],
                aspect_ratio=state["aspect_ratio"],
                style=state["style"],
                feedback=feedback,
                pre_flight_report=state.get("pre_flight_report"),
                clip_scores=state.get("clip_scores")
            )

            try:
                validated_edl = validate_edl(edl, clips_dir, target_duration=float(state["target_duration"]))
                edl_dict = validated_edl.model_dump(mode="json")
                
                # Apply hook corrections (Task 30)
                corrected_edl = self._apply_hook_corrections(
                    edl=edl_dict,
                    clip_scores=state.get("clip_scores"),
                    clips_dir=clips_dir
                )
                
                # Re-validate corrected EDL to ensure continuity & matching durations
                validated_corrected_edl = validate_edl(corrected_edl, clips_dir, target_duration=float(state["target_duration"]))
                
                return {
                    "edl": validated_corrected_edl.model_dump(mode="json"),
                    "edl_feedback": "",
                    "max_edl_retries": max_edl_retries,
                }
            except EDLValidationError as exc:
                max_edl_retries += 1
                state["max_edl_retries"] = max_edl_retries
                feedback = exc.to_feedback()
                state["edl_feedback"] = feedback

                logger.warning(f"EDL validation failed (attempt {max_edl_retries}/3): {feedback}")
                if max_edl_retries >= 3:
                    raise EDLGenerationError(
                        retry_count=max_edl_retries,
                        last_error=feedback,
                        issues=exc.issues,
                    )

                continue

    def node_render_video(self, state: AgentState) -> Dict:
        logger.info("NODE: render_video")
        callback = state.get("progress_callback")
        if callback:
            callback(80, "Rendering video...")
            
        # VideoEditor is instantiated per-render because it needs clips_dir.
        # We assume all clips are in the same directory as the first input path.
        if not state["video_paths"]:
            raise ValueError("No video paths provided.")
            
        clips_dir = os.path.dirname(state["video_paths"][0])
        editor = VideoEditor(clips_dir=clips_dir, output_dir=self.exports_dir)
        
        output_filename = f"render_{state['output_filename']}"
        
        # Dynamically generate typography style
        storyline = state.get("edl", {}).get("storyline", "")
        video_style = state.get("style", "cinematic")
        
        dynamic_style = self.subtitle_agent.generate_aesthetic_style(
            prompt=state["project_title"],
            storyline=storyline,
            video_style=video_style
        )
        
        # Download dynamic stickers and visual effects overlays
        edl = state.get("edl", {})
        timeline = edl.get("timeline", [])
        from backend_ai.utils.effect_downloader import download_giphy_sticker, download_pixabay_effect
        
        for item in timeline:
            details = item.get("details", {})
            sticker_query = details.get("sticker_query", "")
            effect_query = details.get("effect_query", "")
            effect_type = details.get("effect_type", "none")
            
            # Download Giphy sticker if query is set
            sticker_path = None
            if sticker_query:
                try:
                    sticker_path = download_giphy_sticker(sticker_query)
                except Exception as e:
                    logger.warning(f"Failed to download sticker for query '{sticker_query}': {e}")
            if sticker_path:
                item["sticker_path"] = sticker_path
                item["sticker_position"] = details.get("sticker_position", "bottom-center")
                
            # Download Pixabay overlay loop if query is set
            effect_path = None
            if effect_type != "none" and effect_query:
                try:
                    effect_path = download_pixabay_effect(effect_query)
                except Exception as e:
                    logger.warning(f"Failed to download effect for query '{effect_query}': {e}")
            if effect_path:
                item["effect_path"] = effect_path

        logger.info(f"Rendering EDL to {output_filename}...")
        rendered_path = editor.render(
            edl=state["edl"],
            music_path=state.get("music_path"),
            output_filename=output_filename,
            aspect_ratio=state.get("aspect_ratio", "9:16"),
            rhythm_data=state.get("rhythm_data", {}),
            clip_scores=state.get("clip_scores"),
            dynamic_style=dynamic_style
        )
        
        return {
            "rendered_video_path": rendered_path,
            "skipped_clips": getattr(editor, "skipped_clips", []),
            "dynamic_style": dynamic_style
        }

    def node_color_grade(self, state: AgentState) -> Dict:
        logger.info("NODE: color_grade")
        callback = state.get("progress_callback")
        if callback:
            callback(90, "Applying style color grade...")
            
        try:
            graded_path = self.color_grading_agent.apply_grade(
                video_path=state["rendered_video_path"],
                style=state["style"],
                output_dir=self.exports_dir,
            )
            return {"color_graded_path": graded_path}
        except Exception as e:
            # Color grading is non-critical — if it fails, continue with
            # the ungraded video rather than aborting the whole pipeline.
            logger.warning(f"Color grading failed, using ungraded video. Error: {e}")
            return {"color_graded_path": state["rendered_video_path"]}

    def node_review_safety(self, state: AgentState) -> Dict:
        logger.info("NODE: review_safety")
        callback = state.get("progress_callback")
        if callback:
            callback(95, "Checking caption safety zones...")
            
        edl = state["edl"]
        
        report = self.subtitle_agent.check_safe_zones(edl)
        logger.info(f"Safety Verdict: {report['verdict']}")
        
        return {"safe_zone_report": report}

    def route_after_safety(self, state: AgentState) -> str:
        verdict = state["safe_zone_report"]["verdict"]
        retry_count = state.get("retry_count", 0)
        
        if (verdict == "WARN" or verdict == "FAIL") and retry_count < 5:
            # Extract reasons
            flags = []
            for item in state["safe_zone_report"].get("flagged_items", []):
                flags.append(f"Text '{item['text_overlay']}' has flags: {', '.join(item['flags'])}")
            
            # Update state with feedback for the next loop
            state["edl_feedback"] = (
                "The previous EDL generated text_overlays that violated TikTok/Reels UI safe zones. "
                "The text is either too wide or placed too high/low. "
                "Please shorten the text overlays or do not use text overlays if they are not critical. "
                "Specific violations: " + " | ".join(flags)
            )
            state["retry_count"] = retry_count + 1
            logger.warning(f"Routing back to generate_edl due to Safety WARNING! (Retry {state['retry_count']}/5)")
            return "fail"
            
        if retry_count >= 5 and (verdict == "WARN" or verdict == "FAIL"):
            logger.warning("Maximum safety check retries (5) reached. Proceeding with the current video despite safe-zone warnings.")
            
        return "pass"

    def node_burn_subtitles(self, state: AgentState) -> Dict:
        logger.info("NODE: burn_subtitles")
        callback = state.get("progress_callback")
        if callback:
            callback(98, "Burning dynamic subtitles...")
            
        prompt_lower = state["project_title"].lower()
        requires_subtitles = any(k in prompt_lower for k in ["subtitle", "caption", "text"])
        
        video_path = state.get("color_graded_path") or state["rendered_video_path"]
        final_output = os.path.join(self.exports_dir, state["output_filename"])
        
        if not requires_subtitles:
            logger.info("Subtitles not explicitly requested. Skipping transcription and burning.")
            import shutil
            shutil.copy(video_path, final_output)
            
            try:
                overlay_text = None
                if state["project_title"]:
                    words = state["project_title"].split()
                    overlay_text = " ".join(words[:4])
                self.thumbnail_agent.generate_thumbnail(
                    video_path=final_output,
                    output_dir=self.exports_dir,
                    overlay_text=overlay_text
                )
            except Exception as e:
                logger.warning(f"Cover thumbnail generation failed: {e}")
                
            return {
                "transcription": {},
                "final_video_path": final_output
            }

        logger.info(f"Transcribing audio for {video_path}...")
        transcription = self.subtitle_agent.transcribe(video_path)
        
        dynamic_style = state.get("dynamic_style")
        
        if transcription["captions"]:
            logger.info(f"Burning subtitles to {final_output}...")
            style_param = dynamic_style if dynamic_style else getattr(self.subtitle_agent, "caption_style", "hormozi")
            final_video_path = self.subtitle_agent.burn_subtitles(
                video_path=video_path,
                captions=transcription["captions"],
                output_path=final_output,
                style=style_param
            )
        else:
            logger.info("No spoken words detected. Subtitle burning skipped.")
            import shutil
            shutil.copy(video_path, final_output)
            final_video_path = final_output
            
        try:
            overlay_text = None
            if transcription.get("captions"):
                overlay_text = transcription["captions"][0]["text"]
            elif state["project_title"]:
                words = state["project_title"].split()
                overlay_text = " ".join(words[:4])
            self.thumbnail_agent.generate_thumbnail(
                video_path=final_video_path,
                output_dir=self.exports_dir,
                overlay_text=overlay_text,
                style=dynamic_style
            )
        except Exception as e:
            logger.warning(f"Cover thumbnail generation failed: {e}")

        return {
            "transcription": transcription,
            "final_video_path": final_video_path
        }

    # -------------------------------------------------------------------
    # 3. Graph Execution
    # -------------------------------------------------------------------

    def run(self, initial_state: AgentState):
        """Executes the LangGraph workflow."""
        # Retrieve or generate run_id
        run_id = os.path.basename(self.exports_dir)
        if not run_id or run_id == "exports":
            import uuid
            run_id = f"run_{uuid.uuid4().hex[:8]}"

        start_new_agent_run(run_id, initial_state.get("project_title", "Untitled"))
        logger.info(f"Starting Shortify Pipeline for project: '{initial_state['project_title']}'")
        
        # Invoke the graph
        final_state = self.app.invoke(initial_state)
        
        logger.info("=== SHORTIFY PIPELINE COMPLETE ===")
        logger.info(f"Final Video: {final_state.get('final_video_path')}")
        return final_state

    def _apply_hook_corrections(
        self,
        edl: Dict[str, Any],
        clip_scores: Optional[Dict[str, Any]],
        clips_dir: str
    ) -> Dict[str, Any]:
        """
        Task 30: Post-processes the validated EDL.
        Checks hook position, hook duration, and applies hook quality gate.
        """
        timeline = edl.get("timeline", [])
        if not timeline:
            return edl

        # Helper to recalculate timeline_start and timeline_end
        def recalculate_timeline_times(items: List[Dict[str, Any]]):
            cursor = 0.0
            for item in items:
                dur = float(item["end_in_clip"]) - float(item["start_in_clip"])
                item["timeline_start"] = round(cursor, 3)
                item["timeline_end"] = round(cursor + dur, 3)
                cursor += dur

        # Helper to extract basename from virtual clip name
        def get_base_filename(clip_name: str) -> str:
            source = clip_name
            if ":" in clip_name:
                parts = clip_name.split(":", 2)
                if len(parts) == 3:
                    source = parts[0]
            return os.path.basename(source)

        # 1. Hook Position Correction
        # Find index of the hook segment
        hook_idx = -1
        for idx, item in enumerate(timeline):
            if item.get("details", {}).get("is_hook", False):
                hook_idx = idx
                break

        # If no hook found, force the first item to be the hook
        if hook_idx == -1:
            logger.info("Hook post-processing: No hook found in timeline. Forcing timeline[0] to be hook.")
            timeline[0]["details"]["is_hook"] = True
            hook_idx = 0

        # If hook is not at index 0, swap it to index 0
        if hook_idx > 0:
            logger.info(f"Hook post-processing: Moving hook from position {hook_idx} to position 0.")
            # Swap items in list
            temp = timeline[0]
            timeline[0] = timeline[hook_idx]
            timeline[hook_idx] = temp
            
            # Update detail flags
            timeline[0]["details"]["is_hook"] = True
            timeline[hook_idx]["details"]["is_hook"] = False
            
            recalculate_timeline_times(timeline)

        # 2. Hook Quality Gate Correction
        if clip_scores:
            hook_item = timeline[0]
            hook_base = get_base_filename(hook_item["clip_name"])
            hook_score = clip_scores.get(hook_base, {}).get("composite_score", 0.5)

            if hook_score < 0.5:
                # Scan the first 4 segments (index 0 to 3) for a higher quality clip
                best_candidate_idx = -1
                best_candidate_score = hook_score

                for idx in range(1, min(4, len(timeline))):
                    cand_base = get_base_filename(timeline[idx]["clip_name"])
                    cand_score = clip_scores.get(cand_base, {}).get("composite_score", 0.5)
                    if cand_score > best_candidate_score:
                        best_candidate_score = cand_score
                        best_candidate_idx = idx

                if best_candidate_idx != -1:
                    logger.info(
                        f"Hook post-processing: Swapping low-quality hook ({hook_score:.2f}) "
                        f"with higher-quality candidate ({best_candidate_score:.2f}) at index {best_candidate_idx}."
                    )
                    # Swap items
                    temp = timeline[0]
                    timeline[0] = timeline[best_candidate_idx]
                    timeline[best_candidate_idx] = temp
                    
                    # Update detail flags
                    timeline[0]["details"]["is_hook"] = True
                    timeline[best_candidate_idx]["details"]["is_hook"] = False
                    
                    recalculate_timeline_times(timeline)

        # 3. Hook Duration Correction
        hook_item = timeline[0]
        hook_dur = float(hook_item["end_in_clip"]) - float(hook_item["start_in_clip"])
        if hook_dur > 3.5:
            logger.info(f"Hook post-processing: Trimming hook duration from {hook_dur:.2f}s to 3.5s.")
            # Adjust duration
            hook_item["end_in_clip"] = round(float(hook_item["start_in_clip"]) + 3.5, 3)
            # Recalculate
            recalculate_timeline_times(timeline)

        return edl

