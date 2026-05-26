import os
from typing import Dict, Any, List, Optional, Callable
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

# Import our agents
from backend_ai.services.rhythm_service import RhythmEngineer
from backend_ai.services.media_service import MediaAnalyst
from backend_ai.services.director_service import CreativeDirector
from backend_ai.services.editor_service import VideoEditor
from backend_ai.services.subtitle_service import SubtitleAgent
from backend_ai.services.color_service import ColorGradingAgent
from backend_ai.services.analyst_service import ProjectAnalystAgent
from backend_ai.services.thumbnail_service import ThumbnailAgent
from backend_ai.services.edl_validation_service import validate_edl
from backend_ai.schemas.edl import EDLGenerationError, EDLValidationError
from backend_ai.core.config_loader import AGENTS_CONFIG

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
        
        # Subtitle config
        sub_config = AGENTS_CONFIG.get("subtitle_agent", {})
        self.subtitle_agent = SubtitleAgent(
            model_size=sub_config.get("model_size", "base"),
            device=sub_config.get("device", "cpu")
        )
        
        # Build and compile the graph
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_rhythm", self.node_analyze_rhythm)
        workflow.add_node("pre_flight_check", self.node_pre_flight_check)
        workflow.add_node("analyze_media", self.node_analyze_media)
        workflow.add_node("generate_edl", self.node_generate_edl)
        workflow.add_node("render_video", self.node_render_video)
        workflow.add_node("color_grade", self.node_color_grade)
        workflow.add_node("review_safety", self.node_review_safety)
        workflow.add_node("burn_subtitles", self.node_burn_subtitles)

        # Set edges
        workflow.set_entry_point("pre_flight_check")
        workflow.add_edge("pre_flight_check", "analyze_rhythm")
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
        print("\n--- NODE: pre_flight_check ---")
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
        print("\n--- NODE: analyze_rhythm ---")
        callback = state.get("progress_callback")
        if callback:
            callback(25, "Analyzing audio beats...")
            
        music_path = state.get("music_path")
        
        if not music_path or not os.path.exists(music_path):
            print("No music path provided or found. Skipping rhythm analysis.")
            return {"rhythm_data": {}}
        
        print(f"Analyzing beats for: {music_path}")
        rhythm_data = self.rhythm_agent.analyze_music(music_path)
        return {"rhythm_data": rhythm_data}

    def node_analyze_media(self, state: AgentState) -> Dict:
        print("\n--- NODE: analyze_media ---")
        callback = state.get("progress_callback")
        if callback:
            callback(50, "AI Media Analysis...")
            
        visual_data = []
        for path in state["video_paths"]:
            if os.path.exists(path):
                print(f"Analyzing visual context for: {path}")
                analysis = self.media_agent.analyze_video(path)
                visual_data.append(analysis)
                # Small delay to avoid bursting the Gemini API rate limit
                import time
                time.sleep(1.5)
            else:
                print(f"Warning: Video not found at {path}")
                
        return {"visual_data": visual_data}

    def node_generate_edl(self, state: AgentState) -> Dict:
        print("\n--- NODE: generate_edl ---")
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
            print(f"Generating EDL. Prompt: {prompt}")
            edl = self.director_agent.generate_edl(
                user_prompt=prompt,
                media_analyses=state["visual_data"],
                audio_analysis=state.get("rhythm_data", {}),
                target_duration=state["target_duration"],
                aspect_ratio=state["aspect_ratio"],
                style=state["style"],
                feedback=feedback
            )

            try:
                validated_edl = validate_edl(edl, clips_dir, target_duration=float(state["target_duration"]))
                return {
                    "edl": validated_edl.model_dump(mode="json"),
                    "edl_feedback": "",
                    "max_edl_retries": max_edl_retries,
                }
            except EDLValidationError as exc:
                max_edl_retries += 1
                state["max_edl_retries"] = max_edl_retries
                feedback = exc.to_feedback()
                state["edl_feedback"] = feedback

                print(f"EDL validation failed (attempt {max_edl_retries}/3): {feedback}")
                if max_edl_retries >= 3:
                    raise EDLGenerationError(
                        retry_count=max_edl_retries,
                        last_error=feedback,
                        issues=exc.issues,
                    )

                continue

    def node_render_video(self, state: AgentState) -> Dict:
        print("\n--- NODE: render_video ---")
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
        
        print(f"Rendering EDL to {output_filename}...")
        rendered_path = editor.render(
            edl=state["edl"],
            music_path=state.get("music_path"),
            output_filename=output_filename,
            aspect_ratio=state.get("aspect_ratio", "9:16"),
            rhythm_data=state.get("rhythm_data", {})
        )
        
        return {"rendered_video_path": rendered_path}

    def node_color_grade(self, state: AgentState) -> Dict:
        print("\n--- NODE: color_grade ---")
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
            print(f"  Warning: Color grading failed, using ungraded video. Error: {e}")
            return {"color_graded_path": state["rendered_video_path"]}

    def node_review_safety(self, state: AgentState) -> Dict:
        print("\n--- NODE: review_safety ---")
        callback = state.get("progress_callback")
        if callback:
            callback(95, "Checking caption safety zones...")
            
        edl = state["edl"]
        
        report = self.subtitle_agent.check_safe_zones(edl)
        print(f"Safety Verdict: {report['verdict']}")
        
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
            print(f"Routing back to generate_edl due to Safety WARNING! (Retry {state['retry_count']}/5)")
            return "fail"
            
        if retry_count >= 5 and (verdict == "WARN" or verdict == "FAIL"):
            print("Maximum safety check retries (5) reached. Proceeding with the current video despite safe-zone warnings.")
            
        return "pass"

    def node_burn_subtitles(self, state: AgentState) -> Dict:
        print("\n--- NODE: burn_subtitles ---")
        callback = state.get("progress_callback")
        if callback:
            callback(98, "Burning dynamic subtitles...")
            
        prompt_lower = state["project_title"].lower()
        requires_subtitles = any(k in prompt_lower for k in ["subtitle", "caption", "text"])
        
        video_path = state.get("color_graded_path") or state["rendered_video_path"]
        final_output = os.path.join(self.exports_dir, state["output_filename"])
        
        if not requires_subtitles:
            print("Subtitles not explicitly requested. Skipping transcription and burning.")
            import shutil
            shutil.copy(video_path, final_output)
        # Generate cover thumbnail for opt-out subtitles
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
            print(f"  Warning: Cover thumbnail generation failed: {e}")

        return {
            "transcription": {},
            "final_video_path": final_output
        }

        print(f"Transcribing audio for {video_path}...")
        transcription = self.subtitle_agent.transcribe(video_path)
        
        if transcription["captions"]:
            print(f"Burning subtitles to {final_output}...")
            final_video_path = self.subtitle_agent.burn_subtitles(
                video_path=video_path,
                captions=transcription["captions"],
                output_path=final_output
            )
        else:
            print("No spoken words detected. Subtitle burning skipped.")
            # Still copy or rename to final output
            import shutil
            shutil.copy(video_path, final_output)
            final_video_path = final_output
            
        # Generate cover thumbnail for opt-in subtitles
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
                overlay_text=overlay_text
            )
        except Exception as e:
            print(f"  Warning: Cover thumbnail generation failed: {e}")

        return {
            "transcription": transcription,
            "final_video_path": final_video_path
        }

    # -------------------------------------------------------------------
    # 3. Graph Execution
    # -------------------------------------------------------------------

    def run(self, initial_state: AgentState):
        """Executes the LangGraph workflow."""
        print(f"Starting Shortify Pipeline for: '{initial_state['project_title']}'")
        
        # Invoke the graph
        final_state = self.app.invoke(initial_state)
        
        print("\n=== SHORTIFY PIPELINE COMPLETE ===")
        print(f"Final Video: {final_state.get('final_video_path')}")
        return final_state

