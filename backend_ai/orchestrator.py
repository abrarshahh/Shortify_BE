import os
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

# Import our agents
from backend_ai.services.rhythm_service import RhythmEngineer
from backend_ai.services.media_service import MediaAnalyst
from backend_ai.services.director_service import CreativeDirector
from backend_ai.services.editor_service import VideoEditor
from backend_ai.services.subtitle_service import SubtitleAgent

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
    safe_zone_report: Dict[str, Any]
    transcription: Dict[str, Any]
    final_video_path: str
    retry_count: int


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
        self.subtitle_agent = SubtitleAgent(model_size="base", device="cpu")
        
        # Build and compile the graph
        self.app = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_rhythm", self.node_analyze_rhythm)
        workflow.add_node("analyze_media", self.node_analyze_media)
        workflow.add_node("generate_edl", self.node_generate_edl)
        workflow.add_node("render_video", self.node_render_video)
        workflow.add_node("review_safety", self.node_review_safety)
        workflow.add_node("burn_subtitles", self.node_burn_subtitles)

        # Set edges
        workflow.set_entry_point("analyze_rhythm")
        workflow.add_edge("analyze_rhythm", "analyze_media")
        workflow.add_edge("analyze_media", "generate_edl")
        workflow.add_edge("generate_edl", "render_video")
        workflow.add_edge("render_video", "review_safety")
        
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

    def node_analyze_rhythm(self, state: AgentState) -> Dict:
        print("\n--- NODE: analyze_rhythm ---")
        music_path = state.get("music_path")
        
        if not music_path or not os.path.exists(music_path):
            print("No music path provided or found. Skipping rhythm analysis.")
            return {"rhythm_data": {}}
        
        print(f"Analyzing beats for: {music_path}")
        rhythm_data = self.rhythm_agent.analyze_music(music_path)
        return {"rhythm_data": rhythm_data}

    def node_analyze_media(self, state: AgentState) -> Dict:
        print("\n--- NODE: analyze_media ---")
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
        
        feedback = state.get("edl_feedback")
        
        # If we have feedback from a previous safety failure, we should inform the LLM
        # For now, since CreativeDirector doesn't natively accept a "feedback" parameter,
        # we will append it to the project_title/prompt temporarily or we could just 
        # let the director generate a new one with a modified prompt.
        prompt = state["project_title"]
        if feedback:
            print("Applying Safety Feedback to EDL generation!")
            prompt += f"\n\nCRITICAL FIX REQUIRED: {feedback}"
        
        print(f"Generating EDL. Prompt: {prompt}")
        edl = self.director_agent.generate_edl(
            user_prompt=prompt,
            media_analyses=state["visual_data"],
            audio_analysis=state.get("rhythm_data", {}),
            target_duration=state["target_duration"],
            aspect_ratio=state["aspect_ratio"],
            style=state["style"]
        )
        
        return {"edl": edl, "edl_feedback": ""} # clear feedback after applying

    def node_render_video(self, state: AgentState) -> Dict:
        print("\n--- NODE: render_video ---")
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
            output_filename=output_filename
        )
        
        return {"rendered_video_path": rendered_path}

    def node_review_safety(self, state: AgentState) -> Dict:
        print("\n--- NODE: review_safety ---")
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
        
        video_path = state["rendered_video_path"]
        print(f"Transcribing audio for {video_path}...")
        transcription = self.subtitle_agent.transcribe(video_path)
        
        final_output = os.path.join(self.exports_dir, state["output_filename"])
        
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

