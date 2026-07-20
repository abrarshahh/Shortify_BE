import pytest
from pydantic import ValidationError
from backend_ai.agents.inspector_agent import EditingInspector
from backend_ai.orchestrator import ShortifyOrchestrator, AgentState

def test_inspector_review_mocked(monkeypatch):
    inspector = EditingInspector()
    
    # Mock calls to Gemini API
    monkeypatch.setattr(inspector, "_call_gemini", lambda model_id, messages: '{"verdict": "REVISE", "feedback": "First clip is too long."}')
    
    review = inspector.review_timeline(
        user_prompt="Hype video",
        timeline_ir={"title": "Test", "storyline": "Hype", "total_duration": 5.0, "video_clips": []}
    )
    
    assert review["verdict"] == "REVISE"
    assert "too long" in review["feedback"]

def test_orchestrator_inspector_node_pass(monkeypatch):
    orchestrator = ShortifyOrchestrator()
    
    # Mock the inspector agent to return PASS
    monkeypatch.setattr(orchestrator.inspector_agent, "review_timeline", lambda user_prompt, timeline_ir: {
        "verdict": "PASS",
        "feedback": "Perfect flow."
    })
    
    state: AgentState = {
        "project_title": "Cool Reel",
        "edl": {
            "title": "Cool Reel",
            "storyline": "Cool",
            "total_duration": 5.0,
            "music_start_offset": 0.0,
            "timeline": []
        },
        "retry_count": 0
    }
    
    updates = orchestrator.node_review_timeline(state)
    assert updates["edl_feedback"] == ""
    assert updates["safe_zone_report"]["verdict"] == "PASS"

def test_orchestrator_inspector_node_revise(monkeypatch):
    orchestrator = ShortifyOrchestrator()
    
    # Mock the inspector agent to return REVISE
    monkeypatch.setattr(orchestrator.inspector_agent, "review_timeline", lambda user_prompt, timeline_ir: {
        "verdict": "REVISE",
        "feedback": "First clip must be the hook."
    })
    
    state: AgentState = {
        "project_title": "Cool Reel",
        "edl": {
            "title": "Cool Reel",
            "storyline": "Cool",
            "total_duration": 5.0,
            "music_start_offset": 0.0,
            "timeline": []
        },
        "retry_count": 0
    }
    
    updates = orchestrator.node_review_timeline(state)
    assert "Editing Inspector requested revision" in updates["edl_feedback"]
    assert updates["retry_count"] == 1
    assert updates["safe_zone_report"]["verdict"] == "WARN"

def test_route_after_review_logic():
    orchestrator = ShortifyOrchestrator()
    
    # 1. Under limit and WARN -> fail (regeneration loop)
    state1: AgentState = {
        "safe_zone_report": {"verdict": "WARN"},
        "retry_count": 1
    }
    assert orchestrator.route_after_review(state1) == "fail"
    
    # 2. Under limit and PASS -> pass (proceed)
    state2: AgentState = {
        "safe_zone_report": {"verdict": "PASS"},
        "retry_count": 1
    }
    assert orchestrator.route_after_review(state2) == "pass"
    
    # 3. Hit limit and WARN -> pass (proceed anyway)
    state3: AgentState = {
        "safe_zone_report": {"verdict": "WARN"},
        "retry_count": 3
    }
    assert orchestrator.route_after_review(state3) == "pass"
