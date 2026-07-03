import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_ai.services.relevance_service import RelevanceScorer

@pytest.fixture(autouse=True)
def mock_env_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")


@pytest.fixture
def sample_media_analyses():
    return [
        {
            "file_metadata": {
                "filename": "hiker.mp4",
                "duration_seconds": 10.0
            },
            "interesting_segments": [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "description": "hiker struggling in snow"
                }
            ],
            "all_segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "description": "hiker adjusts boots in snow"
                },
                {
                    "start": 5.0,
                    "end": 10.0,
                    "description": "scenic shot of a frozen pine tree"
                }
            ]
        }
    ]

@patch("backend_ai.services.relevance_service.get_gemini_client")
def test_relevance_scorer_success_gemini(mock_get_client, sample_media_analyses):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = '{"scores": [{"id": "hiker.mp4:highlight:0:0.0:4.0", "relevance_score": 0.95}, {"id": "hiker.mp4:segment:0:0.0:5.0", "relevance_score": 0.90}, {"id": "hiker.mp4:segment:1:5.0:10.0", "relevance_score": 0.20}]}'
    mock_client.models.generate_content.return_value = mock_response

    scorer = RelevanceScorer()
    
    # Run the scorer
    scored_analyses = scorer.score_segments("hiker struggling in snow", sample_media_analyses)
    
    # Assertions
    assert len(scored_analyses) == 1
    analysis = scored_analyses[0]
    
    # Verify highlight score
    assert analysis["interesting_segments"][0]["relevance_score"] == 0.95
    # Verify segment scores
    assert analysis["all_segments"][0]["relevance_score"] == 0.90
    assert analysis["all_segments"][1]["relevance_score"] == 0.20

@patch("backend_ai.services.relevance_service.get_gemini_client")
def test_relevance_scorer_fallback_gemini_models(mock_get_client, sample_media_analyses):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    # First model call fails, second succeeds
    mock_response = MagicMock()
    mock_response.text = '{"scores": [{"id": "hiker.mp4:highlight:0:0.0:4.0", "relevance_score": 0.85}, {"id": "hiker.mp4:segment:0:0.0:5.0", "relevance_score": 0.80}, {"id": "hiker.mp4:segment:1:5.0:10.0", "relevance_score": 0.15}]}'
    
    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Primary model rate limit")
        return mock_response
        
    mock_client.models.generate_content.side_effect = side_effect

    scorer = RelevanceScorer()
    scored_analyses = scorer.score_segments("hiker struggling in snow", sample_media_analyses)
    
    assert scored_analyses[0]["interesting_segments"][0]["relevance_score"] == 0.85
    assert scored_analyses[0]["all_segments"][0]["relevance_score"] == 0.80
    assert scored_analyses[0]["all_segments"][1]["relevance_score"] == 0.15
    assert call_count == 2

@patch("backend_ai.services.relevance_service.get_gemini_client")
def test_relevance_scorer_total_failure_graceful_fallback(mock_get_client, sample_media_analyses):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.models.generate_content.side_effect = Exception("All Gemini models failed")

    scorer = RelevanceScorer()
    
    # Should not raise exception, should assign default 0.5 score
    scored_analyses = scorer.score_segments("hiker struggling in snow", sample_media_analyses)
    
    assert scored_analyses[0]["interesting_segments"][0]["relevance_score"] == 0.5
    assert scored_analyses[0]["all_segments"][0]["relevance_score"] == 0.5
    assert scored_analyses[0]["all_segments"][1]["relevance_score"] == 0.5
