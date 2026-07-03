import os
import sys
import uuid
import pytest
from unittest.mock import MagicMock, patch

# Add project root to sys.path to allow imports from backend_ai and backend_main
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend_main.main import app
from backend_main.auth import get_current_user
from backend_main.models import Project, User
from backend_main import worker_service
from backend_main.worker_service import RenderCancelledError

client = TestClient(app)

# Mock User and Project for tests
mock_user = User(id=uuid.uuid4(), username="testuser", session_id="test-session-id")

@pytest.fixture(autouse=True)
def clean_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()

def test_render_cancelled_error_exists():
    assert issubclass(RenderCancelledError, Exception)

def test_cancel_queued_job_via_future():
    project_id = str(uuid.uuid4())
    mock_future = MagicMock()
    mock_future.cancel.return_value = True

    worker_service.active_futures[project_id] = mock_future
    
    mock_db = MagicMock()
    with patch("backend_main.worker_service.SessionLocal", return_value=mock_db):
        result = worker_service.cancel_job(project_id)
        
    assert result is True
    assert mock_future.cancel.called
    assert project_id not in worker_service.active_futures
    assert worker_service.render_jobs[project_id]["status"] == "cancelled"
    assert "cancelled before starting" in worker_service.render_jobs[project_id]["message"].lower()

def test_cancel_running_job_cooperative_cancellation():
    project_id = str(uuid.uuid4())
    
    # 1. Flag job as queued/running
    worker_service.render_jobs[project_id] = {
        "status": "running",
        "is_cancelled": False
    }
    
    # 2. Trigger worker_service.cancel_job
    result = worker_service.cancel_job(project_id)
    assert result is True
    assert worker_service.render_jobs[project_id]["is_cancelled"] is True
    assert worker_service.render_jobs[project_id]["status"] == "cancelled"

@patch("backend_main.worker_service.ShortifyOrchestrator")
@patch("backend_main.worker_service.SessionLocal")
@patch("backend_main.worker_service.STORAGE_ROOT")
def test_execute_render_task_handles_cancellation_exception(mock_storage_root, mock_session_local, mock_orchestrator_cls):
    project_id = str(uuid.uuid4())
    
    # Mock DB Session
    mock_db = MagicMock()
    mock_session_local.return_value = mock_db
    
    # Mock orchestrator instance
    mock_orchestrator = MagicMock()
    mock_orchestrator_cls.return_value = mock_orchestrator
    
    # Simulate orchestrator running and triggering progress callback
    def simulate_run(state):
        callback = state["progress_callback"]
        # Trigger progress callback which should raise RenderCancelledError since we cancelled it
        callback(10, "Initializing pipeline...")
        return state
        
    mock_orchestrator.run.side_effect = simulate_run
    
    # Set job state to running but flag as cancelled
    worker_service.render_jobs[project_id] = {
        "status": "running",
        "is_cancelled": True
    }
    
    # Execute worker task synchronously
    worker_service._execute_render_task(
        project_id=project_id,
        prompt="Test prompt",
        video_paths=["test1.mp4"],
        music_path="music.mp3",
        output_filename="output.mp4",
        target_duration=15,
        aspect_ratio="9:16",
        style="cinematic"
    )
    
    # Assert status is set to cancelled and cleaned up
    assert worker_service.render_jobs[project_id]["status"] == "cancelled"
    assert "cancelled by user request" in worker_service.render_jobs[project_id]["message"].lower()

def test_delete_render_endpoint_unauthorized():
    response = client.delete(f"/projects/{uuid.uuid4()}/render")
    assert response.status_code == 403

def test_delete_render_endpoint_not_found():
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = None
    
    with patch("backend_main.routers.render.SessionLocal", return_value=mock_db):
        response = client.delete(f"/projects/{uuid.uuid4()}/render", headers={"Authorization": "Bearer test-session-id"})
        
    assert response.status_code == 404
    assert "Project not found" in response.json()["detail"]

def test_delete_render_endpoint_forbidden():
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    another_user_id = uuid.uuid4()
    mock_project = Project(id=uuid.uuid4(), user_id=another_user_id, title="Other Project", is_rendering=True)
    
    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = mock_project
    
    with patch("backend_main.routers.render.SessionLocal", return_value=mock_db):
        response = client.delete(f"/projects/{mock_project.id}/render", headers={"Authorization": "Bearer test-session-id"})
        
    assert response.status_code == 403
    assert "Forbidden" in response.json()["detail"]

def test_delete_render_endpoint_success():
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    project_id = uuid.uuid4()
    mock_project = Project(id=project_id, user_id=mock_user.id, title="My Project", is_rendering=True)
    
    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = mock_project
    
    with patch("backend_main.routers.render.SessionLocal", return_value=mock_db), \
         patch("backend_main.routers.render.worker_service.cancel_job") as mock_cancel_job:
         
        mock_cancel_job.return_value = True
        
        response = client.delete(f"/projects/{project_id}/render", headers={"Authorization": "Bearer test-session-id"})
        
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert mock_project.is_rendering is False
    mock_cancel_job.assert_called_once_with(str(project_id))
