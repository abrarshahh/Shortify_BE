import pytest
import uuid
from fastapi.testclient import TestClient
from backend_main.main import app
from io import BytesIO

client = TestClient(app)

def test_upload_file():
    # 1. Sign up a new user
    unique_id = uuid.uuid4().hex[:8]
    username = f"user_{unique_id}"
    email = f"user_{unique_id}@example.com"
    signup_res = client.post("/signup", json={"username": username, "email": email, "password": "testpass"})
    assert signup_res.status_code == 200
    session_id = signup_res.json()["session_id"]
    headers = {"Authorization": f"Bearer {session_id}"}

    # 2. Upload a media asset (e.g., image)
    # 1x1 Transparent PNG bytes
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'
    
    files = [("files", ("test.png", BytesIO(png_bytes), "image/png"))]
    response = client.post("/media/upload", headers=headers, files=files)
    assert response.status_code == 201
    json_data = response.json()
    assert "uploaded" in json_data
    assert len(json_data["uploaded"]) == 1
    assert "id" in json_data["uploaded"][0]
    assert "path" in json_data["uploaded"][0]

def test_create_project():
    # 1. Sign up a new user
    unique_id = uuid.uuid4().hex[:8]
    username = f"user_{unique_id}"
    email = f"user_{unique_id}@example.com"
    signup_res = client.post("/signup", json={"username": username, "email": email, "password": "testpass"})
    assert signup_res.status_code == 200
    session_id = signup_res.json()["session_id"]
    headers = {"Authorization": f"Bearer {session_id}"}

    # 2. Create a project
    project_payload = {
        "title": "My Test Project",
        "description": "Testing project creation API",
        "target_duration": 30,
        "aspect_ratio": "9:16",
        "style": "cinematic"
    }
    response = client.post("/projects", headers=headers, json=project_payload)
    assert response.status_code == 201
    proj = response.json()
    assert proj["title"] == "My Test Project"
    assert proj["target_duration"] == 30
    assert proj["aspect_ratio"] == "9:16"
    assert proj["style"] == "cinematic"
    assert "id" in proj
