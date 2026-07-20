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
        "description": "Testing project creation API"
    }
    response = client.post("/projects", headers=headers, json=project_payload)
    assert response.status_code == 201
    proj = response.json()
    assert proj["title"] == "My Test Project"
    assert proj["target_duration"] == 15
    assert proj["aspect_ratio"] == "9:16"
    assert "id" in proj

def test_delete_project_cache():
    # 1. Sign up a new user
    import os
    import shutil
    unique_id = uuid.uuid4().hex[:8]
    username = f"user_{unique_id}"
    email = f"user_{unique_id}@example.com"
    signup_res = client.post("/signup", json={"username": username, "email": email, "password": "testpass"})
    assert signup_res.status_code == 200
    session_id = signup_res.json()["session_id"]
    headers = {"Authorization": f"Bearer {session_id}"}

    # 2. Create a project
    project_payload = {
        "title": "My Cache Project",
        "description": "Testing cache deletion"
    }
    response = client.post("/projects", headers=headers, json=project_payload)
    assert response.status_code == 201
    proj = response.json()
    project_id = proj["id"]

    # 3. Create dummy cache directory and file
    cache_path = os.path.join("cache", username, project_id)
    os.makedirs(cache_path, exist_ok=True)
    dummy_file = os.path.join(cache_path, "director_analysis.json")
    with open(dummy_file, "w") as f:
        f.write("{}")
    
    assert os.path.exists(dummy_file)

    # 4. Trigger cache deletion endpoint
    del_res = client.delete(f"/projects/{project_id}/cache", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["message"] == "Project cache successfully deleted."

    # 5. Verify folder was deleted
    assert not os.path.exists(cache_path)

    # Clean up empty parent folders
    user_cache_dir = os.path.dirname(cache_path)
    if os.path.exists(user_cache_dir) and not os.listdir(user_cache_dir):
        os.rmdir(user_cache_dir)
    cache_root = os.path.dirname(user_cache_dir)
    if os.path.exists(cache_root) and not os.listdir(cache_root):
        os.rmdir(cache_root)

def test_update_project():
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
        "title": "Initial Title",
        "description": "Initial description"
    }
    response = client.post("/projects", headers=headers, json=project_payload)
    assert response.status_code == 201
    proj = response.json()
    project_id = proj["id"]

    # 3. Patch the project details
    update_payload = {
        "title": "Updated Title",
        "description": "Updated description",
        "target_duration": 30,
        "aspect_ratio": "16:9",
        "style": "cinematic"
    }
    update_response = client.patch(f"/projects/{project_id}", headers=headers, json=update_payload)
    assert update_response.status_code == 200
    updated_proj = update_response.json()
    assert updated_proj["title"] == "Updated Title"
    assert updated_proj["description"] == "Updated description"
    assert updated_proj["target_duration"] == 30
    assert updated_proj["aspect_ratio"] == "16:9"
    assert updated_proj["style"] == "cinematic"
