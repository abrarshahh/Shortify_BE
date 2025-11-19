import pytest
from fastapi.testclient import TestClient
from main import app
from io import BytesIO

client = TestClient(app)

def test_upload_file():
    # Signup, login, create session
    client.post("/signup", json={"username": "testuser4", "email": "test4@example.com", "password": "testpass"})
    login_response = client.post("/token", data={"username": "testuser4", "password": "testpass"})
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    session_response = client.post("/session", headers=headers)
    session_id = session_response.json()["session_id"]

    # Upload a file
    file_content = b"test file content"
    files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}
    data = {"session_id": session_id, "type": "image"}
    response = client.post("/upload", headers=headers, files=files, data=data)
    assert response.status_code == 200
    assert "message" in response.json()

def test_add_mood():
    # Signup, login, create session
    client.post("/signup", json={"username": "testuser5", "email": "test5@example.com", "password": "testpass"})
    login_response = client.post("/token", data={"username": "testuser5", "password": "testpass"})
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    session_response = client.post("/session", headers=headers)
    session_id = session_response.json()["session_id"]

    # Add mood
    data = {"session_id": session_id, "mood_text": "Happy"}
    response = client.post("/mood", headers=headers, data=data)
    assert response.status_code == 200
    assert response.json() == {"message": "Mood added"}
