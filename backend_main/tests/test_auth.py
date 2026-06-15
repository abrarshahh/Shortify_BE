import pytest
import uuid
from fastapi.testclient import TestClient
from backend_main.main import app
from backend_main.config import SessionLocal
from backend_main.models import User

client = TestClient(app)

def test_signup():
    unique_id = uuid.uuid4().hex[:8]
    username = f"user_{unique_id}"
    email = f"user_{unique_id}@example.com"
    response = client.post("/signup", json={"username": username, "email": email, "password": "testpass"})
    assert response.status_code == 200
    assert response.json()["message"] == "User created"
    assert "session_id" in response.json()

def test_login():
    unique_id = uuid.uuid4().hex[:8]
    username = f"user_{unique_id}"
    email = f"user_{unique_id}@example.com"
    
    # First signup
    client.post("/signup", json={"username": username, "email": email, "password": "testpass"})
    
    # Then login
    response = client.post("/login", json={"username": username, "password": "testpass"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "session_id" in response.json()
