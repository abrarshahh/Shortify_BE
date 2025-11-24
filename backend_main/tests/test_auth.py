import pytest
from fastapi.testclient import TestClient
from backend_main.main import app
from backend_main.config import SessionLocal
from backend_main.models import User

client = TestClient(app)

def test_signup():
    response = client.post("/signup", json={"username": "testuser", "email": "test@example.com", "password": "testpass"})
    assert response.status_code == 200
    assert response.json() == {"message": "User created"}

def test_login():
    # First signup
    client.post("/signup", json={"username": "testuser2", "email": "test2@example.com", "password": "testpass"})
    # Then login
    response = client.post("/token", data={"username": "testuser2", "password": "testpass"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"
