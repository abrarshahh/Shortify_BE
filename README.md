# Shortify Backend

A FastAPI-based backend for the Shortify AI application, handling user authentication, session management, file uploads, and mood tracking.

## Features

- User registration and login with JWT authentication
- Session creation for user workflows
- File upload support (video, image, audio)
- Mood text addition to sessions
- PostgreSQL database integration

## Project Structure

- `main.py`: FastAPI app entry point
- `config.py`: Database and app configuration
- `models.py`: SQLAlchemy models
- `auth.py`: Authentication utilities
- `routers/`: API endpoints
  - `auth.py`: Signup and login
  - `sessions.py`: Session management
  - `inputs.py`: File uploads and mood
- `tests/`: Unit tests

## Setup

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn sqlalchemy psycopg2-binary passlib[bcrypt] python-jose pytest
   ```

2. Set up PostgreSQL database and update `DATABASE_URL` in `config.py`.

3. Run the app:
   ```bash
   python main.py
   ```

4. Run tests:
   ```bash
   python -m pytest tests/
   ```

## API Endpoints

- `POST /signup`: Register a new user
- `POST /token`: Login and get access token
- `POST /session`: Create a new session (authenticated)
- `POST /upload`: Upload a file (authenticated)
- `POST /mood`: Add mood text (authenticated)