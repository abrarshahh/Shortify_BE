# Shortify Backend

A FastAPI-based backend for the Shortify AI application, handling user authentication, project and media management, file uploads, and music selection.

## Features

- User registration and login with secure password hashing (argon2) and JWT session management
- Project creation with metadata (title, description, target duration, aspect ratio, style)
- Media file uploads (video, image, audio) associated with projects
- Music file upload and selection for projects (audio only, max 50MB)
- Retrieval of user media and music assets
- PostgreSQL database integration with SQLAlchemy ORM
- File storage organized by user and project in a dedicated storage directory
- Logging of important events to file and console

## Project Structure

- `main.py`: FastAPI application initialization and route inclusion
- `config.py`: Configuration for database URL, engine setup, logging, and storage directory initialization
- `models.py`: SQLAlchemy ORM models defining User, Project, MediaAsset, etc.
- `auth.py`: User authentication utilities and route definitions for signup and login
- `routers/`: API endpoint routers
  - `auth.py`: Signup and login endpoints
  - `inputs.py`: Project creation, media and music upload, media management endpoints
- `tests/`: Unit and integration tests for the application

## Setup

1. Clone the repository and navigate to the project directory.

2. Create a Python virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following content:
   ```
   DATABASE_URL=postgresql://user:password@localhost/dbname
   ```

5. Ensure PostgreSQL is running and the configured database exists.

## Running the Application

Start the FastAPI application using uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `POST /signup`: Register a new user
- `POST /login`: Login and get session ID
- `POST /projects`: Create a new project
- `POST /projects/{project_id}/media`: Upload media files to a project
- `POST /projects/{project_id}/music`: Upload music file to a project
- `PUT /projects/{project_id}/music`: Select uploaded music for a project
- `PUT /projects/{project_id}/media`: Add existing media files to a project
- `GET /media`: List uploaded media files
- `GET /music`: List uploaded music files
   DATABASE_URL=postgresql://user:password@localhost/dbname
   pip install -r requirements.txt
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
