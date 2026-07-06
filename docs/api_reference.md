# API Reference Guide

This document lists the REST API routes, parameter parameters, payload structures, and database interactions exposed via FastAPI.

---

## 1. Authentication Router (`/`)

Security is managed using JWT authentication tokens. Users must pass the `Authorization: Bearer <token>` header on protected endpoints.

### 1.1 POST `/signup`
- **Description**: Registers a new user.
- **Payload**:
  ```json
  {
    "username": "user123",
    "email": "user@example.com",
    "password": "strongpassword"
  }
  ```
- **Response**: Created user record (excluding password).

### 1.2 POST `/login`
- **Description**: Verifies credentials and returns token.
- **Payload**:
  ```json
  {
    "username": "user123",
    "password": "strongpassword"
  }
  ```
- **Response**: Access token and session info.

---

## 2. Projects Router (`/projects`)

### 2.1 GET `/projects`
- **Description**: Lists all projects owned by the authenticated user, including their render status.

### 2.2 POST `/projects`
- **Description**: Registers a new project metadata model.
- **Payload**:
  ```json
  {
    "title": "Cinematic Forest Video",
    "description": "Slow pacing forest trek edit"
  }
  ```

### 2.3 GET `/projects/{project_id}`
- **Description**: Retrieves full details of a specific project, including linked media assets and active background music.

### 2.4 DELETE `/projects/{project_id}`
- **Description**: Soft delete of project record (keeps files).

### 2.5 DELETE `/projects/{project_id}/hard`
- **Description**: Hard delete: removes project, association records, and exclusive files from disk storage.

### 2.6 DELETE `/projects/{project_id}/cache`
- **Description**: Removes all cached project analysis files recursively.

---

## 3. Media & Audio Assets Router (`/media` & `/audio`)

### 3.1 POST `/media/project/{project_id}/upload`
- **Description**: Uploads raw videos/photos and links them to the project.
- **Form**: Multipart file uploads.

### 3.2 POST `/media/project/{project_id}/link`
- **Description**: Links an existing library asset to the project.
- **Payload**:
  ```json
  {
    "media_asset_id": "uuid-here"
  }
  ```

### 3.3 POST `/audio/project/{project_id}/upload`
- **Description**: Uploads an audio file and sets it as the project's background music.

### 3.4 POST `/audio/project/{project_id}/link`
- **Description**: Sets an existing library audio asset as the project music.

---

## 4. Rendering Router (`/projects/{project_id}/render`)

### 4.1 POST `/projects/{project_id}/render`
- **Description**: Triggers the LangGraph pipeline execution in a background worker thread.
- **Query / Body Parameters**:
  - `target_duration`: Target length in seconds.
  - `aspect_ratio`: Target layout (`9:16`, `16:9`).
  - `style`: Aesthetic theme (`cinematic`, `vintage`, `travel`).
  - `caption_style`: Overlay typography (`hormozi`, `clean`, `none`).
  - `add_subtitle`: Booleans to burn speech subtitles.
  - `add_stickers`: Toggle Giphy sticker overlay.
  - `add_textoverlay`: Toggle Creative Director planned title overlays.
- **Response**: `202 Accepted` with rendering job details.

### 4.2 GET `/projects/{project_id}/render/status`
- **Description**: Polls the active status (`IDLE`, `RUNNING`, `FAILED`, `COMPLETED`).
