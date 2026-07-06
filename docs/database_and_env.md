# Database Schema & Configuration

This document outlines the PostgreSQL database models, relations, environment variables, and migration reset logic.

---

## 1. Database Schema Models

Shortify AI uses SQLAlchemy ORM models mapped to PostgreSQL tables:

```mermaid
erDiagram
    users ||--o{ projects : "owns"
    users ||--o{ media_assets : "uploads"
    users ||--o{ login_history : "logs"
    projects ||--o{ project_media_assets : "contains (Many-to-Many)"
    media_assets ||--o{ project_media_assets : "linked_to (Many-to-Many)"
    projects ||--o| media_assets : "selected_music (1:1)"
```

### 1.1 User Model (`User`)
- **Table**: `users`
- **Fields**:
  - `id`: `UUID` (Primary Key, default auto-generated)
  - `username`: `VARCHAR(150)` (Unique, Indexed)
  - `email`: `VARCHAR(255)` (Unique, Indexed)
  - `password_hash`: `VARCHAR(255)` (Stored securely using password-hashing)
  - `created_at`: `TIMESTAMP` (Defaults to UTC now)
  - `session_id`: `VARCHAR(255)` (Optional session tracker)

### 1.2 Media Asset Model (`MediaAsset`)
- **Table**: `media_assets`
- **Fields**:
  - `id`: `UUID` (Primary Key)
  - `user_id`: `UUID` (Foreign Key -> `users.id`)
  - `original_filename`: `VARCHAR(255)`
  - `storage_path`: `VARCHAR(512)` (Absolute path to local storage)
  - `mime_type`: `VARCHAR(100)`
  - `file_size`: `BIGINT`
  - `duration`: `FLOAT` (Optional, for video/audio)
  - `width`: `INTEGER` / `height`: `INTEGER` (For video/image dimensions)
  - `thumbnail_path`: `VARCHAR(512)`
  - `extra_metadata`: `JSONB` (Stores frame-rate, codecs, bitrates)
  - `uploaded_at`: `TIMESTAMP`

### 1.3 Project Model (`Project`)
- **Table**: `projects`
- **Fields**:
  - `id`: `UUID` (Primary Key)
  - `user_id`: `UUID` (Foreign Key -> `users.id`)
  - `title`: `VARCHAR(255)`
  - `description`: `TEXT`
  - `target_duration`: `INTEGER` (e.g. 15, 30, 60 seconds)
  - `aspect_ratio`: `VARCHAR(20)` (e.g. `9:16`, `16:9`)
  - `style`: `VARCHAR(100)` (e.g. `cinematic`, `vintage`, `travel`)
  - `music_id`: `UUID` (Foreign Key -> `media_assets.id`)
  - `status`: `VARCHAR(50)` (Current pipeline status: `IDLE`, `RUNNING`, `FAILED`, `COMPLETED`)
  - `output_video_path`: `VARCHAR(512)`
  - `created_at`: `TIMESTAMP`

### 1.4 Project Media Asset Association Table
- **Table**: `project_media_assets`
- **Fields**:
  - `project_id`: `UUID` (Primary Key, Foreign Key -> `projects.id`)
  - `media_asset_id`: `UUID` (Primary Key, Foreign Key -> `media_assets.id`)
  - `added_at`: `TIMESTAMP`

### 1.5 Reel Job Model (`ReelJob`)
- **Table**: `reels`
- **Fields**:
  - `id`: `UUID` (Primary Key)
  - `project_id`: `UUID` (Foreign Key -> `projects.id`)
  - `status`: `VARCHAR(50)`
  - `logs`: `TEXT`
  - `created_at`: `TIMESTAMP`

---

## 2. Environment Configuration (`.env`)

The server and AI engine behaviors are controlled using variables specified in the `.env` file:

| Key | Format | Example | Description |
|---|---|---|---|
| `DATABASE_URL` | PostgreSQL URI | `postgresql://admin:pass@localhost:5432/db` | Connection endpoint for database engine. |
| `SECRET_KEY` | Hex / String | `supersecret` | JWT encryption signature. |
| `ALGORITHM` | String | `HS256` | JWT hash algorithm. |
| `GROQ_API_KEY` | String | `gsk_xxxx...` | Authorization for Llama 3.3 director agent. |
| `GEMINI_API_KEY` | String | `AIzaSy...` | Active Gemini key. |
| `GEMINI_API_KEY_[1-3]` | String | `AIzaSy...` | Backup keys for round-robin rotation. |
| `GIPHY_API_KEY` | String | `GXnAt...` | Authentication for Giphy API searches. |
| `PIXABAY_API_KEY` | String | `56313...` | Authentication for Pixabay API searches. |
| `PIXABAY_APPLY` | Boolean | `false` | Enables/disables overlay download logic. |
| `EDL_VALIDATION_FAIL` | Enum | `pass` | `pass` bypasses EDL syntax errors on 3rd retry; `stop` halts render. |

---

## 3. Database Reset Script (`tests/reset_db.py`)

To reset all temporary folders and DB state during testing or local deployment, developers can run:
```bash
python tests/reset_db.py
```

### 3.1 Script Execution Flow:
1. **Directory Deletion**: Recursively removes the `cache/`, `storage/`, `data/`, and `.pytest_cache/` folders.
2. **Directory Recreation**: Recreates all required base directories empty (`storage/users`, `storage/exports`, `data/fonts`, `data/luts`, etc.).
3. **Log Truncation**: Deletes or truncates logs under `logs/`.
4. **Table Drop**: Executes SQL drops in reverse foreign key dependency order to clean the database without locking constraint failures:
   ```sql
   DROP TABLE IF EXISTS reels CASCADE;
   DROP TABLE IF EXISTS project_media_assets CASCADE;
   DROP TABLE IF EXISTS login_history CASCADE;
   DROP TABLE IF EXISTS media_assets CASCADE;
   DROP TABLE IF EXISTS projects CASCADE;
   DROP TABLE IF EXISTS users CASCADE;
   ```
5. **Schema Synchronization**: Imports SQLAlchemy metadata models and calls `Base.metadata.create_all(bind=engine)` to rebuild the database schema.
