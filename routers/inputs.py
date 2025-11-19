import os, shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from config import SessionLocal, UPLOAD_DIR, logger
from models import Media, Audio, Mood, User
from auth import get_current_user

router = APIRouter(tags=["Inputs"])

# ----- Upload File -----
@router.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    type: str = Form(...),  # video | image | audio
    duration: int = Form(None),
    user: User = Depends(get_current_user),
):
    db = SessionLocal()
    save_dir = UPLOAD_DIR / user.username / "inputs"
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / file.filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if type in ["video", "image"]:
        db.add(Media(user_id=user.id, type=type, filepath=str(filepath)))
    elif type == "audio":
        db.add(Audio(user_id=user.id, filepath=str(filepath), duration=duration or 0))
    else:
        raise HTTPException(status_code=400, detail="Invalid type")
    db.commit()
    logger.info(f"User {user.username} uploaded {type} file: {file.filename}")
    return {"message": f"{type} uploaded", "path": str(filepath)}

# ----- Add Mood -----
@router.post("/mood")
def add_mood(mood_text: str = Form(...), user: User = Depends(get_current_user)):
    db = SessionLocal()
    db.add(Mood(user_id=user.id, mood_text=mood_text))
    db.commit()
    logger.info(f"User {user.username} added mood: {mood_text}")
    return {"message": "Mood added"}
