import os
import logging
import httpx

logger = logging.getLogger("backend_main.supabase_storage")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "shortify")

def is_supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)

def upload_to_supabase(local_path: str, storage_path: str, mime_type: str = "application/octet-stream") -> bool:
    """
    Uploads a local file to the Supabase storage bucket.
    """
    if not is_supabase_configured():
        return False
    
    if not os.path.exists(local_path):
        logger.error(f"[Supabase Storage] Local file does not exist: {local_path}")
        return False

    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{storage_path}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": mime_type,
        "x-upsert": "true"
    }
    
    try:
        with open(local_path, "rb") as f:
            file_data = f.read()
            
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, headers=headers, content=file_data)
            if response.status_code in (200, 201):
                logger.info(f"[Supabase Storage] Successfully uploaded {local_path} to {storage_path}")
                return True
            else:
                logger.error(f"[Supabase Storage] Upload failed for {local_path}. Status: {response.status_code}, Body: {response.text}")
                return False
    except Exception as e:
        logger.error(f"[Supabase Storage] Exception during upload of {local_path}: {e}")
        return False

def download_from_supabase(storage_path: str, local_path: str) -> bool:
    """
    Downloads a file from the public Supabase storage bucket to a local path.
    """
    if not is_supabase_configured():
        return False

    url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_path}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(url)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"[Supabase Storage] Successfully downloaded {storage_path} to {local_path}")
                return True
            else:
                logger.error(f"[Supabase Storage] Download failed for {storage_path}. Status: {response.status_code}")
                return False
    except Exception as e:
        logger.error(f"[Supabase Storage] Exception during download of {storage_path}: {e}")
        return False
