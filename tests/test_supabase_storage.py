import os
import pytest
from unittest import mock
from backend_main.supabase_storage import is_supabase_configured, upload_to_supabase, download_from_supabase

def test_supabase_configured_toggle():
    # Patch module level globals
    with mock.patch("backend_main.supabase_storage.SUPABASE_URL", "http://mock.supabase.co"), \
         mock.patch("backend_main.supabase_storage.SUPABASE_KEY", "mockkey"):
        
        # Scenario 1: USE_SUPABASE is false or unset (default)
        with mock.patch.dict(os.environ, {"USE_SUPABASE": "false"}):
            assert is_supabase_configured() is False

        # Scenario 2: USE_SUPABASE is true, and keys are set
        with mock.patch.dict(os.environ, {"USE_SUPABASE": "true"}):
            assert is_supabase_configured() is True

    # Scenario 3: USE_SUPABASE is true, but keys are missing
    with mock.patch("backend_main.supabase_storage.SUPABASE_URL", ""), \
         mock.patch("backend_main.supabase_storage.SUPABASE_KEY", ""):
        with mock.patch.dict(os.environ, {"USE_SUPABASE": "true"}):
            assert is_supabase_configured() is False

def test_supabase_operations_disabled_when_false():
    with mock.patch("backend_main.supabase_storage.SUPABASE_URL", "http://mock.supabase.co"), \
         mock.patch("backend_main.supabase_storage.SUPABASE_KEY", "mockkey"):
        with mock.patch.dict(os.environ, {"USE_SUPABASE": "false"}):
            # Operations should return False instantly without making network requests
            assert upload_to_supabase("mock_local_path.mp4", "mock_storage_path.mp4") is False
            assert download_from_supabase("mock_storage_path.mp4", "mock_local_path.mp4") is False
