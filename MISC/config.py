import os
from datetime import datetime, timezone


# Attempt to import env.py if present to allow setting env vars via a python file.
# If present, values in env.py take precedence over OS environment variables.
try:
    import env as _env  # type: ignore
except Exception:
    _env = None


def _env_get(name: str, default=None):
    """Return value from local env.py if present, otherwise from OS env, else default."""
    if _env is not None and hasattr(_env, name):
        return getattr(_env, name)
    return os.getenv(name, default)


# Use env.py values first, then OS environment variables, then defaults
ANALYSIS_DURATION = int(_env_get("ANALYSIS_DURATION", "7"))
TLE_AGE_LIMIT = int(_env_get("TLE_AGE_LIMIT", "10"))
API_BASE_URL = _env_get("API_BASE_URL", "http://127.0.0.1")
BATCH_SIZE = int(_env_get("BATCH_SIZE", "0") or 0)

# Ephemeris upload settings
EPHEMERIS_UPLOAD_PATH = _env_get("EPHEMERIS_UPLOAD_PATH", "/api/v1/ephemeris/upload")
# Optional API key for Authorization header when uploading ephemeris
EPHEMERIS_API_KEY = _env_get("EPHEMERIS_API_KEY")

# Bearer token for CA event API (falls back to EPHEMERIS_API_KEY if not set)
CA_API_TOKEN = _env_get("CA_API_TOKEN") or _env_get("EPHEMERIS_API_KEY")

# Preferred TLE source: 'auto' | 'spacetrack' | 'celestrak'
TLE_SOURCE = _env_get("TLE_SOURCE", "auto")


def now_epoch():
    return datetime.now(timezone.utc)
