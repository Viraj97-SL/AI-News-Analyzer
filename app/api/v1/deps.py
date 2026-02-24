"""
Shared FastAPI dependencies for v1 API routes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.core.security import verify_api_key

# Re-export for convenience in route files
AuthenticatedUser = Annotated[str, Depends(verify_api_key)]
AppSettings = Annotated[Settings, Depends(get_settings)]
