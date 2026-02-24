"""
Security utilities: API key auth, HMAC-signed approval tokens, rate limiting.

Aligned with OWASP LLM Top 10 (2025) — see Section 9 of the design doc.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from jose import JWTError, jwt
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import Settings, get_settings

# ── Rate limiter (attached to FastAPI app in main.py) ───────
limiter = Limiter(key_func=get_remote_address)

# ── API Key authentication ──────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Annotated[str | None, Security(_api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    if not api_key or not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
    return api_key


# ── HMAC-signed approval tokens (for email approve/reject links) ─
def create_approval_token(run_id: str, action: str, settings: Settings) -> str:
    """Create a short-lived JWT for one-click approve/reject from email."""
    payload = {
        "run_id": run_id,
        "action": action,  # "approve" or "reject"
        "exp": datetime.now(UTC) + timedelta(hours=settings.approval_token_expiry_hours),
        "jti": secrets.token_hex(16),  # unique token ID for one-time-use
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_approval_token(token: str, settings: Settings) -> dict:
    """Verify and decode an approval token. Raises on expiry / tampering."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired approval token: {e}",
        ) from e
    return payload


# ── Content sanitisation (OWASP LLM02 - Information Disclosure) ─
def sanitize_for_display(text: str) -> str:
    """Strip potential prompt injection patterns and sensitive markers from output."""
    dangerous_patterns = [
        "SYSTEM:", "ASSISTANT:", "USER:", "```system",
        "<|im_start|>", "<|im_end|>", "<<SYS>>", "<</SYS>>",
    ]
    sanitized = text
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern, "[REDACTED]")
    return sanitized


def hash_content(content: str) -> str:
    """Deterministic content hash for deduplication & cache keys."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
