"""
LinkedIn publishing service — Posts API (replaces deprecated ugcPosts).

Handles:
  - Text-only posts
  - Image posts (two-step upload: initialize → PUT binary → reference URN)
  - Token refresh (access tokens last 60 days, refresh tokens 365 days)
"""

from __future__ import annotations

from pathlib import Path

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

LINKEDIN_API_BASE = "https://api.linkedin.com"
LINKEDIN_API_VERSION = "202501"  # YYYYMM format — update periodically


class LinkedInService:
    def __init__(self) -> None:
        self.access_token = settings.linkedin_access_token
        self.person_urn = settings.linkedin_person_urn

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "LinkedIn-Version": LINKEDIN_API_VERSION,
            "X-Restli-Protocol-Version": "2.0.0",
        }

    def publish_text_post(self, text: str) -> dict:
        """Publish a text-only post to the authenticated user's profile."""
        payload = {
            "author": self.person_urn,
            "commentary": text,
            "visibility": "PUBLIC",
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "lifecycleState": "PUBLISHED",
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{LINKEDIN_API_BASE}/rest/posts",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()

        post_id = resp.headers.get("x-restli-id", "unknown")
        logger.info("linkedin_post_published", post_id=post_id, char_count=len(text))
        return {"post_id": post_id, "status": "published"}

    def upload_image(self, image_path: str) -> str:
        """Upload an image and return its URN for use in a post."""
        # Step 1: Initialize upload
        init_payload = {
            "initializeUploadRequest": {
                "owner": self.person_urn,
            }
        }

        with httpx.Client(timeout=60) as client:
            init_resp = client.post(
                f"{LINKEDIN_API_BASE}/rest/images?action=initializeUpload",
                headers=self._headers,
                json=init_payload,
            )
            init_resp.raise_for_status()
            init_data = init_resp.json()["value"]

            upload_url = init_data["uploadUrl"]
            image_urn = init_data["image"]

            # Step 2: PUT the binary image
            with open(image_path, "rb") as f:
                upload_resp = client.put(
                    upload_url,
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    content=f.read(),
                )
                upload_resp.raise_for_status()

        logger.info("linkedin_image_uploaded", image_urn=image_urn)
        return image_urn

    def publish_image_post(self, text: str, image_path: str) -> dict:
        """Publish a post with an attached image."""
        image_urn = self.upload_image(image_path)

        payload = {
            "author": self.person_urn,
            "commentary": text,
            "visibility": "PUBLIC",
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "content": {
                "media": {
                    "title": "AI/ML Weekly Digest",
                    "id": image_urn,
                }
            },
            "lifecycleState": "PUBLISHED",
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{LINKEDIN_API_BASE}/rest/posts",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()

        post_id = resp.headers.get("x-restli-id", "unknown")
        logger.info("linkedin_image_post_published", post_id=post_id)
        return {"post_id": post_id, "image_urn": image_urn, "status": "published"}

    # TODO: Implement token refresh using requests-oauthlib
    # Access tokens expire in 60 days, refresh tokens in 365 days
    def refresh_access_token(self) -> str:
        """Refresh the LinkedIn access token. Store new token in DB."""
        raise NotImplementedError("Token refresh not yet implemented")
