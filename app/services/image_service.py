"""
Image service — centralises news card generation and S3 upload.

For MVP, images are stored locally and served from the FastAPI static mount.
For production, upload to S3/CloudFront and return public URLs.
"""

from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("./output/images")


class ImageService:
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def get_public_url(self, local_path: str) -> str:
        """Convert a local image path to a publicly accessible URL."""
        filename = Path(local_path).name
        # MVP: serve from FastAPI static mount
        return f"{self.base_url}/static/images/{filename}"

    def get_all_card_urls(self, image_paths: list[str]) -> list[str]:
        """Convert all local paths to public URLs."""
        return [self.get_public_url(p) for p in image_paths]

    # TODO: Add S3 upload method for production
    # def upload_to_s3(self, local_path: str) -> str: ...
