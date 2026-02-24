"""
Email delivery service via Resend API.

Sends the newsletter HTML with inline news card images.
Free tier: 100 emails/day, 3,000/month — more than enough for 2x/week.
"""

from __future__ import annotations

from pathlib import Path

import resend

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmailService:
    def __init__(self) -> None:
        resend.api_key = settings.resend_api_key

    def send_newsletter(
        self,
        html_content: str,
        subject: str = "🤖 Your AI/ML Weekly Digest",
        image_paths: list[str] | None = None,
    ) -> dict:
        """Send the newsletter to all configured recipients."""
        try:
            params: resend.Emails.SendParams = {
                "from": settings.email_from,
                "to": settings.email_recipients,
                "subject": subject,
                "html": html_content,
            }

            # Attach images if provided
            if image_paths:
                attachments = []
                for path_str in image_paths:
                    path = Path(path_str)
                    if path.exists():
                        with open(path, "rb") as f:
                            attachments.append(
                                {"filename": path.name, "content": list(f.read())}
                            )
                if attachments:
                    params["attachments"] = attachments

            result = resend.Emails.send(params)
            logger.info(
                "newsletter_sent",
                email_id=result.get("id"),
                recipients=len(settings.email_recipients),
            )
            return result

        except Exception as e:
            logger.error("email_send_error", error=str(e))
            raise

    def send_approval_email(
        self,
        run_id: str,
        linkedin_preview: str,
        approve_url: str,
        reject_url: str,
    ) -> dict:
        """Send an approval request email with approve/reject links."""
        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>📋 AI News Pipeline — Review Required</h2>
            <p>Run <code>{run_id}</code> has generated content ready for publishing.</p>

            <h3>LinkedIn Post Preview:</h3>
            <div style="background: #f5f5f5; padding: 16px; border-radius: 8px; white-space: pre-wrap;">
                {linkedin_preview[:1500]}
            </div>

            <div style="margin-top: 24px; text-align: center;">
                <a href="{approve_url}"
                   style="background: #0a66c2; color: white; padding: 12px 32px;
                          border-radius: 6px; text-decoration: none; margin-right: 12px;">
                    ✅ Approve & Publish
                </a>
                <a href="{reject_url}"
                   style="background: #dc3545; color: white; padding: 12px 32px;
                          border-radius: 6px; text-decoration: none;">
                    ❌ Reject & Revise
                </a>
            </div>

            <p style="margin-top: 24px; color: #666; font-size: 12px;">
                These links expire in {settings.approval_token_expiry_hours} hours and can only be used once.
            </p>
        </div>
        """

        return resend.Emails.send(
            {
                "from": settings.email_from,
                "to": settings.email_recipients[:1],  # approval goes to primary only
                "subject": f"🔔 Approve AI Newsletter — Run {run_id[:8]}",
                "html": html,
            }
        )
