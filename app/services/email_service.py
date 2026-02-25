"""
Email delivery service via SMTP (Gmail-compatible).

Uses Python's built-in smtplib with STARTTLS so it works with any
SMTP provider: Gmail App Passwords, SendGrid, Mailgun, etc.

Gmail setup:
  1. Enable 2-Step Verification on your Google account.
  2. Generate an App Password (Google Account → Security → App Passwords).
  3. Set SMTP_USER=you@gmail.com and SMTP_PASSWORD=<app-password> in .env.
"""

from __future__ import annotations

import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class EmailService:
    def _send(self, msg: MIMEMultipart, recipients: list[str]) -> None:
        """Open SMTP connection, send, close. Raises on failure."""
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            if settings.smtp_user and settings.smtp_password:
                smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.sendmail(settings.email_sender, recipients, msg.as_string())

    def send_newsletter(
        self,
        html_content: str,
        subject: str = "Your AI/ML Weekly Digest",
        image_paths: list[str] | None = None,
    ) -> None:
        """Send the newsletter to all configured recipients."""
        recipients = settings.email_recipients
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = settings.email_sender
        msg["To"] = ", ".join(recipients)

        msg.attach(MIMEText(html_content, "html", "utf-8"))

        if image_paths:
            for path_str in image_paths:
                path = Path(path_str)
                if path.exists():
                    with open(path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f'attachment; filename="{path.name}"',
                    )
                    msg.attach(part)

        try:
            self._send(msg, recipients)
            logger.info("newsletter_sent", recipients=len(recipients))
        except Exception as e:
            logger.error("email_send_error", error=str(e))
            raise

    def send_approval_email(
            self,
            run_id: str,
            linkedin_preview: str,
            approve_url: str,
            reject_url: str,
            image_paths: list[str] | None = None,
    ) -> None:
        recipients = settings.email_recipients[:1]

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                    max-width: 600px; margin: 0 auto;">
            <h2>AI Pipeline - Review Required</h2>
            <p>Run <code>{run_id[:8]}</code> has generated content ready for publishing.</p>

            <h3>LinkedIn Post Preview:</h3>
            <div style="background: #f5f5f5; padding: 16px; border-radius: 8px;
                        white-space: pre-wrap;">
                {linkedin_preview}
            </div>

            <div style="margin-top: 24px; text-align: center;">
                <a href="{approve_url}"
                   style="background: #0a66c2; color: white; padding: 12px 32px;
                          border-radius: 6px; text-decoration: none; margin-right: 12px;">
                    Approve &amp; Publish
                </a>
                <a href="{reject_url}"
                   style="background: #dc3545; color: white; padding: 12px 32px;
                          border-radius: 6px; text-decoration: none;">
                    Reject &amp; Revise
                </a>
            </div>
        </div>
        """

        # "mixed" is required to allow file attachments
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Approve AI Content - Run {run_id[:8]}"
        msg["From"] = settings.email_sender
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(html, "html", "utf-8"))

        # THIS ATTACHES THE IMAGE FILE TO THE EMAIL
        if image_paths:
            for path_str in image_paths:
                path = Path(path_str)
                if path.exists():
                    with open(path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f'attachment; filename="{path.name}"',
                    )
                    msg.attach(part)

        try:
            self._send(msg, recipients)
            logger.info("approval_email_sent", run_id=run_id)
        except Exception as e:
            logger.error("approval_email_error", run_id=run_id, error=str(e))
            raise
