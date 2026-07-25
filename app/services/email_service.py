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
from app.services.image_service import ImageService

logger = get_logger(__name__)
settings = get_settings()


def _build_slide_thumbnail_grid(
    slide_image_paths: list[str],
    approve_url: str,
    reject_url: str,
) -> str:
    """
    Build the fast-review thumbnail grid + prominent approve/reject block.

    Design decision (per approval-model audit): the existing DB/API layer
    (``ApprovalStatus`` on ``LinkedInPostModel``, and the ``/approvals/{run_id}``
    + ``/approvals/via-token`` routes) only models a *run-level* approval
    decision — there is no per-slide approval row or endpoint. Rather than
    inventing a parallel per-slide data model, each thumbnail here links only
    to its own full-resolution image (opens in a new tab) so the reviewer can
    zoom in on a specific slide. The actual approve/reject *decision* is a
    single run-level action, surfaced here as one prominent button pair at
    the top of the email so a reviewer can scan the thumbnails and decide in
    ~20 seconds, instead of scrolling through a full article dump.
    """
    if not slide_image_paths:
        return ""

    image_service = ImageService(settings.app_base_url)
    slide_urls = image_service.get_all_card_urls(slide_image_paths)
    if not slide_urls:
        return ""

    thumbnails = "".join(
        f"""
        <a href="{url}" target="_blank" rel="noopener"
           style="display:inline-block;margin:6px;text-decoration:none">
            <img src="{url}" alt="Slide {i}" width="140"
                 style="width:140px;height:140px;object-fit:cover;border-radius:8px;
                        border:1px solid #e0e0e0;display:block" />
        </a>"""
        for i, url in enumerate(slide_urls, start=1)
    )

    return f"""
    <div style="text-align:center;margin-bottom:24px;padding:20px 12px;
                background:#eef6ff;border-radius:10px;border:2px solid #0a66c2">
        <p style="margin:0 0 14px;font-size:14px;color:#333">
            {len(slide_urls)} carousel slides generated — scan below, then decide:
        </p>
        <a href="{approve_url}"
           style="background:#0a66c2;color:white;padding:16px 40px;border-radius:8px;
                  text-decoration:none;font-size:16px;font-weight:700;margin-right:12px;
                  display:inline-block">
            Approve &amp; Publish
        </a>
        <a href="{reject_url}"
           style="background:#dc3545;color:white;padding:16px 40px;border-radius:8px;
                  text-decoration:none;font-size:16px;font-weight:700;display:inline-block">
            Reject &amp; Revise
        </a>
    </div>

    <h3 style="margin-bottom:8px">Slide Preview</h3>
    <div style="text-align:center;margin-bottom:12px">
        {thumbnails}
    </div>
    """


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
            research_article_html: str = "",
            slide_image_paths: list[str] | None = None,
    ) -> None:
        """
        Send the human-review email for a pipeline run awaiting approval.

        ``slide_image_paths`` is optional and additive: when omitted (the
        news pipeline's call site never sets it), the email renders exactly
        as before — a LinkedIn preview, optional article dump, and a single
        approve/reject button pair. When present (research pipeline runs
        that generated carousel slides), a fast-review thumbnail grid plus a
        prominent top-of-email approve/reject block is prepended. See
        ``_build_slide_thumbnail_grid`` for why this stays run-level rather
        than per-slide.
        """
        recipients = settings.email_recipients[:1]

        slide_grid_section = _build_slide_thumbnail_grid(
            slide_image_paths or [], approve_url, reject_url
        )

        article_section = (
            f"""
            <details open style="margin-top:28px">
              <summary style="cursor:pointer;font-size:15px;font-weight:600;color:#0a66c2;
                              padding:10px 0;border-top:1px solid #e0e0e0">
                📄 Full Research Article Preview (click to collapse)
              </summary>
              <div style="border:1px solid #e8eaf0;border-radius:10px;padding:8px;
                          margin-top:12px;max-height:700px;overflow-y:auto;background:#fafbff">
                {research_article_html}
              </div>
            </details>
            """
            if research_article_html else ""
        )

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                    max-width: 700px; margin: 0 auto;">
            <h2>AI Pipeline - Review Required</h2>
            <p>Run <code>{run_id[:8]}</code> has generated content ready for publishing.</p>

            {slide_grid_section}

            <h3>LinkedIn Post Preview:</h3>
            <div style="background: #f5f5f5; padding: 16px; border-radius: 8px;
                        white-space: pre-wrap; font-size: 14px; line-height: 1.6;">
                {linkedin_preview}
            </div>

            {article_section}

            <div style="margin-top: 28px; text-align: center;">
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
                        if path.suffix.lower() == ".pdf":
                            part = MIMEBase("application", "pdf")
                        else:
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
