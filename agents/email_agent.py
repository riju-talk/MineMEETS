# agents/email_agent.py
from typing import Dict, Any, List, Optional, Union
from .base_agent import BaseAgent, AgentResponse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
from datetime import datetime
import json
import logging

class EmailAgent(BaseAgent):
    """Agent for handling email communications of meeting insights."""

    def __init__(self, smtp_server: str = None, smtp_port: int = 587):
        super().__init__(name="email_agent", description="Handles sending meeting insights and summaries via email")
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("EMAIL_ADDRESS")
        self.sender_password = os.getenv("EMAIL_PASSWORD")

        if not self.sender_email or not self.sender_password:
            self.logger.warning("EMAIL_ADDRESS or EMAIL_PASSWORD not set. EmailAgent will fail to login without credentials.")

    async def process(self, email_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Send an email.
        email_data keys:
          - to: str | List[str]
          - subject: str
          - body: str
          - is_html: bool
          - attachments: list of file paths (not implemented here)
        """
        try:
            # Validate recipients
            if "to" not in email_data or not email_data["to"]:
                return AgentResponse(success=False, content="Missing 'to' in email_data", metadata={})

            recipients = email_data["to"]
            if isinstance(recipients, list):
                recipients_joined = ", ".join(recipients)
            else:
                recipients_joined = recipients

            msg = MIMEMultipart()
            msg["From"] = self.sender_email or "no-reply@minemeets.local"
            msg["To"] = recipients_joined
            msg["Subject"] = email_data.get("subject", "Meeting Insights from MineMEETS")

            body = email_data.get("body", "")
            if email_data.get("is_html", False):
                msg.attach(MIMEText(body, "html"))
            else:
                msg.attach(MIMEText(body, "plain"))

            # Attach context JSON if required
            if context:
                context_str = json.dumps(context, indent=2)
                part = MIMEText(context_str, "plain")
                part.add_header("Content-Disposition", "attachment", filename="meeting_context.json")
                msg.attach(part)

            # Send over SMTP (with TLS)
            if not self.sender_email or not self.sender_password:
                raise RuntimeError("Email credentials not configured (EMAIL_ADDRESS/EMAIL_PASSWORD missing).")

            # Use blocking SMTP inside a thread if called from async code
            def send_email():
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.ehlo()
                    server.starttls()
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)

            # run in thread to avoid blocking event loop
            import asyncio
            await asyncio.to_thread(send_email)

            return AgentResponse(
                success=True,
                content=f"Email sent successfully to {recipients_joined}",
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "recipient": recipients_joined,
                    "subject": msg["Subject"]
                }
            )

        except Exception as e:
            self.logger.exception("Failed to send email")
            return AgentResponse(
                success=False,
                content=f"Failed to send email: {str(e)}",
                metadata={"error": str(e), "timestamp": datetime.utcnow().isoformat()}
            )

    async def send_meeting_insights(
        self,
        to: Union[str, List[str]],
        meeting_data: Dict[str, Any],
        insights: Dict[str, Any],
        additional_notes: str = ""
    ) -> AgentResponse:
        """Format meeting insights into HTML and send the email via process()."""
        title = meeting_data.get("title", "Untitled Meeting")
        date = meeting_data.get("date", "N/A")
        participants = meeting_data.get("participants", [])

        # Build HTML body (safe f-string formatting)
        key_points = insights.get("key_points", [])
        action_items = insights.get("action_items", [])

        html_body = f"""
        <html>
          <body>
            <h2>Meeting Insights: {title}</h2>
            <p><strong>Date:</strong> {date}</p>
            <p><strong>Participants:</strong> {', '.join(participants) if participants else 'N/A'}</p>
            <h3>Summary</h3>
            <p>{insights.get('summary','No summary available.')}</p>
            <h3>Key Points</h3>
            <ul>
        """
        for p in key_points:
            html_body += f"<li>{p}</li>"

        html_body += "</ul>"

        if action_items:
            html_body += "<h3>Action Items</h3><ul>"
            for item in action_items:
                # action item might be a dict or string
                if isinstance(item, dict):
                    text = item.get("text") or str(item)
                else:
                    text = str(item)
                html_body += f"<li>{text}</li>"
            html_body += "</ul>"

        html_body += f"""
            <h3>Additional Notes</h3><p>{additional_notes}</p>
            <p>--<br>This email was automatically generated by MineMEETS</p>
          </body>
        </html>
        """

        return await self.process({
            "to": to,
            "subject": f"Meeting Insights: {title}",
            "body": html_body,
            "is_html": True
        }, {"meeting_data": meeting_data, "insights": insights})
