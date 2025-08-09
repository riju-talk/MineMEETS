"""Email Agent for sending meeting insights and summaries."""
from typing import Dict, Any, List, Optional, Union
from .base_agent import BaseAgent, AgentResponse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os
from datetime import datetime
import json

class EmailAgent(BaseAgent):
    """Agent for handling email communications of meeting insights."""
    
    def __init__(self, smtp_server: str = None, smtp_port: int = 587):
        """Initialize the Email agent.
        
        Args:
            smtp_server: SMTP server address (default: from env)
            smtp_port: SMTP server port (default: 587)
        """
        super().__init__(
            name="email_agent",
            description="Handles sending meeting insights and summaries via email"
        )
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("EMAIL_ADDRESS")
        self.sender_password = os.getenv("EMAIL_PASSWORD")
    
    async def process(self, email_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process and send an email.
        
        Args:
            email_data: Dictionary containing email details:
                - to: Recipient email(s) - string or list of strings
                - subject: Email subject
                - body: Email body content
                - is_html: Whether the body is HTML (default: False)
                - attachments: List of file paths to attach
            context: Additional context for the email
            
        Returns:
            AgentResponse with the result of the email sending
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(email_data['to']) if isinstance(email_data['to'], list) else email_data['to']
            msg['Subject'] = email_data.get('subject', 'Meeting Insights from MineMEETS')
            
            # Add body
            body = email_data.get('body', '')
            if email_data.get('is_html', False):
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            # Add context as JSON attachment if needed
            if context:
                context_str = json.dumps(context, indent=2)
                part = MIMEText(context_str, 'plain')
                part.add_header('Content-Disposition', 'attachment', filename='meeting_context.json')
                msg.attach(part)
            
            # Send the email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return AgentResponse(
                success=True,
                content=f"Email sent successfully to {msg['To']}",
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "recipient": msg['To'],
                    "subject": msg['Subject']
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                content=f"Failed to send email: {str(e)}",
                metadata={
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def send_meeting_insights(
        self,
        to: Union[str, List[str]],
        meeting_data: Dict[str, Any],
        insights: Dict[str, Any],
        additional_notes: str = ""
    ) -> AgentResponse:
        """Send meeting insights via email.
        
        Args:
            to: Recipient email(s)
            meeting_data: Meeting metadata (title, date, participants, etc.)
            insights: Dictionary containing meeting insights
            additional_notes: Any additional notes to include
            
        Returns:
            AgentResponse with the result of the email sending
        """
        # Format the email body
        subject = f"Meeting Insights: {meeting_data.get('title', 'Untitled Meeting')}"
        
        # Create HTML email body
        html = f"""
        <html>
            <body>
                <h2>Meeting Insights: {title}</h2>
                <p><strong>Date:</strong> {date}</p>
                <p><strong>Participants:</strong> {participants}</p>
                
                <h3>Summary</h3>
                <p>{summary}</p>
                
                <h3>Key Points</h3>
                <ul>
        """.format(
            title=meeting_data.get('title', 'Untitled Meeting'),
            date=meeting_data.get('date', 'N/A'),
            participants=", ".join(meeting_data.get('participants', ['N/A'])),
            summary=insights.get('summary', 'No summary available.')
        )
        
        # Add key points
        for point in insights.get('key_points', []):
            html += f"<li>{point}</li>"
        
        # Add action items if available
        if 'action_items' in insights and insights['action_items']:
            html += """
                </ul>
                
                <h3>Action Items</h3>
                <ul>
            """
            for item in insights['action_items']:
                html += f"<li>{item}</li>"
        
        # Close HTML
        html += """
                </ul>
                
                <h3>Additional Notes</h3>
                <p>{notes}</p>
                
                <p>--<br>This email was automatically generated by MineMEETS</p>
            </body>
        </html>
        """.format(notes=additional_notes)
        
        # Send the email
        return await self.process({
            'to': to,
            'subject': subject,
            'body': html,
            'is_html': True
        }, {
            'meeting_data': meeting_data,
            'insights': insights
        })
