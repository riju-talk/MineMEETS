# email.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_summary_email(recipients, tasks, transcript):
    msg = MIMEMultipart()
    msg['Subject'] = "Meeting Summary from MineMEETs"
    msg['From'] = os.getenv("SMTP_USER")
    
    # Create HTML content
    html = """
    <h2>Meeting Summary</h2>
    <h3>Tasks:</h3>
    <ul>
    {tasks}
    </ul>
    <h3>Transcript:</h3>
    <pre>{transcript}</pre>
    """.format(
        tasks="\n".join([f"<li>{t['task']} (Owner: {t['owner']}, Deadline: {t['deadline']})</li>" for t in tasks]),
        transcript="\n".join([f"[{seg['start']:.2f}s] {seg['text']}" for seg in transcript])
    )
    
    part = MIMEText(html, "html")
    msg.attach(part)
    
    # Send via SMTP
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
        for recipient in recipients.split(","):
            msg["To"] = recipient.strip()
            server.sendmail(msg["From"], msg["To"], msg.as_string())