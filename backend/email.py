import smtplib
from email.message import EmailMessage
import os

def send_email(recipients, transcript):
    msg = EmailMessage()
    msg['Subject'] = "MinMEETs – Your Meeting Transcript"
    msg['From'] = os.getenv("SMTP_USER")
    msg['To'] = ", ".join(recipients)
    msg.set_content(transcript)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
        smtp.send_message(msg)
t