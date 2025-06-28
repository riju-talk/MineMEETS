from fastapi import FastAPI, File, UploadFile, Form
from backend.transcribe import transcribe_audio
from backend.embedder import embed_transcript, store_to_faiss
from backend.rag_agent import query_with_rag
from backend.emailer import send_email
from backend.task_parser import extract_tasks
import os

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...), recipients: str = Form(...)):
    transcript = transcribe_audio(file.file)
    chunks = embed_transcript(transcript)
    store_to_faiss(chunks)
    send_email(recipients.split(","), transcript)
    return {"message": "Transcription done and mailed!"}

@app.get("/query")
def query(q: str):
    return query_with_rag(q)

@app.get("/tasks")
def get_tasks():
    return extract_tasks()
