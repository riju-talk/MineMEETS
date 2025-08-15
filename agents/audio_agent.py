# agents/audio_agent.py

from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentResponse
import os
import whisper
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter


class AudioAgent(BaseAgent):
    """Agent for transcribing audio into text chunks using Whisper."""

    def __init__(self, model_size: str = "base"):
        super().__init__(
            name="audio_agent",
            description="Transcribes meeting audio files into text chunks using Whisper."
        )
        self.model = whisper.load_model(model_size)

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        try:
            file_path = input_data.get("file_path", "")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            # Transcribe audio
            result = self.model.transcribe(file_path)
            transcript_text = result["text"].strip()

            # Split transcript into chunks for vector storage
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = splitter.split_text(transcript_text)

            chunk_docs = [
                {
                    "text": chunk,
                    "metadata": {
                        "session_id": str(uuid.uuid4()),
                        "chunk_index": i,
                        "meeting_id": context.get("meeting_id") if context else None
                    }
                }
                for i, chunk in enumerate(chunks)
            ]

            return AgentResponse(
                success=True,
                content={
                    "transcript": transcript_text,
                    "chunks": chunk_docs
                },
                metadata={
                    "audio_length": result.get("duration", "unknown"),
                    "source": "audio_agent",
                    **(context or {})
                }
            )

        except Exception as e:
            return AgentResponse(
                success=False,
                content=f"Failed to transcribe audio: {str(e)}",
                metadata={"error": str(e)}
            )
