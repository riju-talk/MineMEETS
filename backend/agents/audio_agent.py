# agents/audio_agent.py

from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentResponse
import os
import tempfile
import whisper
import uuid

class AudioAgent(BaseAgent):
    """Agent for transcribing audio using Whisper."""

    def __init__(self, model_size: str = "base"):
        """Initialize the audio transcription agent.

        Args:
            model_size: Size of the Whisper model (e.g., 'tiny', 'base', 'small', 'medium', 'large')
        """
        super().__init__(
            name="audio_agent",
            description="Transcribes meeting audio files using Whisper."
        )
        self.model = whisper.load_model(model_size)

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Transcribe an audio file.

        Args:
            input_data: Dictionary with a key "file_path" pointing to the audio file
            context: Additional context (e.g. meeting_id)

        Returns:
            AgentResponse with the transcribed text
        """
        try:
            file_path = input_data.get("file_path", "")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            result = self.model.transcribe(file_path)

            return AgentResponse(
                success=True,
                content={"transcript": result["text"].strip()},
                metadata={
                    "audio_length": result.get("duration", "unknown"),
                    "source": "audio_agent",
                    "session_id": str(uuid.uuid4()),
                    **(context or {})
                }
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                content=f"Failed to transcribe audio: {str(e)}",
                metadata={"error": str(e)}
            )
