from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import asyncio
import os
import whisper
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentResponse


class AudioTranscription(BaseModel):
    """Structured output for audio transcription."""
    text: str = Field(..., description="The transcribed text")
    language: str = Field(..., description="Detected language code")
    duration: float = Field(..., description="Audio duration in seconds")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Transcription chunks with timestamps")


class AudioAgent(BaseAgent):
    """Agent for transcribing audio files using Whisper."""

    SUPPORTED_FORMATS = {
        '.mp3', '.wav', '.m4a', '.ogg', '.flac',
        '.mp4', '.m4v', '.webm', '.mpga', '.mpeg'
    }
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def __init__(self, model_size: str = "base"):
        super().__init__(
            name="audio_agent",
            description="Transcribes meeting audio files into text chunks using Whisper."
        )
        self.model_size = model_size
        self._model_lock = asyncio.Lock()  # Prevent concurrent model loading
        self._model = None

    @property
    def model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call setup() first.")
        return self._model

    async def setup(self) -> None:
        """Load Whisper model asynchronously with error handling."""
        if self._model is not None:
            return

        async with self._model_lock:
            if self._model is not None:  # Check again in case another task loaded it
                return
            try:
                self.logger.info(f"Loading Whisper model: {self.model_size}")
                self._model = await asyncio.to_thread(
                    whisper.load_model,
                    self.model_size,
                    download_root=os.getenv("WHISPER_CACHE_DIR")
                )
                self.logger.info("Whisper model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {str(e)}")
                raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Transcribe audio file to text using Whisper.
        
        Args:
            input_data: Must contain 'file_path' key with path to audio file
            context: Optional context with 'meeting_id' and other metadata
            
        Returns:
            AgentResponse with transcription result or error
        """
        try:
            # Validate input
            if not input_data or 'file_path' not in input_data:
                return self.failure_response("Missing 'file_path' in input data")
                
            file_path = input_data["file_path"]
            
            # Validate file
            try:
                file_path = self._validate_audio_file(file_path)
            except ValueError as e:
                return self.failure_response(str(e))
            except Exception as e:
                self.logger.error(f"File validation error: {str(e)}", exc_info=True)
                return self.failure_response(f"Invalid audio file: {str(e)}")

            # Load model if not already loaded
            try:
                await self.setup()
            except Exception as e:
                self.logger.error(f"Model setup failed: {str(e)}", exc_info=True)
                return self.failure_response("Failed to initialize transcription model")

            # Transcribe with timeout
            try:
                self.logger.info(f"Starting transcription for {file_path}")
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.model.transcribe,
                        str(file_path),
                        verbose=False,
                        language='en',  # Force English for now
                        fp16=False  # Disable FP16 to avoid CUDA issues
                    ),
                    timeout=3600  # 1 hour timeout for long recordings
                )
                
                # Process result
                transcription = AudioTranscription(
                    text=result.get("text", "").strip(),
                    language=result.get("language", "en"),
                    duration=result.get("duration", 0.0),
                    chunks=result.get("segments", [])
                )
                
                self.logger.info(f"Successfully transcribed {file_path} ({len(transcription.text)} chars)")
                return self.success_response(transcription.dict())
                
            except asyncio.TimeoutError:
                return self.failure_response("Transcription timed out")
                
            except Exception as e:
                self.logger.error(f"Transcription failed: {str(e)}", exc_info=True)
                return self.failure_response(f"Transcription failed: {str(e)}")

        except Exception as e:
            self.logger.error(f"Unexpected error in process: {str(e)}", exc_info=True)
            return self.failure_response(f"Audio processing error: {str(e)}")
    
    def _validate_audio_file(self, file_path: Union[str, Path]) -> Path:
        """Validate audio file exists and is in supported format."""
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                raise ValueError(f"File not found: {file_path}")
                
            if not path.is_file():
                raise ValueError(f"Not a file: {file_path}")
                
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported file format: {path.suffix}. "
                    f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
                )
                
            file_size = path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                raise ValueError(
                    f"File too large: {file_size/1024/1024:.1f}MB "
                    f"(max {self.MAX_FILE_SIZE/1024/1024}MB)"
                )
                
            return path
            
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid audio file: {str(e)}")
