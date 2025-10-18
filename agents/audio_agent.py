from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import asyncio
import os
import whisper

class AudioTranscription:
    """Structured output for audio transcription."""
    def __init__(self, text: str, language: str, duration: float, chunks: List[Dict[str, Any]]):
        self.text = text
        self.language = language
        self.duration = duration
        self.chunks = chunks

class AudioAgent:
    """Agent for transcribing audio files using Whisper."""

    SUPPORTED_FORMATS = {
        '.mp3', '.wav', '.m4a', '.ogg', '.flac'
    }
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model_lock = asyncio.Lock()  # Prevent concurrent model loading
        self._model = None

    async def setup(self) -> None:
        """Load Whisper model asynchronously with error handling."""
        if self._model is not None:
            return

        async with self._model_lock:
            if self._model is not None:  # Check again in case another task loaded it
                return
            try:
                self._model = await asyncio.to_thread(
                    whisper.load_model,
                    self.model_size,
                    download_root=os.getenv("WHISPER_CACHE_DIR")
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Transcribe audio file to text using Whisper."""
        try:
            # Validate input
            if not input_data or 'file_path' not in input_data:
                return {"success": False, "content": "Missing 'file_path' in input data"}

            file_path = input_data["file_path"]

            # Validate file
            try:
                file_path = self._validate_audio_file(file_path)
            except ValueError as e:
                return {"success": False, "content": str(e)}

            # Load model if not already loaded
            try:
                await self.setup()
            except Exception as e:
                return {"success": False, "content": "Failed to initialize transcription model"}

            # Transcribe with timeout
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._model.transcribe,
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

                return {"success": True, "content": transcription.__dict__}

            except asyncio.TimeoutError:
                return {"success": False, "content": "Transcription timed out"}

            except Exception as e:
                return {"success": False, "content": f"Transcription failed: {str(e)}"}

        except Exception as e:
            return {"success": False, "content": f"Audio processing error: {str(e)}"}

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
