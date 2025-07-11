import whisper
from whisper.utils import get_writer
import numpy as np

class WhisperTranscriber:
    def __init__(self, model_path="models/ggml-base.bin"):
        self.model = whisper.load_model(model_path)  # whisper.cpp binding
        
    def transcribe(self, audio_path):
        result = self.model.transcribe(
            audio_path,
            language="en",
            word_timestamps=True,
            vad_filter=True,  # Voice Activity Detection
            vad_threshold=0.5
        )
        return result["segments"]  # Returns timestamped chunks