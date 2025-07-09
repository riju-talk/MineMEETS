import whisper
def transcribe_audio(file_obj):
    model = whisper.load_model("base")
    result = model.transcribe(file_obj)
    return result["text"]
