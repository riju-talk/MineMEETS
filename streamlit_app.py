# streamlit_app.py
import streamlit as st
from transcription_stream import TranscriptionStream
from backend.transcribe import WhisperTranscriber
from backend.embedder import MeetingEmbedder
from backend.rag_agent import MeetingAnalyzer
from backend.task_parser import TaskExtractor
import os

def main():
    st.title("MineMEETs 🎙️")
    
    # Sidebar for configuration
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Meeting Audio", type=["mp3", "wav"])
        recipients = st.text_input("Email Recipients (comma-separated)")
    
    # Main content area
    transcript_col, summary_col = st.columns(2)
    
    if uploaded_file:
        # Initialize components
        transcriber = WhisperTranscriber()
        embedder = MeetingEmbedder()
        analyzer = MeetingAnalyzer(os.getenv("GROQ_API_KEY"), embedder.index)
        extractor = TaskExtractor(analyzer)
        
        # Process file
        segments = transcriber.transcribe(uploaded_file)
        embedder.add_transcript(segments)
        
        # Display live transcript
        for segment in segments:
            transcript_col.markdown(f"[{segment['start']:.2f}s] {segment['text']}")
            
        # Generate and display summary
        summary = analyzer.query("Generate a meeting summary")
        summary_col.markdown(f"### Summary\n{summary}")
        
        # Extract and display tasks
        tasks = extractor.extract_tasks()
        st.json(tasks)
        
        # Email functionality
        if recipients and st.button("Send Email"):
            send_summary_email(recipients, tasks, segments)

if __name__ == "__main__":
    main()