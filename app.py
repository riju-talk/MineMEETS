import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Debug: Print environment variables (masked where appropriate)
print("Environment variables loaded:")
print(f"PINECONE_API_KEY: {'*' * len(os.getenv('PINECONE_API_KEY', '')) if os.getenv('PINECONE_API_KEY') else 'Not set'}")
p#rint(f"GROQ_API_KEY: {'*' * len(os.getenv('GROQ_API_KEY', '')) if os.getenv('GROQ_API_KEY') else 'Not set'}")

# Import coordinator (your async orchestrator)
from agents.coordinator import coordinator

# -------- Streamlit page config --------
st.set_page_config(
    page_title="MineMEETS - Meeting Assistant",
    layout="wide"
)

# -------- Styles --------
st.markdown("""
    <style>
    .meeting-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .speaker-0 { color: #1f77b4; font-weight: bold; }
    .speaker-1 { color: #ff7f0e; font-weight: bold; }
    .speaker-2 { color: #2ca02c; font-weight: bold; }
    .speaker-3 { color: #d62728; font-weight: bold; }
    .speaker-4 { color: #9467bd; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# -------- Helpers --------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def make_meeting_id() -> str:
    return f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def run_async(coro):
    """Run an async coroutine from Streamlit safely."""
    return asyncio.run(coro)


# -------- App class --------
class MeetingAssistantApp:
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        if 'current_meeting' not in st.session_state:
            st.session_state.current_meeting = None
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = {}
        if 'last_uploaded_path' not in st.session_state:
            st.session_state.last_uploaded_path = None
        if 'last_uploaded_type' not in st.session_state:
            st.session_state.last_uploaded_type = None
    
    def process_meeting_sync(self, file_path: str, file_type: str) -> bool:
        """Sync wrapper to call async coordinator.process_meeting."""
        meeting_id = make_meeting_id()
        meeting_data = {
            'id': meeting_id,
            'title': Path(file_path).stem,
            'date': datetime.now().isoformat(),
            'participants': [],  # Optional: collect from UI
            'content_path': file_path,
            'type': file_type  # 'transcript' | 'audio' | 'video'
        }
        result = run_async(coordinator.process_meeting(meeting_data))
        if result.get('status') == 'processed':
            st.session_state.current_meeting = meeting_id
            st.session_state.qa_history[meeting_id] = []
            return True
        return False
    
    def ask_question_sync(self, question: str) -> Dict[str, Any]:
        """Sync wrapper to call async coordinator.ask_question."""
        meeting_id = st.session_state.current_meeting
        response = run_async(coordinator.ask_question(question=question, meeting_id=meeting_id))
        return response
    
    def render_sidebar(self):
        st.sidebar.title("Meetings")
        st.sidebar.caption("Upload a meeting file: transcript (.txt), audio (.mp3, .wav, etc), video (transcribed), or screenshots/images (.png, .jpg, .webp, .bmp).")

        uploaded_file = st.sidebar.file_uploader(
            "Upload Meeting File",
            type=["txt", "mp3", "wav", "m4a", "ogg", "flac", "mp4", "m4v", "webm", "mpga", "mpeg", "png", "jpg", "jpeg", "webp", "bmp"],
            help="Supported: .txt, .mp3, .wav, .m4a, .ogg, .flac, .mp4, .m4v, .webm, .mpga, .mpeg, .png, .jpg, .jpeg, .webp, .bmp"
        )

        if uploaded_file is not None:
            # Determine file type
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix == '.txt':
                file_type = 'transcript'
            elif suffix in ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mpga', '.mpeg']:
                file_type = 'audio'
            elif suffix in ['.mp4', '.m4v', '.webm']:
                file_type = 'video'
            elif suffix in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                file_type = 'image'
            else:
                st.sidebar.error(f"Unsupported file type: {suffix}")
                return

            meeting_id = make_meeting_id()
            dest = RAW_DIR / f"{meeting_id}_{uploaded_file.name}"
            with open(dest, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.last_uploaded_path = str(dest)
            st.session_state.last_uploaded_type = file_type
            st.sidebar.success(f"Saved to {dest} ({file_type})")

            if st.sidebar.button("Process Meeting", use_container_width=True):
                ok = self.process_meeting_sync(
                    file_path=st.session_state.last_uploaded_path,
                    file_type=st.session_state.last_uploaded_type
                )
                if ok:
                    st.sidebar.success("Meeting processed!")
                else:
                    st.sidebar.error("Failed to process meeting. Check logs.")

        # List meetings
        meetings = coordinator.list_meetings()
        if meetings:
            st.sidebar.subheader("Your Meetings")
            for m in meetings:
                if st.sidebar.button(f"📄 {m['title']}", key=f"meeting_{m['id']}", use_container_width=True):
                    st.session_state.current_meeting = m["id"]
    
    def render_main(self):
        if not st.session_state.current_meeting:
            self._render_welcome()
            return

        meeting_id = st.session_state.current_meeting
        meeting = coordinator.get_meeting(meeting_id)
        if not meeting:
            st.error("Meeting not found.")
            return

        st.title(meeting.get('title', 'Untitled Meeting'))
        st.caption(f"Date: {meeting.get('date', 'N/A')}")

        tab1, tab2, tab3 = st.tabs(["📄 Content", "🔍 Insights", "❓ Q&A"])
        with tab1:
            self._render_meeting_content(meeting)
        with tab2:
            self._render_insights(meeting)
        with tab3:
            self._render_qa(meeting_id)
    
    def _render_welcome(self):
        st.markdown("""
        # 🤖 Welcome to MineMEETS
        Your intelligent meeting assistant that helps you:
        - 📝 Ingest text transcripts and transcribe audio/video
        - 🖼️ Embed screenshots/images for retrieval (CLIP ViT-B/32)
        - 🔍 Extract key insights and action items
        - ❓ Get answers to your questions
        - 📧 Share insights via email

        ### Get Started
        1. Upload a meeting file using the sidebar
        2. View and analyze the content
        3. Ask questions about the meeting

        **Supported formats:** TXT, MP3/WAV/M4A/OGG/FLAC (transcribed), MP4/M4V/WEBM (transcribed), PNG/JPG/JPEG/WEBP/BMP (embedded)
        """)
    
    def _render_meeting_content(self, meeting: Dict[str, Any]):
        st.subheader("Meeting Transcript")
        transcript = meeting.get('transcript', 'No transcript available.')
        st.code(transcript, language='text')

        st.markdown("---")
        st.subheader("Share Insights")
        email = st.text_input("Recipient email", key="insights_email")
        if st.button("📧 Email Insights", disabled=not bool(email)):
            with st.spinner("Sending email..."):
                result = run_async(coordinator.send_insights_email(
                    meeting_id=meeting['id'],
                    recipient_emails=[email],
                    additional_notes="Here are the insights from our meeting."
                ))
                # coordinator returns email_resp.content (string on success). Handle both shapes.
                if isinstance(result, dict) and result.get('error'):
                    st.error(f"Failed to send email: {result.get('error')}")
                else:
                    st.success(f"Email sent: {result}")
    
    def _render_insights(self, meeting: Dict[str, Any]):
        st.subheader("Meeting Insights")
        insights = meeting.get('insights', {})

        st.markdown("### 📝 Summary")
        st.markdown(insights.get('summary', 'No summary available.'))

        st.markdown("### 🔑 Key Points")
        key_points = insights.get('key_points', [])
        if isinstance(key_points, str):
            st.markdown(key_points or "No key points available.")
        else:
            if not key_points:
                st.markdown("No key points available.")
            for point in key_points:
                st.markdown(f"- {point}")

        st.markdown("### ✅ Action Items")
        action_items = insights.get('action_items', [])
        if isinstance(action_items, str):
            st.markdown(action_items or "No action items found.")
        else:
            if not action_items:
                st.markdown("No action items found.")
            for item in action_items:
                st.markdown(f"- {item if isinstance(item, str) else str(item)}")
    
    def _render_qa(self, meeting_id: str):
        st.subheader("Ask a Question")
        question = st.text_input("Ask something about this meeting:", key=f"question_{meeting_id}")
        if st.button("Ask"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    response = self.ask_question_sync(question)
                    st.markdown(f"**Answer:** {response.get('answer', 'No answer found.')}")

                    # Optional: show sources if present
                    sources = response.get("sources", [])
                    if sources:
                        st.markdown("**Sources:**")
                        for s in sources:
                            title = s.get("title") or s.get("metadata", {}).get("source") or "source"
                            url = s.get("url") or s.get("metadata", {}).get("url") or ""
                            st.markdown(f"- {title}{f' — {url}' if url else ''}")

                    # Keep chat history
                    st.session_state.qa_history.setdefault(meeting_id, [])
                    st.session_state.qa_history[meeting_id].append({"role": "user", "content": question})
                    st.session_state.qa_history[meeting_id].append({"role": "assistant", "content": response.get('answer', '')})

        # Conversation history
        history = st.session_state.qa_history.get(meeting_id, [])
        if history:
            st.markdown("---")
            st.subheader("Conversation History")
            for msg in history:
                who = "You" if msg["role"] == "user" else "Assistant"
                st.markdown(f"**{who}:** {msg['content']}")
                st.markdown("---")


def main():
    if not os.getenv("PINECONE_API_KEY"):
        st.error("PINECONE_API_KEY not set. Add it to your .env file.")
        return
    app = MeetingAssistantApp()
    app.render_sidebar()
    app.render_main()

if __name__ == "__main__":
    load_dotenv()
    main()
