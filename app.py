import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import asyncio
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import traceback

# Load environment variables from .env file
load_dotenv(override=True)

# Debug: Print environment variables (masked where appropriate)
print("Environment variables loaded:")
print(f"PINECONE_API_KEY: {'*' * len(os.getenv('PINECONE_API_KEY', '')) if os.getenv('PINECONE_API_KEY') else 'Not set'}")

# Import coordinator (your async orchestrator)
from agents.coordinator import coordinator

# -------- Streamlit page config --------
st.set_page_config(
    page_title="MineMEETS - Meeting Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
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
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .chat-message.assistant {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #4CAF50;
    }
    .status-offline {
        background-color: #f44336;
    }
    </style>
""", unsafe_allow_html=True)


# -------- Helpers --------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def make_meeting_id() -> str:
    return f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

def run_async(coro):
    """Run an async coroutine from Streamlit safely with error handling."""
    try:
        return asyncio.run(coro)
    except Exception as e:
        st.error(f"Async operation failed: {str(e)}")
        st.code(traceback.format_exc())
        return None


# -------- App class --------
class MeetingAssistantApp:
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        default_states = {
            'current_meeting': None,
            'qa_history': {},
            'processing_state': None,
            'processing_message': '',
            'uploaded_files': [],
            'meetings_initialized': False,
            'last_db_check': None,
            'total_vectors': 0
        }
        
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get real-time database statistics."""
        try:
            # Cache for 2 seconds to avoid too frequent calls
            current_time = datetime.now().timestamp()
            if (st.session_state.last_db_check and 
                current_time - st.session_state.last_db_check < 2 and
                st.session_state.total_vectors > 0):
                return {'total_vector_count': st.session_state.total_vectors}
            
            stats = coordinator.pinecone_db.get_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            # Update session state
            st.session_state.total_vectors = total_vectors
            st.session_state.last_db_check = current_time
            
            return stats
        except Exception as e:
            st.error(f"Failed to get database stats: {e}")
            return {'total_vector_count': 0}
    
    def has_embeddings(self) -> bool:
        """Check if database has any embeddings."""
        stats = self.get_database_stats()
        return stats.get('total_vector_count', 0) > 0
    
    def save_uploaded_file(self, uploaded_file, file_type: str) -> Optional[str]:
        """Save uploaded file and return path."""
        try:
            if uploaded_file is None:
                return None
                
            meeting_id = make_meeting_id()
            file_extension = Path(uploaded_file.name).suffix
            dest = RAW_DIR / f"{meeting_id}{file_extension}"
            
            with open(dest, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Store file info
            file_info = {
                'path': str(dest),
                'type': file_type,
                'name': uploaded_file.name,
                'meeting_id': meeting_id,
                'upload_time': datetime.now().isoformat()
            }
            
            # Add to uploaded files list
            st.session_state.uploaded_files.append(file_info)
            
            return str(dest)
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return None
    
    async def process_meeting_async(self, file_path: str, file_type: str, meeting_id: str) -> Dict[str, Any]:
        """Async function to process meeting."""
        try:
            meeting_data = {
                'id': meeting_id,
                'title': Path(file_path).stem,
                'date': datetime.now().isoformat(),
                'participants': [],
                'content_path': file_path,
                'type': file_type
            }
            
            result = await coordinator.process_meeting(meeting_data)
            return result
        except Exception as e:
            return {'status': 'failed', 'error': str(e), 'meeting_id': meeting_id}
    
    def process_meeting_sync(self, file_path: str, file_type: str) -> bool:
        """Sync wrapper to call async coordinator.process_meeting with progress."""
        try:
            meeting_id = make_meeting_id()
            
            # Set processing state
            st.session_state.processing_state = 'processing'
            st.session_state.processing_message = f"Processing {file_type} file..."
            
            # Show progress in main area for better visibility
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Starting processing...")
            progress_bar.progress(10)
            
            # Process the meeting
            result = run_async(self.process_meeting_async(file_path, file_type, meeting_id))
            
            if result is None:
                st.session_state.processing_state = 'error'
                st.session_state.processing_message = "Processing failed - no result returned"
                progress_bar.progress(0)
                status_text.text("❌ Processing failed")
                return False
            
            progress_bar.progress(70)
            status_text.text("📊 Upserting to database...")
            
            if result.get('status') == 'processed':
                st.session_state.current_meeting = meeting_id
                st.session_state.qa_history[meeting_id] = []
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                st.session_state.processing_state = 'success'
                st.session_state.processing_message = "Meeting processed successfully!"
                
                # Force refresh of database stats
                st.session_state.last_db_check = None
                
                # Small delay to show completion
                import time
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                st.rerun()
                return True
            else:
                error_msg = result.get('error', 'Unknown error')
                st.session_state.processing_state = 'error'
                st.session_state.processing_message = f"Processing failed: {error_msg}"
                progress_bar.progress(0)
                status_text.text(f"❌ Error: {error_msg}")
                return False
                
        except Exception as e:
            st.session_state.processing_state = 'error'
            st.session_state.processing_message = f"Processing error: {str(e)}"
            st.error(f"Unexpected error: {str(e)}")
            return False
    
    def ask_question_sync(self, question: str) -> Dict[str, Any]:
        """Sync wrapper to call async coordinator.ask_question."""
        meeting_id = st.session_state.current_meeting
        if not meeting_id:
            return {"answer": "No meeting selected. Please process a meeting first.", "sources": []}
        
        try:
            response = run_async(coordinator.ask_question(question=question, meeting_id=meeting_id))
            return response if response else {"answer": "Error getting response", "sources": []}
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "sources": []}
    
    def flush_database(self) -> bool:
        """Flush the entire database."""
        try:
            coordinator.pinecone_db.delete_vectors(delete_all=True)
            # Clear session state
            st.session_state.total_vectors = 0
            st.session_state.last_db_check = None
            st.session_state.current_meeting = None
            st.session_state.qa_history = {}
            coordinator.active_meetings.clear()
            return True
        except Exception as e:
            st.error(f"Failed to flush database: {e}")
            return False
    
    def render_sidebar(self):
        st.sidebar.title("📊 MineMEETS")
        
        # Database status with real-time indicator
        stats = self.get_database_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            status_class = "status-online" if total_vectors > 0 else "status-offline"
            st.markdown(f'<div class="status-indicator {status_class}"></div>', unsafe_allow_html=True)
        with col2:
            status_text = "Database Online" if total_vectors > 0 else "Database Empty"
            st.sidebar.caption(f"{status_text}")
        
        st.sidebar.metric("Total Vectors", total_vectors)
        
        # Show processing status
        if st.session_state.processing_state:
            if st.session_state.processing_state == 'processing':
                st.sidebar.info(f"🔄 {st.session_state.processing_message}")
            elif st.session_state.processing_state == 'success':
                st.sidebar.success(f"✅ {st.session_state.processing_message}")
            elif st.session_state.processing_state == 'error':
                st.sidebar.error(f"❌ {st.session_state.processing_message}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📁 Upload Content")
        
        # Single file uploader for all types
        uploaded_file = st.sidebar.file_uploader(
            "Choose a file to process",
            type=["mp3", "wav", "m4a", "ogg", "flac", "png", "jpg", "jpeg", "webp", "bmp", "txt", "pdf", "docx"],
            help="Supported: Audio, Images, Documents",
            key="main_uploader"
        )
        
        if uploaded_file:
            # Determine file type
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']:
                file_type = 'audio'
            elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                file_type = 'image'
            else:
                file_type = 'file'
            
            if st.sidebar.button("🚀 Process Meeting", use_container_width=True):
                file_path = self.save_uploaded_file(uploaded_file, file_type)
                if file_path:
                    self.process_meeting_sync(file_path, file_type)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🗄️ Database Management")
        
        if total_vectors > 0:
            if st.sidebar.button("🗑️ Flush Database", use_container_width=True, type="secondary"):
                if st.sidebar.checkbox("I understand this will delete ALL data permanently", key="confirm_flush"):
                    if self.flush_database():
                        st.sidebar.success("Database flushed successfully!")
                        st.rerun()
        else:
            st.sidebar.info("No data to flush")

        # List meetings
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Your Meetings")
        
        meetings = coordinator.list_meetings()
        if meetings:
            for m in meetings:
                meeting_btn = st.sidebar.button(
                    f"📄 {m['title'][:25]}...",
                    key=f"select_{m['id']}",
                    use_container_width=True
                )
                if meeting_btn:
                    st.session_state.current_meeting = m["id"]
                    st.rerun()
        else:
            st.sidebar.info("👆 Upload a file to get started!")
    
    def render_main(self):
        try:
            # Real-time database check
            has_data = self.has_embeddings()
            
            if not has_data:
                self._render_welcome()
                return

            # Check if any meetings exist and select the latest if none is selected
            meetings = coordinator.list_meetings()
            if meetings and not st.session_state.current_meeting:
                latest_meeting = max(meetings, key=lambda m: m.get('date', m['id']))
                st.session_state.current_meeting = latest_meeting['id']

            if not st.session_state.current_meeting and meetings:
                st.session_state.current_meeting = meetings[0]['id']

            if not st.session_state.current_meeting:
                self._render_data_but_no_meeting()
                return

            meeting_id = st.session_state.current_meeting
            meeting = coordinator.get_meeting(meeting_id)
            if not meeting:
                st.error("Meeting not found. Please select another meeting.")
                st.session_state.current_meeting = None
                return

            # Header with meeting info
            col1, col2 = st.columns([3, 1])
            with col1:
                st.title(meeting.get('title', 'Untitled Meeting'))
                st.caption(f"Date: {meeting.get('date', 'N/A')} | ID: {meeting_id}")
            with col2:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()

            # Meeting stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Transcript Length", f"{len(meeting.get('transcript', '')):,} chars")
            with col2:
                st.metric("Chunks Created", len(meeting.get('chunks', [])))
            with col3:
                st.metric("Participants", len(meeting.get('participants', [])))

            # Main content tabs
            tab1, tab2 = st.tabs(["📄 Meeting Content", "💬 Chat with Data"])
            with tab1:
                self._render_meeting_content(meeting)
            with tab2:
                self._render_qa(meeting_id)
                
        except Exception as e:
            st.error(f"Error rendering main content: {str(e)}")
            st.code(traceback.format_exc())
    
    def _render_welcome(self):
        """Render welcome screen when no embeddings exist."""
        st.markdown("""
        # 🤖 Welcome to MineMEETS
        
        <div style='background-color: #e8f5e8; padding: 2rem; border-radius: 10px; border-left: 6px solid #4CAF50;'>
        <h3 style='color: #2e7d32; margin-top: 0;'>🚀 Ready to Get Started?</h3>
        <p>Your intelligent meeting assistant is waiting for data. Upload your first meeting to begin!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 📋 What You Can Do:
        - **📝 Ingest** text files and transcribe audio meetings
        - **🖼️ Embed** images for visual content retrieval  
        - **💬 Chat** with your meeting data using AI
        - **🔍 Search** across all your meeting content

        ### 🎯 How to Start:
        1. **Use the sidebar** on the left to upload a file
        2. **Choose** from supported formats below
        3. **Click "Process Meeting"** to ingest your data
        4. **Start chatting** with your content!

        ### ✅ Supported Formats:
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🎙️ Audio Files**
            - MP3, WAV, M4A
            - OGG, FLAC
            """)
        with col2:
            st.markdown("""
            **🖼️ Image Files**  
            - PNG, JPG, JPEG
            - WebP, BMP
            """)
        with col3:
            st.markdown("""
            **📄 Documents**
            - TXT, PDF, DOCX
            - Text extraction
            """)
        
        # Quick upload section
        st.markdown("---")
        st.markdown("### 🚀 Quick Upload")
        quick_upload = st.file_uploader(
            "Or upload a file directly here:",
            type=["mp3", "wav", "m4a", "ogg", "flac", "png", "jpg", "jpeg", "webp", "bmp", "txt", "pdf", "docx"],
            key="quick_upload"
        )
        
        if quick_upload:
            file_ext = Path(quick_upload.name).suffix.lower()
            if file_ext in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']:
                file_type = 'audio'
            elif file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                file_type = 'image'
            else:
                file_type = 'file'
            
            if st.button("🚀 Process This File", type="primary", use_container_width=True):
                file_path = self.save_uploaded_file(quick_upload, file_type)
                if file_path:
                    self.process_meeting_sync(file_path, file_type)
    
    def _render_data_but_no_meeting(self):
        """Render when there's data but no meeting selected."""
        st.warning("📊 Database has data but no meeting is selected.")
        st.info("Please select a meeting from the sidebar or upload a new file.")
        
        # Show available meetings
        meetings = coordinator.list_meetings()
        if meetings:
            st.subheader("Available Meetings:")
            for m in meetings:
                if st.button(f"📄 Select: {m['title']}", key=f"select_main_{m['id']}"):
                    st.session_state.current_meeting = m["id"]
                    st.rerun()
    
    def _render_meeting_content(self, meeting: Dict[str, Any]):
        """Render meeting content tab."""
        st.subheader("Meeting Content")
        
        transcript = meeting.get('transcript', '')
        if transcript:
            with st.expander("📝 View Full Transcript", expanded=False):
                st.text_area("Transcript", transcript, height=300, key="transcript_view", label_visibility="collapsed")
        else:
            st.info("No transcript available for this meeting.")
        
        # Show chunks info
        chunks = meeting.get('chunks', [])
        if chunks:
            st.subheader(f"Document Chunks ({len(chunks)} total)")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                with st.expander(f"Chunk {i+1} - {chunk.get('metadata', {}).get('length', 0)} chars"):
                    st.text(chunk.get('text', '')[:500] + "..." if len(chunk.get('text', '')) > 500 else chunk.get('text', ''))
            
            if len(chunks) > 3:
                st.info(f"Showing first 3 of {len(chunks)} chunks. All chunks are stored in the vector database.")

    def _render_qa(self, meeting_id: str):
        """Render Q&A chat interface."""
        st.subheader("💬 Chat with Your Meeting Data")
        
        # Initialize chat history for this meeting
        if meeting_id not in st.session_state.qa_history:
            st.session_state.qa_history[meeting_id] = []
        
        # Display chat history
        history = st.session_state.qa_history[meeting_id]
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Show sources for assistant messages
                if msg["role"] == "assistant" and "sources" in msg:
                    sources = msg["sources"]
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources):
                                title = source.get("title") or source.get("metadata", {}).get("source") or f"Source {i+1}"
                                url = source.get("url") or source.get("metadata", {}).get("url") or ""
                                st.markdown(f"- **{title}**{f' — {url}' if url else ''}")
        
        # Chat input
        if prompt := st.chat_input("Ask something about this meeting..."):
            # Add user message to chat history
            st.session_state.qa_history[meeting_id].append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching through meeting content..."):
                    response = self.ask_question_sync(prompt)
                    answer = response.get('answer', 'No answer found.')
                    
                    st.markdown(answer)
                    
                    # Show sources if available
                    sources = response.get("sources", [])
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources):
                                title = source.get("title") or source.get("metadata", {}).get("source") or f"Source {i+1}"
                                url = source.get("url") or source.get("metadata", {}).get("url") or ""
                                st.markdown(f"- **{title}**{f' — {url}' if url else ''}")
            
            # Add assistant response to chat history with sources
            assistant_msg = {"role": "assistant", "content": answer}
            if sources:
                assistant_msg["sources"] = sources
            st.session_state.qa_history[meeting_id].append(assistant_msg)
            
            # Auto-refresh to show new message
            st.rerun()


def main():
    # Check for required environment variables
    if not os.getenv("PINECONE_API_KEY"):
        st.error("🔑 PINECONE_API_KEY not set")
        st.info("""
        **To fix this:**
        1. Create a `.env` file in your project root
        2. Add your Pinecone API key: `PINECONE_API_KEY=your_key_here`
        3. Restart the application
        """)
        return
    
    try:
        app = MeetingAssistantApp()
        app.render_sidebar()
        app.render_main()
    except Exception as e:
        st.error(f"🚨 Application error: {str(e)}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()