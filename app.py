import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Debug: Print environment variables
print("Environment variables loaded:")
print(f"PINECONE_API_KEY: {'*' * len(os.getenv('PINECONE_API_KEY', '')) if os.getenv('PINECONE_API_KEY') else 'Not set'}")
print(f"GROQ_API_KEY: {'*' * len(os.getenv('GROQ_API_KEY', '')) if os.getenv('GROQ_API_KEY') else 'Not set'}")
print(f"SMTP_SERVER: {os.getenv('SMTP_SERVER', 'Not set')}")

# Import coordinator
from backend.coordinator import coordinator

# Configure page
st.set_page_config(
    page_title="MineMEETS - Meeting Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
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

class MeetingAssistantApp:
    def __init__(self):
        """Initialize the app."""
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables."""
        if 'current_meeting' not in st.session_state:
            st.session_state.current_meeting = None
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = {}
    
    async def process_meeting(self, file_path: str, file_type: str):
        """Process a meeting file."""
        with st.spinner("Processing meeting..."):
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create meeting data
            meeting_data = {
                'id': f"meeting_{int(datetime.now().timestamp())}",
                'title': Path(file_path).stem,
                'date': datetime.now().isoformat(),
                'participants': ["Participant 1", "Participant 2"],  # Mock participants
                'content': content,
                'type': file_type
            }
            
            # Process with coordinator
            result = await coordinator.process_meeting(meeting_data)
            
            if result['status'] == 'processed':
                st.session_state.current_meeting = meeting_data['id']
                st.session_state.qa_history[meeting_data['id']] = []
                return True
            return False
    
    async def ask_question(self, question: str):
        """Ask a question about the current meeting."""
        if not st.session_state.current_meeting:
            return "No active meeting. Please upload a meeting first."
        
        # Add to QA history
        meeting_id = st.session_state.current_meeting
        st.session_state.qa_history[meeting_id].append({"role": "user", "content": question})
        
        # Get answer
        response = await coordinator.ask_question(
            question=question,
            meeting_id=meeting_id
        )
        
        # Add to QA history
        st.session_state.qa_history[meeting_id].append({
            "role": "assistant",
            "content": response.get('answer', 'No answer found.')
        })
        
        return response
    
    def render_sidebar(self):
        """Render the sidebar with meeting list and uploader."""
        st.sidebar.title("Meetings")
        
        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Upload Meeting",
            type=["txt"],  # Add more types as needed
            help="Upload a meeting transcript or recording"
        )
        
        if uploaded_file is not None:
            # Save file
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            file_path = temp_dir / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process file
            if st.sidebar.button("Process Meeting"):
                asyncio.run(self.process_meeting(
                    str(file_path),
                    file_type="transcript"
                ))
                st.experimental_rerun()
            
            # Cleanup
            file_path.unlink()
        
        # List meetings
        meetings = coordinator.list_meetings()
        if meetings:
            st.sidebar.subheader("Your Meetings")
            for meeting in meetings:
                if st.sidebar.button(
                    f"📄 {meeting['title']}",
                    key=f"meeting_{meeting['id']}",
                    use_container_width=True
                ):
                    st.session_state.current_meeting = meeting['id']
                    st.experimental_rerun()
    
    def render_main(self):
        """Render the main content area."""
        if not st.session_state.current_meeting:
            self._render_welcome()
            return
        
        # Get current meeting
        meeting_id = st.session_state.current_meeting
        meeting = coordinator.get_meeting(meeting_id)
        
        if not meeting:
            st.error("Meeting not found")
            return
        
        # Display meeting info
        st.title(meeting.get('title', 'Untitled Meeting'))
        st.caption(f"Date: {meeting.get('date', 'N/A')}")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📄 Content", "🔍 Insights", "❓ Q&A"])
        
        with tab1:
            self._render_meeting_content(meeting)
        
        with tab2:
            self._render_insights(meeting)
        
        with tab3:
            self._render_qa(meeting_id)
    
    def _render_welcome(self):
        """Render welcome screen."""
        st.markdown("""
        # 🤖 Welcome to MineMEETS
        
        Your intelligent meeting assistant that helps you:
        - 📝 Transcribe and analyze meeting content
        - 🔍 Extract key insights and action items
        - ❓ Get answers to your questions
        - 📧 Share insights via email
        
        ### Get Started
        1. Upload a meeting file using the sidebar
        2. View and analyze the content
        3. Ask questions about the meeting
        
        Supported formats: TXT (more coming soon!)
        """)
    
    def _render_meeting_content(self, meeting: Dict[str, Any]):
        """Render meeting content tab."""
        st.subheader("Meeting Content")
        
        # Display content with syntax highlighting
        st.code(meeting.get('content', 'No content available'), language='text')
        
        # Email insights button
        if st.button("📧 Email Insights"):
            email = st.text_input("Enter email address:")
            if email:
                with st.spinner("Sending email..."):
                    result = asyncio.run(coordinator.send_insights_email(
                        meeting_id=meeting['id'],
                        recipient_emails=[email],
                        additional_notes="Here are the insights from our meeting."
                    ))
                    
                    if result.get('success'):
                        st.success(f"Email sent to {email}")
                    else:
                        st.error(f"Failed to send email: {result.get('error', 'Unknown error')}")
    
    def _render_insights(self, meeting: Dict[str, Any]):
        """Render insights tab."""
        st.subheader("Meeting Insights")
        
        # Show loading spinner while fetching insights
        with st.spinner("Generating insights..."):
            insights = meeting.get('insights', {})
            
            # Display key insights
            st.markdown("### 📝 Summary")
            st.markdown(insights.get('what is the main purpose of this meeting', 'No summary available.'))
            
            # Key points
            st.markdown("### 🔑 Key Points")
            key_points = insights.get('what are the key decisions made', 'No key points available.')
            if isinstance(key_points, str):
                st.markdown(key_points)
            else:
                for point in key_points:
                    st.markdown(f"- {point}")
            
            # Action items
            st.markdown("### ✅ Action Items")
            action_items = insights.get('list any action items with owners and deadlines', 'No action items found.')
            if isinstance(action_items, str):
                st.markdown(action_items)
            else:
                for item in action_items:
                    st.markdown(f"- {item}")
    
    def _render_qa(self, meeting_id: str):
        """Render Q&A tab."""
        st.subheader("Ask a Question")
        
        # Question input
        question = st.text_input(
            "Ask something about this meeting:",
            key=f"question_{meeting_id}"
        )
        
        if question:
            with st.spinner("Thinking..."):
                response = asyncio.run(self.ask_question(question))
                st.markdown(f"**Answer:** {response.get('answer', 'No answer found.')}")
        
        # Display conversation history
        if meeting_id in st.session_state.qa_history and st.session_state.qa_history[meeting_id]:
            st.markdown("---")
            st.subheader("Conversation History")
            
            for msg in st.session_state.qa_history[meeting_id]:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Assistant:** {msg['content']}")
                    st.markdown("---")

def main():
    """Main function to run the app."""
    # Initialize app
    app = MeetingAssistantApp()
    
    # Check for required API keys
    if not os.getenv("PINECONE_API_KEY"):
        st.error("PINECONE_API_KEY environment variable not set. Please set it in a .env file.")
        return
    
    # Render the app
    app.render_sidebar()
    app.render_main()

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    main()
