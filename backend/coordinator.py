"""Coordinator service for managing agents and their interactions."""
from typing import Dict, Any, List, Optional, Union
from .agents.base_agent import BaseAgent, AgentResponse
from .agents.qa_agent import QAAgent
from .agents.internet_agent import InternetAgent
from .agents.email_agent import EmailAgent
from .pinecone_db import PineconeDB
import asyncio
from datetime import datetime
import json
import os

class MeetingCoordinator:
    """Coordinates between different agents and manages meeting data."""
    
    def __init__(self):
        """Initialize the coordinator with all agents."""
        # Initialize Pinecone DB
        self.pinecone_db = PineconeDB()
        
        # Initialize agents
        self.qa_agent = QAAgent(self.pinecone_db)
        self.internet_agent = InternetAgent()
        self.email_agent = EmailAgent()
        
        # Active meetings cache
        self.active_meetings: Dict[str, Dict[str, Any]] = {}
    
    async def process_meeting(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a new meeting and store it in the system.
        
        Args:
            meeting_data: Dictionary containing meeting data:
                - id: Unique meeting ID
                - title: Meeting title
                - date: Meeting date
                - participants: List of participants
                - content: Raw meeting content (transcript, audio, etc.)
                - type: Type of content ('transcript', 'audio', 'video')
                
        Returns:
            Dictionary with processing results
        """
        meeting_id = meeting_data['id']
        
        # Store meeting metadata
        self.active_meetings[meeting_id] = {
            'id': meeting_id,
            'title': meeting_data.get('title', f'Meeting {meeting_id}'),
            'date': meeting_data.get('date', datetime.now().isoformat()),
            'participants': meeting_data.get('participants', []),
            'type': meeting_data.get('type', 'transcript'),
            'created_at': datetime.now().isoformat()
        }
        
        # Process content based on type
        if meeting_data['type'] == 'transcript':
            # For text transcripts, chunk and embed directly
            chunks = self._chunk_text(meeting_data['content'])
        else:
            # For audio/video, we'd first transcribe then chunk
            # This is a simplified version - in practice, you'd use Whisper or similar
            raise NotImplementedError("Audio/Video processing not implemented yet")
        
        # Add to vector store
        self.qa_agent.add_meeting_context(meeting_id, chunks)
        
        # Generate initial insights
        insights = await self._generate_insights(meeting_id, chunks)
        
        # Store insights
        self.active_meetings[meeting_id]['insights'] = insights
        
        return {
            'meeting_id': meeting_id,
            'status': 'processed',
            'insights': insights
        }
    
    async def ask_question(self, question: str, meeting_id: Optional[str] = None) -> Dict[str, Any]:
        """Ask a question about one or all meetings.
        
        Args:
            question: The question to ask
            meeting_id: Optional meeting ID to scope the question to
            
        Returns:
            Dictionary with the answer and sources
        """
        # Build context for the question
        context = {}
        if meeting_id:
            context['meeting_id'] = meeting_id
            
            # Add meeting context to help with answering
            if meeting_id in self.active_meetings:
                context['meeting_context'] = {
                    'title': self.active_meetings[meeting_id]['title'],
                    'date': self.active_meetings[meeting_id]['date'],
                    'participants': self.active_meetings[meeting_id]['participants']
                }
        
        # First try to answer from local knowledge
        response = await self.qa_agent.process(question, context)
        
        # If confidence is low or answer is not satisfactory, try internet search
        if not response.success or self._needs_internet_search(response.content):
            internet_response = await self.internet_agent.process(question, context)
            if internet_response.success:
                # Combine responses
                response.content = {
                    'answer': f"{response.content.get('answer', 'I found this information:')}\n\n"
                             f"From the web: {internet_response.content.get('answer', 'No additional information found.')}",
                    'sources': response.content.get('sources', []) + internet_response.content.get('sources', [])
                }
        
        return response.content
    
    async def send_insights_email(
        self,
        meeting_id: str,
        recipient_emails: Union[str, List[str]],
        additional_notes: str = ""
    ) -> Dict[str, Any]:
        """Send meeting insights via email.
        
        Args:
            meeting_id: ID of the meeting
            recipient_emails: Email address(es) to send to
            additional_notes: Any additional notes to include
            
        Returns:
            Dictionary with the result of the email sending
        """
        if meeting_id not in self.active_meetings:
            return {
                'success': False,
                'error': f'Meeting {meeting_id} not found'
            }
        
        meeting = self.active_meetings[meeting_id]
        insights = meeting.get('insights', {})
        
        # Send the email
        response = await self.email_agent.send_meeting_insights(
            to=recipient_emails,
            meeting_data={
                'title': meeting['title'],
                'date': meeting['date'],
                'participants': meeting['participants']
            },
            insights=insights,
            additional_notes=additional_notes
        )
        
        return response.content
    
    def list_meetings(self) -> List[Dict[str, Any]]:
        """List all meetings in the system."""
        return [
            {
                'id': meeting_id,
                'title': meeting.get('title', 'Untitled'),
                'date': meeting.get('date'),
                'participants': meeting.get('participants', []),
                'type': meeting.get('type', 'unknown'),
                'created_at': meeting.get('created_at')
            }
            for meeting_id, meeting in self.active_meetings.items()
        ]
    
    def get_meeting(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific meeting."""
        return self.active_meetings.get(meeting_id)
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        return [
            {
                'text': chunk,
                'metadata': {
                    'chunk_id': f"chunk_{i}",
                    'chunk_size': len(chunk),
                    'chunk_position': i,
                    'total_chunks': len(chunks)
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    async def _generate_insights(self, meeting_id: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights from meeting chunks."""
        # Extract key information using the QA agent
        questions = [
            "What is the main purpose of this meeting?",
            "What are the key decisions made?",
            "List any action items with owners and deadlines.",
            "Who are the main participants and what were their contributions?",
            "What are the main topics discussed?"
        ]
        
        insights = {}
        
        for question in questions:
            response = await self.qa_agent.process(
                question,
                context={'meeting_id': meeting_id}
            )
            
            if response.success:
                key = question.lower().replace('?', '').replace(' ', '_')
                insights[key] = response.content.get('answer', 'No information found.')
        
        return insights
    
    def _needs_internet_search(self, response_content: Dict[str, Any]) -> bool:
        """Determine if an internet search is needed based on the response."""
        if not response_content or 'answer' not in response_content:
            return True
            
        answer = response_content['answer'].lower()
        uncertain_phrases = [
            "i don't know",
            "i'm not sure",
            "no information",
            "not mentioned",
            "can't find"
        ]
        
        return any(phrase in answer for phrase in uncertain_phrases)


# Singleton instance
coordinator = MeetingCoordinator()
