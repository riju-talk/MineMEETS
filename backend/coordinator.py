# backend/coordinator.py

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import os
import asyncio

from .agents.base_agent import AgentResponse
from .agents.audio_agent import AudioAgent
from .agents.insights_agent import InsightsAgent
from .agents.qa_agent import QAAgent
from .agents.internet_agent import InternetAgent
from .agents.email_agent import EmailAgent
from .pinecone_db import PineconeDB


class MeetingCoordinator:
    """Coordinates between different agents and manages meeting data."""

    def __init__(self):
        # initialize vector store
        self.pinecone_db = PineconeDB()

        # initialize agents
        self.audio_agent = AudioAgent(model_size=os.getenv("WHISPER_MODEL", "base"))
        self.insights_agent = InsightsAgent()
        self.qa_agent = QAAgent(self.pinecone_db)
        self.internet_agent = InternetAgent()
        self.email_agent = EmailAgent()

        # in‐memory meeting store
        self.active_meetings: Dict[str, Dict[str, Any]] = {}

    async def process_meeting(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a new meeting (text or audio/video), store embeddings, and extract insights.

        meeting_data keys:
          - id: str
          - title: str
          - date: ISO‐string
          - participants: List[str]
          - content_path: filesystem path to file
          - type: 'transcript' | 'audio' | 'video'
        """
        meeting_id = meeting_data["id"]
        now = datetime.utcnow().isoformat()
        self.active_meetings[meeting_id] = {
            "id": meeting_id,
            "title": meeting_data.get("title", f"Meeting {meeting_id}"),
            "date": meeting_data.get("date", now),
            "participants": meeting_data.get("participants", []),
            "type": meeting_data.get("type", "transcript"),
            "created_at": now,
        }

        # 1️⃣ Get raw transcript
        if meeting_data["type"] == "transcript":
            raw_text = open(meeting_data["content_path"], "r", encoding="utf-8").read()
        else:
            # audio or video → transcribe
            audio_resp: AgentResponse = await self.audio_agent.process(
                {"file_path": meeting_data["content_path"]},
                context={"meeting_id": meeting_id}
            )
            if not audio_resp.success:
                raise RuntimeError(f"Transcription failed: {audio_resp.content}")
            raw_text = audio_resp.content["transcript"]

        # 2️⃣ Chunk & embed into Pinecone
        chunks = self._chunk_text(raw_text)
        self.qa_agent.add_meeting_context(meeting_id, chunks)

        # store transcript
        self.active_meetings[meeting_id]["transcript"] = raw_text
        self.active_meetings[meeting_id]["chunks"] = chunks

        # 3️⃣ Extract structured insights
        insights_resp: AgentResponse = await self.insights_agent.process(
            {"transcript": raw_text},
            context={"meeting_id": meeting_id}
        )
        insights = insights_resp.content if insights_resp.success else {}
        self.active_meetings[meeting_id]["insights"] = insights

        return {"meeting_id": meeting_id, "status": "processed", "insights": insights}

    async def ask_question(
        self,
        question: str,
        meeting_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer questions scoped to a meeting (or globally), with internet fallback."""
        context: Dict[str, Any] = {}
        if meeting_id:
            context["meeting_id"] = meeting_id
            meta = self.active_meetings.get(meeting_id, {})
            context["meeting_context"] = {
                "title": meta.get("title"),
                "date": meta.get("date"),
                "participants": meta.get("participants"),
            }

        # 4️⃣ First try vector‐based QA
        qa_resp: AgentResponse = await self.qa_agent.process(question, context)
        answer_content = qa_resp.content if qa_resp.success else {}

        # 5️⃣ If uncertain → web search
        if not qa_resp.success or self._needs_internet_search(answer_content.get("answer", "")):
            web_resp: AgentResponse = await self.internet_agent.process(question, context)
            if web_resp.success:
                combined = (
                    f"{answer_content.get('answer','')}\n\n"
                    f"ℹ️ From web: {web_resp.content.get('answer')}"
                )
                sources = (answer_content.get("sources", []) or []) + web_resp.content.get("sources", [])
                answer_content = {"answer": combined, "sources": sources}

        return answer_content

    async def send_insights_email(
        self,
        meeting_id: str,
        recipient_emails: Union[str, List[str]],
        additional_notes: str = ""
    ) -> Dict[str, Any]:
        """Email the stored insights for a given meeting."""
        if meeting_id not in self.active_meetings:
            return {"success": False, "error": "Meeting not found"}

        meeting = self.active_meetings[meeting_id]
        insights = meeting.get("insights", {})

        email_resp: AgentResponse = await self.email_agent.send_meeting_insights(
            to=recipient_emails,
            meeting_data={
                "title": meeting["title"],
                "date": meeting["date"],
                "participants": meeting["participants"],
            },
            insights=insights,
            additional_notes=additional_notes,
        )
        return email_resp.content

    def list_meetings(self) -> List[Dict[str, Any]]:
        """Return basic metadata for all processed meetings."""
        return [
            {
                "id": m["id"],
                "title": m["title"],
                "date": m["date"],
                "participants": m["participants"],
                "type": m["type"],
                "created_at": m["created_at"],
            }
            for m in self.active_meetings.values()
        ]

    def get_meeting(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Get full meeting record (transcript, insights, etc.)."""
        return self.active_meetings.get(meeting_id)

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        raw_chunks = splitter.split_text(text)
        return [
            {
                "text": chunk,
                "metadata": {
                    "chunk_id": f"{meeting_id}_chunk_{i}",
                    "position": i,
                    "length": len(chunk),
                },
            }
            for i, chunk in enumerate(raw_chunks)
        ]

    def _needs_internet_search(self, answer: str) -> bool:
        """Simple heuristics to decide if we should hit the web."""
        ans = (answer or "").lower()
        triggers = [
            "i don't know", "i'm not sure", "no information",
            "not mentioned", "can't find"
        ]
        return any(t in ans for t in triggers)


# Create a singleton for import
coordinator = MeetingCoordinator()
