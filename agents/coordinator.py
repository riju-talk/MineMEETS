# backend/coordinator.py

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import os

from .agents.base_agent import AgentResponse
from .agents.audio_agent import AudioAgent
from .agents.insights_agent import InsightsAgent
from .agents.qa_agent import QAAgent
from .agents.internet_agent import InternetAgent
from .agents.email_agent import EmailAgent
from .pinecone_db import PineconeDB


class MeetingCoordinator:
    """Coordinates meeting ingestion, QA, and insights extraction."""

    def __init__(self):
        self.pinecone_db = PineconeDB()

        self.audio_agent = AudioAgent(model_size=os.getenv("WHISPER_MODEL", "base"))
        self.insights_agent = InsightsAgent()
        self.qa_agent = QAAgent(self.pinecone_db)
        self.internet_agent = InternetAgent()
        self.email_agent = EmailAgent()

        self.active_meetings: Dict[str, Dict[str, Any]] = {}

    async def process_meeting(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
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

        # 1️⃣ Get transcript & chunks
        if meeting_data["type"] == "transcript":
            raw_text = open(meeting_data["content_path"], "r", encoding="utf-8").read()
            chunks = self._chunk_text(raw_text, meeting_id)
        else:
            audio_resp: AgentResponse = await self.audio_agent.process(
                {"file_path": meeting_data["content_path"]},
                context={"meeting_id": meeting_id}
            )
            if not audio_resp.success:
                raise RuntimeError(f"Transcription failed: {audio_resp.content}")

            raw_text = audio_resp.content["transcript"]
            chunks = audio_resp.content["chunks"]

        # 2️⃣ Store in vector DB via QAAgent
        self.qa_agent.add_meeting_context(meeting_id, chunks)

        # Store transcript/chunks
        self.active_meetings[meeting_id]["transcript"] = raw_text
        self.active_meetings[meeting_id]["chunks"] = chunks

        # 3️⃣ Extract insights
        insights_resp: AgentResponse = await self.insights_agent.process(
            {"transcript": raw_text},
            context={"meeting_id": meeting_id}
        )
        insights = insights_resp.content if insights_resp.success else {}
        self.active_meetings[meeting_id]["insights"] = insights

        return {"meeting_id": meeting_id, "status": "processed", "insights": insights}

    def _chunk_text(self, text: str, meeting_id: str) -> List[Dict[str, Any]]:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return [
            {
                "text": chunk,
                "metadata": {
                    "meeting_id": meeting_id,
                    "chunk_index": i,
                    "length": len(chunk)
                }
            }
            for i, chunk in enumerate(splitter.split_text(text))
        ]
