# backend/coordinator.py
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import os
from agents.audio_agent import AudioAgent
from agents.insights_agent import InsightsAgent
from agents.qa_agent import QAAgent
from agents.internet_agent import InternetAgent
from agents.email_agent import EmailAgent
from agents.pinecone_db import PineconeDB
from agents.base_agent import AgentResponse

class MeetingCoordinator:
    def __init__(self):
        self.pinecone_db = PineconeDB()
        self.audio_agent = AudioAgent(model_size=os.getenv("WHISPER_MODEL", "base"))
        self.insights_agent = InsightsAgent()
        self.qa_agent = QAAgent(self.pinecone_db)
        self.internet_agent = InternetAgent()
        self.email_agent = EmailAgent()
        self.active_meetings = {}  # {meeting_id: meta+insights}

    def list_meetings(self):
        """
        Return a summary list of all known meetings (for listing in UI)
        """
        out = []
        for m in self.active_meetings.values():
            out.append({
                "id": m["id"],
                "title": m.get("title", m["id"]),
                "date": m.get("date", "N/A")
            })
        return out

    def get_meeting(self, meeting_id):
        """
        Retrieve the full meeting info for the given ID.
        """
        return self.active_meetings.get(meeting_id)

    async def process_meeting(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        meeting_id = meeting_data["id"]
        now = datetime.now(datetime.timezone.utc).isoformat()
        meta = {"id": meeting_id, "title": meeting_data.get("title", meeting_id), "date": meeting_data.get("date", now), "participants": meeting_data.get("participants", [])}
        self.active_meetings[meeting_id] = meta

        # Get transcript
        if meeting_data.get("type") == "transcript":
            if "content" in meeting_data:
                raw = meeting_data["content"]
            else:
                with open(meeting_data["content_path"], "r", encoding="utf-8") as f:
                    raw = f.read()
            # chunk
            chunks = self._chunk_text(raw, meeting_id)
        else:
            # audio/video
            audio_resp: AgentResponse = await self.audio_agent.process({"file_path": meeting_data["content_path"]}, context={"meeting_id": meeting_id})
            if not audio_resp.success:
                raise RuntimeError(audio_resp.content)
            raw = audio_resp.content["transcript"]
            chunks = audio_resp.content.get("chunks") or self._chunk_text(raw, meeting_id)

        # Upsert chunks
        self.pinecone_db.upsert_documents(chunks)

        # Extract insights
        insights_resp: AgentResponse = await self.insights_agent.process({"transcript": raw}, context={"meeting_id": meeting_id})
        insights = insights_resp.content if insights_resp.success else {}
        self.active_meetings[meeting_id].update({"transcript": raw, "chunks": chunks, "insights": insights})

        # Proactive recommendation: if action_items exist, prepare follow-ups
        action_items = insights.get("action_items", [])
        followups = []
        for ai in (action_items if isinstance(action_items, list) else [action_items]):
            followups.append({"task": ai, "suggested_email": self._prepare_email_for_ai(ai, meeting_id)})
        self.active_meetings[meeting_id]["followups"] = followups

        return {"meeting_id": meeting_id, "status": "processed", "insights": insights}

    def _prepare_email_for_ai(self, ai_text: str, meeting_id: str) -> Dict[str, str]:
        # Prepare a draft email; do not send automatically.
        return {"subject": f"Follow-up: {self.active_meetings[meeting_id]['title']}", "body": f"Automatic follow-up for action item:\n\n{ai_text}"}

    async def ask_question(self, question: str, meeting_id: Optional[str] = None) -> Dict[str, Any]:
        context = {}
        if meeting_id:
            context["meeting_id"] = meeting_id
        resp: AgentResponse = await self.qa_agent.process(question, context=context)
        answer = resp.content if resp.success else {"answer": "", "sources": []}
        if not resp.success or self._needs_internet_search(answer.get("answer","")):
            web_resp: AgentResponse = await self.internet_agent.process(question, context=context)
            if web_resp.success:
                combined = (answer.get("answer","") + "\n\nWeb: " + web_resp.content.get("answer","")).strip()
                sources = (answer.get("sources",[]) or []) + web_resp.content.get("sources",[])
                answer = {"answer": combined, "sources": sources}
        return answer

    async def send_insights_email(self, meeting_id: str, recipient_emails: Union[str, List[str]], additional_notes: str = ""):
        meeting = self.active_meetings.get(meeting_id)
        if not meeting:
            return {"success": False, "error": "Meeting not found"}
        resp = await self.email_agent.send_meeting_insights(to=recipient_emails, meeting_data={"title": meeting["title"], "date": meeting["date"], "participants": meeting["participants"]}, insights=meeting.get("insights", {}), additional_notes=additional_notes)
        return resp

    def _chunk_text(self, text: str, meeting_id: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        chunks = []
        start = 0
        i = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end]
            chunks.append({"text": chunk, "metadata": {"meeting_id": meeting_id, "chunk_id": f"{meeting_id}_chunk_{i}", "position": i, "length": len(chunk)}})
            start = max(0, end - chunk_overlap)
            i += 1
        return chunks

    def _needs_internet_search(self, answer: str) -> bool:
        a = (answer or "").lower()
        return any(x in a for x in ["i don't know", "not sure", "no information", "not mentioned"])
