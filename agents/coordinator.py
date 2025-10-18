from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import os
from agents.audio_agent import AudioAgent
from agents.insights_agent import InsightsAgent
from agents.qa_agent import QAAgent
from agents.internet_agent import InternetAgent
from agents.image_agent import ImageAgent
from agents.pinecone_db import PineconeDB
from agents.base_agent import AgentResponse

class MeetingCoordinator:
    def __init__(self):
        self.pinecone_db = PineconeDB()
        self.audio_agent = AudioAgent(model_size=os.getenv("WHISPER_MODEL", "base"))
        self.image_agent = ImageAgent()
        self.insights_agent = InsightsAgent()
        self.qa_agent = QAAgent(self.pinecone_db)
        self.internet_agent = InternetAgent()
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
        now = datetime.now(timezone.utc).isoformat()
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
        elif meeting_data.get("type") in ("audio", "video"):
            # audio/video
            audio_resp: AgentResponse = await self.audio_agent.process({"file_path": meeting_data["content_path"]}, context={"meeting_id": meeting_id})
            if not audio_resp.success:
                raise RuntimeError(audio_resp.content)
            # AudioTranscription returns keys: text, language, duration, chunks
            raw = audio_resp.content.get("text", "")
            chunks = audio_resp.content.get("chunks") or self._chunk_text(raw, meeting_id)
            # Normalize Whisper segments to our chunk format if necessary
            if chunks and isinstance(chunks[0], dict) and "metadata" not in chunks[0]:
                norm = []
                for i, seg in enumerate(chunks):
                    txt = seg.get("text", "")
                    start = seg.get("start")
                    end = seg.get("end")
                    norm.append({
                        "text": txt,
                        "metadata": {
                            "meeting_id": meeting_id,
                            "chunk_id": f"{meeting_id}_chunk_{i}",
                            "position": i,
                            "start": start,
                            "end": end,
                            "length": len(txt)
                        }
                    })
                chunks = norm
        elif meeting_data.get("type") in ("image", "screenshot"):
            # image/screenshots: compute embeddings and upsert directly
            img_resp: AgentResponse = await self.image_agent.process({"file_path": meeting_data["content_path"]}, context={"meeting_id": meeting_id})
            if not img_resp.success:
                raise RuntimeError(img_resp.content)
            vectors = img_resp.content.get("vectors", [])
            self.pinecone_db.upsert_vectors(vectors, namespace=meeting_id)
            raw = meeting_data.get("caption", "")
            chunks = []
        else:
            raise RuntimeError(f"Unsupported meeting type: {meeting_data.get('type')}")

        # Upsert text chunks into Pinecone (uses internal embeddings)
        if chunks:
            self.pinecone_db.upsert_documents(chunks, namespace=meeting_id)

        # Extract insights
        insights_resp: AgentResponse = await self.insights_agent.process({"transcript": raw}, context={"meeting_id": meeting_id})
        insights = insights_resp.content if insights_resp.success else {}
        self.active_meetings[meeting_id].update({"transcript": raw, "chunks": chunks, "insights": insights})

        # Proactive recommendation: if action_items exist, prepare follow-ups
        action_items = insights.get("action_items", [])
        followups = []
        for ai in (action_items if isinstance(action_items, list) else [action_items]):
            followups.append({"task": ai})
        self.active_meetings[meeting_id]["followups"] = followups

        return {"meeting_id": meeting_id, "status": "processed", "insights": insights}

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

# Module-level coordinator instance for app imports
coordinator = MeetingCoordinator()
