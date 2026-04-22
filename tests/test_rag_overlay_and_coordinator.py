"""Behavioral tests for overlay prompting and coordinator persistence."""

from unittest.mock import AsyncMock, Mock


from agents.coordinator import MeetingCoordinator
from agents.multimodal_rag import MultimodalRAGChain
from agents.qa_agent import QAAgent


class TestOverlayFlow:
    def test_prompt_includes_overlay_block(self):
        """Overlay config should be injected into the final prompt body."""
        rag = MultimodalRAGChain.__new__(MultimodalRAGChain)
        prompt = rag._create_prompt(
            context="Demo context",
            question="What happened?",
            modality_contexts={"text": [{"text": "demo"}], "audio": [], "image": [], "other": []},
            overlay_instructions="assistant_name=OpsBot\nresponse_style=Executive summary",
        )

        assert "ASSISTANT OVERLAY CONFIG" in prompt
        assert "assistant_name=OpsBot" in prompt
        assert "response_style=Executive summary" in prompt

    def test_qa_agent_passes_overlay_to_rag_chain(self):
        """QA layer should forward overlay settings to multimodal query."""
        qa = QAAgent.__new__(QAAgent)
        qa.multimodal_rag = Mock()
        qa.multimodal_rag.query = AsyncMock(
            return_value={
                "success": True,
                "answer": "stub-answer",
                "sources": [],
                "context_stats": {"sources_used": 0},
            }
        )

        import asyncio

        result = asyncio.run(
            qa.process(
                "Summarize",
                context={"meeting_id": "meeting_1", "overlay_instructions": "assistant_name=Nova"},
            )
        )

        assert result["success"] is True
        qa.multimodal_rag.query.assert_awaited_once_with(
            "Summarize", "meeting_1", overlay_instructions="assistant_name=Nova"
        )


class TestCoordinatorPersistence:
    def test_process_meeting_upserts_chunks_for_retrieval(self):
        """Processed chunks should be persisted into Pinecone namespace for later RAG."""
        coordinator = MeetingCoordinator.__new__(MeetingCoordinator)
        coordinator.active_meetings = {}
        coordinator.pinecone_db = Mock()

        async def _fake_process(*args, **kwargs):
            return {
                "success": True,
                "raw_content": "hello world",
                "chunks": [{"text": "chunk 1", "metadata": {"type": "text_chunk"}}],
            }

        coordinator._process_meeting_content = _fake_process

        import asyncio

        result = asyncio.run(coordinator.process_meeting({"id": "meeting_42", "type": "file"}))

        assert result["status"] == "processed"
        assert result["chunk_count"] == 1
        coordinator.pinecone_db.upsert_documents.assert_called_once_with(
            [{"text": "chunk 1", "metadata": {"type": "text_chunk"}}],
            namespace="meeting_42",
        )

    def test_text_type_alias_routes_to_file_processor(self):
        """`text` content type should be accepted as a compatibility alias."""
        coordinator = MeetingCoordinator.__new__(MeetingCoordinator)

        async def _fake_file_processor(meeting_data, meeting_id):
            return {"success": True, "raw_content": "ok", "chunks": []}

        coordinator._process_file = _fake_file_processor

        import asyncio

        result = asyncio.run(
            coordinator._process_meeting_content(
                {"id": "meeting_77", "type": "text", "content_path": "demo.txt"},
                "meeting_77",
            )
        )
        assert result["success"] is True
