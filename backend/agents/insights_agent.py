# agents/insights_agent.py

from typing import Dict, Any, Optional
from .base_agent import BaseAgent, AgentResponse
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import os

class InsightsAgent(BaseAgent):
    """Agent that extracts summaries, decisions, and action items from meeting content."""

    def __init__(self):
        """Initialize the insights extraction agent."""
        super().__init__(
            name="insights_agent",
            description="Extracts summary, key decisions, and action items from meeting transcripts."
        )
        self.llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.prompts = {
            "summary": PromptTemplate.from_template(
                "Summarize the following meeting:\n\n{transcript}\n\nSummary:"
            ),
            "key_decisions": PromptTemplate.from_template(
                "Extract the key decisions made during this meeting:\n\n{transcript}\n\nKey Decisions:"
            ),
            "action_items": PromptTemplate.from_template(
                "List any action items discussed, including owners and deadlines if available:\n\n{transcript}\n\nAction Items:"
            )
        }
        self.chains = {
            name: LLMChain(llm=self.llm, prompt=prompt)
            for name, prompt in self.prompts.items()
        }

    async def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Extract insights from the meeting transcript.

        Args:
            input_data: Must contain 'transcript' (string)
            context: Optional additional metadata (e.g. meeting_id)

        Returns:
            AgentResponse with insights (summary, key decisions, actions)
        """
        try:
            transcript = input_data.get("transcript", "")
            if not transcript:
                raise ValueError("Transcript is required for insights extraction.")

            results = {}
            for key, chain in self.chains.items():
                result = chain.run(transcript=transcript)
                results[key] = result.strip()

            return AgentResponse(
                success=True,
                content={
                    "summary": results.get("summary", ""),
                    "key_points": results.get("key_decisions", ""),
                    "action_items": results.get("action_items", "")
                },
                metadata={"source": "insights_agent", **(context or {})}
            )

        except Exception as e:
            return AgentResponse(
                success=False,
                content=f"Failed to extract insights: {str(e)}",
                metadata={"error": str(e)}
            )
