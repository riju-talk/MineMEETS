"""Internet Agent for performing web searches and gathering information."""
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResponse
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool, AgentType
import os

class InternetAgent(BaseAgent):
    """Agent for performing web searches and gathering information."""
    
    def __init__(self):
        """Initialize the Internet agent with search capabilities."""
        super().__init__(
            name="internet_agent",
            description="Performs web searches to gather additional information"
        )
        self.search = GoogleSearchAPIWrapper(google_api_key=os.getenv("GOOGLE_API_KEY"),
                                           google_cse_id=os.getenv("GOOGLE_CSE_ID"))
        self.tools = [
            Tool(
                name="Search",
                func=self.search.run,
                description="Useful for when you need to answer questions about current events or find recent information"
            )
        ]
        self.agent = initialize_agent(
            self.tools,
            ChatOpenAI(temperature=0, model_name="gpt-4"),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process a search query and return results.
        
        Args:
            query: The search query or question
            context: Additional context for the search
            
        Returns:
            AgentResponse with search results
        """
        try:
            # Add context to the query if available
            if context:
                if "meeting_context" in context:
                    query = f"{query} (Meeting context: {context['meeting_context']})"
                
                if "search_type" in context:
                    query = f"{context['search_type']}: {query}"
            
            # Perform the search
            result = self.agent.run(query)
            
            return AgentResponse(
                success=True,
                content={
                    "answer": result,
                    "sources": [{"type": "web_search", "query": query}]
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                content=f"Error performing search: {str(e)}"
            )
    
    async def verify_information(self, statement: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Verify a statement by searching for supporting or contradicting information.
        
        Args:
            statement: The statement to verify
            context: Additional context for the verification
            
        Returns:
            AgentResponse with verification results
        """
        query = f"Verify the following statement and find reliable sources: {statement}"
        return await self.process(query, {"search_type": "Fact check", **context})
