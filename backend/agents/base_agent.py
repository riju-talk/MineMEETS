"""Base agent class for all specialized agents."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class AgentResponse(BaseModel):
    """Standardized response format for all agents."""
    success: bool = Field(..., description="Whether the operation was successful")
    content: Any = Field(..., description="The response content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, description: str):
        """Initialize the agent with a name and description."""
        self.name = name
        self.description = description
        self.is_active = True
    
    @abstractmethod
    async def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Process the input and return a response.
        
        Args:
            input_data: The input data to process
            context: Optional context for the processing
            
        Returns:
            AgentResponse containing the result
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        return {
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active
        }
    
    async def __call__(self, *args, **kwargs) -> AgentResponse:
        """Make the agent callable."""
        return await self.process(*args, **kwargs)
