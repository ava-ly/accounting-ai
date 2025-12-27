import logging
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from config import Config, create_llm
from prompts import SYSTEM_PROMPT
from agent_types import AgentState, RouteDecision

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccountingAgent:
    """Main accounting AI agent class."""
    
    def __init__(self, debug_enabled: bool = True):
        """Initialize the accounting agent.
        
        Args:
            debug_enabled: Whether to enable debug printing
        """
        self.debug_enabled = debug_enabled
        self.llm = create_llm()
        self.tool_node = ToolNode(Config.TOOLS)
        self.workflow = self._build_workflow()
        self.agent_executor = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow graph.
        
        Returns:
            Configured StateGraph instance
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        
        return workflow
    
    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Call the LLM model with the current state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with model response
        """
        try:
            messages = state["messages"]
            
            # Add system message if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            
            response = self.llm.invoke(messages)
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in model call: {e}")
            raise
    
    def _should_continue(self, state: AgentState) -> RouteDecision:
        """Determine if the agent should continue to tools or end.
        
        Args:
            state: Current agent state
            
        Returns:
            Route decision
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        step_count = len(messages)
        
        if self.debug_enabled:
            logger.info(f"Step {step_count}: Agent produced {len(last_message.content)} chars.")
        
        # Check if the LLM wants to call tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            if self.debug_enabled:
                tool_name = last_message.tool_calls[0]['name']
                logger.info(f"DECISION: Calling Tool ({tool_name})")
            return "tools"
        
        if self.debug_enabled:
            logger.info("DECISION: Stop (Final Answer)")
        return "end"
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent with input data.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent response
        """
        try:
            return self.agent_executor.invoke(input_data)
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            raise


# Create global agent instance for backward compatibility
agent = AccountingAgent(debug_enabled=Config.ENABLE_DEBUG_PRINTING)
agent_executor = agent.agent_executor