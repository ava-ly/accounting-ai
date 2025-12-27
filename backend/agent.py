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
            
            # Add system message only once at the beginning
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            
            if self.debug_enabled:
                logger.debug(f"Calling LLM with {len(messages)} messages")
            
            response = self.llm.invoke(messages)
            
            if self.debug_enabled:
                logger.debug(f"LLM response received: {type(response)}")
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.debug(f"Tool calls requested: {[tc['name'] for tc in response.tool_calls]}")
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in model call: {e}", exc_info=True)
            # Return error message instead of crashing
            error_msg = f"Model call failed: {str(e)}"
            return {"messages": [{"content": error_msg, "type": "error"}]}
    
    def _should_continue(self, state: AgentState) -> RouteDecision:
        """Determine if the agent should continue to tools or end.
        
        Args:
            state: Current agent state
            
        Returns:
            Route decision
        """
        messages = state["messages"]
        if not messages:
            logger.warning("No messages in state")
            return "end"
        
        last_message = messages[-1]
        
        # Check for error messages
        if isinstance(last_message, dict) and last_message.get("type") == "error":
            if self.debug_enabled:
                logger.info("DECISION: Stop (Error encountered)")
            return "end"
        
        step_count = len([msg for msg in messages if not isinstance(msg, SystemMessage)])
        
        if self.debug_enabled:
            logger.info(f"Step {step_count}: Agent produced message")
            if hasattr(last_message, 'content'):
                logger.debug(f"Message preview: {str(last_message.content)[:100]}...")
        
        # Check if the LLM wants to call tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            if self.debug_enabled:
                tool_names = [tc['name'] for tc in last_message.tool_calls]
                logger.info(f"DECISION: Calling Tools ({', '.join(tool_names)})")
            return "tools"
        
        # Check for maximum steps to prevent infinite loops
        max_steps = getattr(Config, 'MAX_AGENT_STEPS', 10)
        if step_count >= max_steps:
            if self.debug_enabled:
                logger.warning(f"DECISION: Stop (Max steps {max_steps} reached)")
            return "end"
        
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
            if self.debug_enabled:
                logger.info(f"Agent invoked with input: {list(input_data.keys())}")
            
            result = self.agent_executor.invoke(input_data)
            
            if self.debug_enabled:
                logger.info(f"Agent completed successfully")
                if "messages" in result:
                    final_message = result["messages"][-1]
                    if hasattr(final_message, 'content'):
                        logger.debug(f"Final response: {final_message.content[:200]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error invoking agent: {e}", exc_info=True)
            # Return structured error instead of raising
            return {
                "messages": [{"content": f"Agent invocation failed: {str(e)}", "type": "error"}],
                "error": str(e)
            }

    def get_agent_state_summary(self, state: AgentState) -> Dict[str, Any]:
        """Get a summary of the current agent state for debugging.
        
        Args:
            state: Current agent state
            
        Returns:
            State summary
        """
        messages = state.get("messages", [])
        return {
            "total_messages": len(messages),
            "has_system_message": any(isinstance(msg, SystemMessage) for msg in messages),
            "last_message_type": type(messages[-1]).__name__ if messages else None,
            "has_tool_calls": hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls if messages else False
        }


    def get_agent_state_summary(self, state: AgentState) -> Dict[str, Any]:
        """Get a summary of the current agent state for debugging.
        
        Args:
            state: Current agent state
            
        Returns:
            State summary
        """
        messages = state.get("messages", [])
        return {
            "total_messages": len(messages),
            "has_system_message": any(isinstance(msg, SystemMessage) for msg in messages),
            "last_message_type": type(messages[-1]).__name__ if messages else None,
            "has_tool_calls": hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls if messages else False
        }

    def get_agent_state_summary(self, state: AgentState) -> Dict[str, Any]:
        """Get a summary of the current agent state for debugging.
        
        Args:
            state: Current agent state
            
        Returns:
            State summary
        """
        messages = state.get("messages", [])
        return {
            "total_messages": len(messages),
            "has_system_message": any(isinstance(msg, SystemMessage) for msg in messages),
            "last_message_type": type(messages[-1]).__name__ if messages else None,
            "has_tool_calls": hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls if messages else False
        }

# Create global agent instance for backward compatibility
agent = AccountingAgent(debug_enabled=Config.ENABLE_DEBUG_PRINTING)
agent_executor = agent.agent_executor
