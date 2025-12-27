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
            messages = state.get("messages", [])
            
            # Track if we need to add system message
            # Check if any message is a SystemMessage with our SYSTEM_PROMPT
            has_system_prompt = False
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    if msg.content == SYSTEM_PROMPT:
                        has_system_prompt = True
                        break
            
            # Add system message if not present
            if not has_system_prompt:
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            
            if self.debug_enabled:
                logger.debug(f"Calling LLM with {len(messages)} messages")
                # Log first few messages for debugging
                for i, msg in enumerate(messages[:3]):
                    msg_type = type(msg).__name__
                    content_preview = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
                    logger.debug(f"Message {i} ({msg_type}): {content_preview}...")
            
            response = self.llm.invoke(messages)
            
            if self.debug_enabled:
                logger.debug(f"LLM response type: {type(response)}")
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
                    logger.debug(f"Tool calls requested: {tool_names}")
                elif hasattr(response, 'content'):
                    content_preview = str(response.content)[:100]
                    logger.debug(f"Response content preview: {content_preview}...")
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error in model call: {e}", exc_info=True)
            # Create a proper error message that follows the message format
            from langchain_core.messages import AIMessage
            error_content = f"Model call failed: {str(e)}"
            error_message = AIMessage(content=error_content)
            return {"messages": [error_message]}
    
    def _should_continue(self, state: AgentState) -> RouteDecision:
        """Determine if the agent should continue to tools or end.
        
        Args:
            state: Current agent state
            
        Returns:
            Route decision
        """
        messages = state.get("messages", [])
        if not messages:
            logger.warning("No messages in state")
            return "end"
        
        last_message = messages[-1]
        
        # Check for error messages
        # Handle both dict and AIMessage error cases
        is_error = False
        if isinstance(last_message, dict):
            if last_message.get("type") == "error":
                is_error = True
        elif hasattr(last_message, 'content'):
            content = str(last_message.content)
            if "Model call failed:" in content or "Agent invocation failed:" in content:
                is_error = True
        
        if is_error:
            if self.debug_enabled:
                logger.info("DECISION: Stop (Error encountered)")
            return "end"
        
        # Count non-system messages to track steps
        step_count = 0
        for msg in messages:
            if not isinstance(msg, SystemMessage):
                step_count += 1
        
        if self.debug_enabled:
            logger.info(f"Step {step_count}: Processing last message")
            if hasattr(last_message, 'content'):
                content_preview = str(last_message.content)[:100]
                logger.debug(f"Message preview: {content_preview}...")
        
        # Check if the LLM wants to call tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            if self.debug_enabled:
                tool_names = []
                for tc in last_message.tool_calls:
                    if isinstance(tc, dict):
                        tool_names.append(tc.get('name', 'unknown'))
                    else:
                        tool_names.append(getattr(tc, 'name', 'unknown'))
                logger.info(f"DECISION: Calling Tools ({', '.join(tool_names)})")
            return "tools"
        
        # Check for maximum steps to prevent infinite loops
        max_steps = getattr(Config, 'MAX_AGENT_STEPS', 10)
        if step_count >= max_steps:
            if self.debug_enabled:
                logger.warning(f"DECISION: Stop (Max steps {max_steps} reached)")
            return "end"
        
        # Default to end if no tool calls
        if self.debug_enabled:
            logger.info("DECISION: Stop (Final Answer)")
        return "end"
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent with input data.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent response
            
        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if not isinstance(input_data, dict):
            raise ValueError("input_data must be a dictionary")
        
        if "messages" not in input_data:
            raise ValueError("input_data must contain 'messages' key")
        
        if not isinstance(input_data["messages"], list):
            raise ValueError("input_data['messages'] must be a list")
        
        if self.debug_enabled:
            logger.info(f"Agent invoked with input: {list(input_data.keys())}")
            logger.debug(f"Number of initial messages: {len(input_data.get('messages', []))}")
        
        try:
            result = self.agent_executor.invoke(input_data)
            
            if self.debug_enabled:
                logger.info("Agent completed successfully")
                if "messages" in result and result["messages"]:
                    final_message = result["messages"][-1]
                    if hasattr(final_message, 'content'):
                        content_preview = str(final_message.content)[:200]
                        logger.debug(f"Final response preview: {content_preview}...")
                    else:
                        logger.debug(f"Final message type: {type(final_message).__name__}")
            
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

    def _cleanup_state(self, state: AgentState) -> AgentState:
        """Clean up the agent state by removing unnecessary data.
        
        Args:
            state: Current agent state
            
        Returns:
            Cleaned up state
        """
        # This is a placeholder for future state cleanup logic
        # For now, just return the state as-is
        if self.debug_enabled:
            logger.debug("State cleanup called")
        return state

# Create global agent instance for backward compatibility
agent = AccountingAgent(debug_enabled=Config.ENABLE_DEBUG_PRINTING)
agent_executor = agent.agent_executor
