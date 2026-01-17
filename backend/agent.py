# import python's built-in logging module for logging messages
import logging
# import type hints `Any` (any type) and `Dict` (dictionary type)
from typing import Any, Dict

# import 'SystemMessage' class for system prompts in langchain
# giving AI agent its job description and rules
from langchain_core.messages import SystemMessage
# import LangGraph components: 
# 'StateGraph': Creates a map of how your agent processes information from start to finish
# 'START': The entry point (when a user asks a question)
# 'END': The exit point (when you deliver the final answer)
from langgraph.graph import StateGraph, START, END
# import 'ToolNode' for executing tools in the graph 
from langgraph.prebuilt import ToolNode

# import configuration settings and LLM creation function
from config import Config, create_llm
# import the system prompt text for the agent
from prompts import SYSTEM_PROMPT
# import custom types: 'AgentState' for state structure, 'RouteDecision' for routing
from agent_types import AgentState, RouteDecision
# import tools for fallback
from tools import search_accounting_law, calculate_vat
# import tools for fallback
from tools import search_accounting_law, calculate_vat

# Configure logging
# set up basic logging configuration to INFO level
# a system that tracks what the AI agent is doing behind the scenes
logging.basicConfig(level=logging.INFO)
# create logger instance named after the current module
logger = logging.getLogger(__name__)


class AccountingAgent:
    """Main accounting AI agent class."""
    
    # constructor method, take 'debug_enabled' param, defaults to True
    def __init__(self, debug_enabled: bool = True):
        """Initialize the accounting agent.
        
        Args:
            debug_enabled: Whether to enable debug printing
        """
        # store debug flag
        self.debug_enabled = debug_enabled
        
        # Get tools, handling potential issues
        tools = []
        try:
            tools = Config.TOOLS
            if not tools:
                raise ValueError("Config.TOOLS is empty")
            logger.info(f"Loaded {len(tools)} tools from Config")
        except Exception as e:
            logger.error(f"Failed to get tools from Config: {e}")
            # Fallback to directly imported tools
            tools = [search_accounting_law, calculate_vat]
            logger.warning(f"Using fallback tools: {[t.name for t in tools if hasattr(t, 'name')]}")
        
        # Log tool names for debugging
        if self.debug_enabled:
            tool_names = []
            for tool in tools:
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                else:
                    tool_names.append(str(tool))
            logger.info(f"Available tools: {tool_names}")
        
        # create llm instance using imported function, passing the same tools
        self.llm = create_llm(tools)
        # create toolnode with tools
        self.tool_node = ToolNode(tools)
        # build the workflow graph
        self.workflow = self._build_workflow()
        # compile the graph into an executable agent
        self.agent_executor = self.workflow.compile()
    
    # Method signature returning `StateGraph`
    def _build_workflow(self) -> StateGraph:
        """Build the agent workflow graph.
        
        Returns:
            Configured StateGraph instance
        """
        # Creates a new StateGraph with `AgentState` as state type
        workflow = StateGraph(AgentState)
        
        # Add nodes
        # Adds two nodes: "agent" (calls LLM) and "tools" (executes tools)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)
        
        # Add edges
        # Adds edge from START → "agent" (always starts with agent)
        workflow.add_edge(START, "agent")
        # Adds conditional edge from "agent" → decides next step
        # Calls `_should_continue` to decide, If returns "tools" → goes to tools node
        # If returns "end" → goes to END
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END}
        )
        # Adds edge from "tools" → "agent" (loop back after tools)
        workflow.add_edge("tools", "agent")
        # Returns the built workflow
        return workflow
    
    # Method takes `AgentState`, returns dictionary
    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Call the LLM model with the current state.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with model response
        """
        try:
            messages = state.get("messages", [])
            messages = self._ensure_system_prompt(messages)
            
            self._log_messages_preview(messages)
            response = self.llm.invoke(messages)
            self._log_response_preview(response)
            
            return {"messages": [response]}
        except Exception as e:
            return self._handle_model_error(e)
    
    def _ensure_system_prompt(self, messages: list) -> list:
        """Ensure system prompt is present in messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Messages with system prompt prepended if not present
        """
        # Check if system prompt already exists
        for msg in messages:
            if isinstance(msg, SystemMessage) and msg.content == SYSTEM_PROMPT:
                return messages
        
        # Add system prompt at the beginning
        return [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    def _log_messages_preview(self, messages: list) -> None:
        """Log preview of messages for debugging.
        
        Args:
            messages: List of messages to log
        """
        if not self.debug_enabled:
            return
        
        logger.debug(f"Calling LLM with {len(messages)} messages")
        for i, msg in enumerate(messages[:3]):
            msg_type = type(msg).__name__
            content_preview = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
            logger.debug(f"Message {i} ({msg_type}): {content_preview}...")
    
    def _log_response_preview(self, response: Any) -> None:
        """Log preview of LLM response for debugging.
        
        Args:
            response: LLM response object
        """
        if not self.debug_enabled:
            return
        
        logger.debug(f"LLM response type: {type(response)}")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
            logger.debug(f"Tool calls requested: {tool_names}")
        elif hasattr(response, 'content'):
            content_preview = str(response.content)[:100]
            logger.debug(f"Response content preview: {content_preview}...")
    
    def _handle_model_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors during model call.
        
        Args:
            error: Exception that occurred
            
        Returns:
            State with error message
        """
        logger.error(f"Error in model call: {error}", exc_info=True)
        from langchain_core.messages import AIMessage
        error_content = f"Model call failed: {str(error)}"
        error_message = AIMessage(content=error_content)
        return {"messages": [error_message]}
    
    # Method takes `AgentState`, returns `RouteDecision` (string)
    def _should_continue(self, state: AgentState) -> RouteDecision:
        """Determine if the agent should continue to tools or end.
        
        Args:
            state: Current agent state
            
        Returns:
            Route decision
        """
        # Gets messages from state
        messages = state.get("messages", [])
        # If no messages, end
        if not messages:
            logger.warning("No messages in state")
            return "end"
        # Gets last message
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
        
        # If error, end
        if is_error:
            if self.debug_enabled:
                logger.info("DECISION: Stop (Error encountered)")
            return "end"
        
        # Count non-system messages to track steps
        step_count = 0
        for msg in messages:
            if not isinstance(msg, SystemMessage):
                step_count += 1
        
        # Debug logging
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
        if step_count >= Config.MAX_AGENT_STEPS:
            if self.debug_enabled:
                logger.warning(f"DECISION: Stop (Max steps {Config.MAX_AGENT_STEPS} reached)")
            return "end"
        
        # Default to end if no tool calls
        if self.debug_enabled:
            logger.info("DECISION: Stop (Final Answer)")
        return "end"
    
    # Takes input dictionary, returns result dictionary
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent with input data.
        
        Args:
            input_data: Input data for the agent
            
        Returns:
            Agent response
            
        Raises:
            ValueError: If input validation fails
        """
        self._validate_input(input_data)
        
        if self.debug_enabled:
            logger.info(f"Agent invoked with input keys: {list(input_data.keys())}")
            logger.debug(f"Number of initial messages: {len(input_data.get('messages', []))}")
        
        try:
            result = self.agent_executor.invoke(input_data)
            self._log_result_preview(result)
            return result
        except Exception as e:
            return self._handle_invocation_error(e)
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """Validate input data for agent invocation.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(input_data, dict):
            raise ValueError("input_data must be a dictionary")
        
        if "messages" not in input_data:
            raise ValueError("input_data must contain 'messages' key")
        
        if not isinstance(input_data["messages"], list):
            raise ValueError("input_data['messages'] must be a list")
    
    def _log_result_preview(self, result: Dict[str, Any]) -> None:
        """Log preview of agent result for debugging.
        
        Args:
            result: Agent result to log
        """
        if not self.debug_enabled:
            return
        
        logger.info("Agent completed successfully")
        if "messages" in result and result["messages"]:
            final_message = result["messages"][-1]
            if hasattr(final_message, 'content'):
                content_preview = str(final_message.content)[:200]
                logger.debug(f"Final response preview: {content_preview}...")
            else:
                logger.debug(f"Final message type: {type(final_message).__name__}")
    
    def _handle_invocation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle errors during agent invocation.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Structured error response
        """
        logger.error(f"Error invoking agent: {error}", exc_info=True)
        return {
            "messages": [{"content": f"Agent invocation failed: {str(error)}", "type": "error"}],
            "error": str(error)
        }

    # Debug helper method, returns summary statistics about the state
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

# Create global agent instance for backward compatibility, easy import
# `agent`: AccountingAgent instance
agent = AccountingAgent(debug_enabled=Config.ENABLE_DEBUG_PRINTING)
# `agent_executor`: The compiled executable agent
agent_executor = agent.agent_executor

# Summary Flow:
# 1. User calls `agent.invoke()` with messages
# 2. Input is validated
# 3. Workflow starts at `agent` node (calls LLM)
# 4. Decision function checks if tools are needed
# 5. If tools needed, executes tools, loops back to agent
# 6. If final answer or error -> ends
# 7. Returns final result
