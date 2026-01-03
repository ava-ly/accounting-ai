# data structure and routing logic

"""Type definitions for the accounting AI agent."""

from typing import Literal, Optional

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Extended state for the accounting agent."""

    step_count: Optional[int] = None
    debug_enabled: bool = True


RouteDecision = Literal["tools", "end"]
