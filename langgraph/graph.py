from typing import Dict, Any, List, Tuple, Optional, Annotated, TypedDict, cast
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from schemas import State
from agents import (
    create_mediator_agent,
    retrieval_agent,
    drafting_agent,
    reviewing_agent,
    integration_agent,
    human_assistance_tool,
    get_tools,
)


class ConfigSchema(TypedDict):
    db_id: int
    model: str


# Add memory checkpoint
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}

# Define specialized agents as tools
tools = [
    retrieval_agent,
    drafting_agent,
    reviewing_agent,
    integration_agent,
    human_assistance_tool,
]
tool_node = ToolNode(tools=tools)
mediator = create_mediator_agent(tools)  # Create mediator agent with tools

graph_builder = StateGraph(State, config_schema=ConfigSchema)
graph_builder.set_entry_point("mediator")
graph_builder.add_node("mediator", mediator)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("mediator", tools_condition)
graph_builder.add_edge("tools", "mediator")
graph = graph_builder.compile(checkpointer=memory)
