from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage

import os
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))

# Add memory checkpoint
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}


def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Use this to ask the human for assistance."""
    human_response = interrupt(
        {"query": "Is this correct?", "name": name, "birthday": birthday}
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        human_response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)


# Tool definitions
tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]  # context-specific metadata
    name: str
    birthday: str


graph_builder = StateGraph(State)
# add a node that uses the tool
tool_node = ToolNode(tools=tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str, expert_input: bool = False):
    if expert_input:
        events = graph.stream(
            Command(resume={"data": user_input}), config, stream_mode="values"
        )
    else:
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


while True:
    print(graph.get_state(config).next)
    try:
        if "tools" in graph.get_state(config).next:
            user_input = input("Expert: ")
            expert_input = True
        else:
            user_input = input("User: ")
            expert_input = False
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

        stream_graph_updates(user_input, expert_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input, expert_input)
        break
