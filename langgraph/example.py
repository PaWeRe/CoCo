"""
Example script demonstrating the CoCo Blog Collaboration Tool with LangGraph.

This script shows how to initialize the graph, run a single turn,
and use the checkpointing and time travel features.
"""

import asyncio
from typing import Dict, Any
import os
import sys
from dotenv import load_dotenv

# Try to load environment variables from multiple possible locations
env_paths = ["../backend/.env", "./.env", "../.env", "../../.env"]

env_loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
        env_loaded = True
        break

if not env_loaded:
    print(
        "Warning: Could not find .env file. Please ensure your API keys are set in environment variables."
    )

from graph import create_memory_graph, get_default_state


async def run_example():
    """
    Run a simple example of the blog collaboration tool.
    """
    print("=== CoCo Blog Collaboration Tool Example ===\n")

    # Create the graph with memory checkpointing
    graph = create_memory_graph()

    # Create initial state with a user message
    state = get_default_state()
    state["messages"] = [
        {
            "role": "user",
            "content": "I want to write a blog post about using LangGraph for building AI agents.",
        }
    ]

    # Create a thread ID for this conversation
    config = {"configurable": {"thread_id": "example-thread"}}

    print(
        "Initial user message: I want to write a blog post about using LangGraph for building AI agents.\n"
    )
    print("Running the graph for one turn...\n")

    # Run the graph for one turn
    checkpoints = []
    events = []

    for event in graph.stream(state, config, stream_mode="updates"):
        # Store checkpoints
        if "checkpoint" in event:
            checkpoints.append(event["checkpoint"])

        # Print messages from assistant or tools
        if "messages" in event and event["messages"] and len(event["messages"]) > 0:
            for msg in event["messages"]:
                if msg["role"] == "assistant":
                    print(f"Assistant: {msg['content']}\n")
                elif msg["role"] == "tool":
                    print(
                        f"[Tool: {msg.get('tool_call_id', 'unknown')}]: {msg['content']}\n"
                    )

        # Store events for state updates
        events.append(event)

    # Update state with events
    for event in events:
        for key, value in event.items():
            if key != "checkpoint":
                state[key] = value

    print("\n=== Checkpoints Created ===")
    for i, cp in enumerate(checkpoints):
        phase = cp.state.get("phase", "unknown")
        print(f"{i}: Checkpoint at phase: {phase}")

    print("\n=== Time Travel Example ===")
    if checkpoints:
        print("Now we'll demonstrate time travel by going back to the first checkpoint")
        # Use the first checkpoint's config
        first_cp = checkpoints[0]
        time_travel_config = {"configurable": first_cp.config["configurable"]}

        print("Continuing from the checkpoint with a new user message...\n")

        # Create a new user message for the time travel example
        time_travel_msg = "Actually, I'd like to focus on how LangGraph enables time travel features for agent workflows."
        new_state = {"messages": [{"role": "user", "content": time_travel_msg}]}

        print(f"New user message: {time_travel_msg}\n")

        # Stream the response after time travel
        for event in graph.stream(new_state, time_travel_config, stream_mode="updates"):
            if "messages" in event and event["messages"] and len(event["messages"]) > 0:
                for msg in event["messages"]:
                    if msg["role"] == "assistant":
                        print(f"Assistant (after time travel): {msg['content']}\n")
                    elif msg["role"] == "tool":
                        print(
                            f"[Tool (after time travel): {msg.get('tool_call_id', 'unknown')}]: {msg['content']}\n"
                        )

    print("\n=== Example Complete ===")
    print("For interactive usage, run: python main.py")


if __name__ == "__main__":
    asyncio.run(run_example())
