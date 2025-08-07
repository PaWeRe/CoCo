import sys
import os
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import uuid

# Try to load environment variables from multiple possible locations
env_paths = ["../backend/.env", "./.env", "../.env", "../../.env"]

env_loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        env_loaded = True
        break

# Import our graph
from graph import create_memory_graph, get_default_state

# TODO: create agent gui where you can go down tree branches and up with arrow keys


async def run_cli():
    """
    Run the blog collaboration workflow in CLI mode with checkpointing.
    """
    print("=== CoCo Blog Collaboration Assistant ===")
    print("Enter your blog post request, or type 'exit' to quit.")
    print("Special commands:")
    print("  'time-travel': Go back to a previous checkpoint")
    print("  'list-checkpoints': List available checkpoints")
    print("  'exit': Quit the application\n")

    # Create graph with memory checkpointing
    graph = create_memory_graph()

    # Generate a thread ID for this conversation
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Get initial user input
    try:
        initial_input = input("What kind of blog post would you like to create? ")
        if initial_input.lower() == "exit":
            print("Exiting...")
            return
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        return

    # Create initial state with the first user message
    state = get_default_state()
    state["messages"] = [{"role": "user", "content": initial_input}]

    # Main interaction loop
    while True:
        try:
            # Run the graph with current state and config
            checkpoints = []

            for event in graph.stream(state, config, stream_mode="updates"):
                # Store checkpoints during streaming
                if "checkpoint" in event:
                    checkpoints.append(event["checkpoint"])

                # Print messages from assistant
                if (
                    "messages" in event
                    and event["messages"]
                    and len(event["messages"]) > 0
                ):
                    for msg in event["messages"]:
                        if msg["role"] == "assistant":
                            print(f"\nAssistant: {msg['content']}")
                        elif msg["role"] == "tool":
                            print(
                                f"\n[Tool: {msg.get('tool_call_id', 'unknown')}]: {msg['content']}"
                            )

                # Update the state with the event
                for key, value in event.items():
                    if key != "checkpoint":
                        state[key] = value

            # After the graph execution, check if we need to continue or exit
            print("\n--- Enter your response, or a special command ---")
            user_input = input("You: ")

            if user_input.lower() == "exit":
                print("Exiting...")
                break

            elif user_input.lower() == "time-travel":
                # Time travel functionality
                if not checkpoints:
                    print(
                        "No checkpoints available yet. Continue the conversation first."
                    )
                    continue

                print("\nAvailable checkpoints:")
                for i, cp in enumerate(checkpoints):
                    # Show a short description of each checkpoint
                    phase = cp.state.get("phase", "unknown")
                    print(f"{i}: Checkpoint at phase: {phase}")

                try:
                    cp_idx = int(input("Enter checkpoint number to go back to: "))
                    if 0 <= cp_idx < len(checkpoints):
                        selected_cp = checkpoints[cp_idx]
                        # Use the selected checkpoint's config to resume from that point
                        config = {"configurable": selected_cp.config["configurable"]}
                        state = None  # We'll use the saved state from the checkpoint
                        print(f"Traveling back to checkpoint {cp_idx}...")
                    else:
                        print("Invalid checkpoint number.")
                except ValueError:
                    print("Please enter a valid number.")

            elif user_input.lower() == "list-checkpoints":
                if not checkpoints:
                    print("No checkpoints available yet.")
                else:
                    print("\nAvailable checkpoints:")
                    for i, cp in enumerate(checkpoints):
                        phase = cp.state.get("phase", "unknown")
                        print(f"{i}: Checkpoint at phase: {phase}")

            else:
                # Add the user's response to the state
                state["messages"].append({"role": "user", "content": user_input})

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            break


if __name__ == "__main__":
    asyncio.run(run_cli())
