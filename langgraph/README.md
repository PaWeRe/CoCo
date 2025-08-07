# CoCo Blog Collaboration Tool with LangGraph

A multi-agent system for collaborative blog post creation built with LangGraph.

## Overview

This tool provides an interactive workflow for blog post creation with the following phases:

1. **Discovery Phase**: Understand user intent, preferences, and gather context
2. **Execution Phase**: Draft the blog post content based on the gathered information
3. **Verification Phase**: Review, get feedback, and refine the draft
4. **Finalization Phase**: Prepare for integration into the website

The system uses a multi-agent architecture with the following specialized agents:

- **Mediator Agent**: Orchestrates the workflow and guides the user through each phase
- **Retrieval Agent**: Gathers relevant context and information from available sources
- **Drafting Agent**: Creates and refines blog post drafts
- **Reviewing Agent**: Analyzes drafts and provides feedback for improvement
- **Integration Agent**: Prepares the final blog post for website integration

## Features

- **Human-in-the-loop collaboration**: Get human input at any step of the process
- **Checkpointing**: Save state between sessions and enable time travel to previous points
- **Multi-turn conversations**: Support for extended interactions within the same context
- **Tool-based architecture**: Agents can use specialized tools to accomplish tasks

## Installation

1. Clone the repository
2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -e .
```
4. Create a `.env` file in the `backend` directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Run the CLI interface:
```bash
python main.py
```

Special commands during the CLI session:
- `time-travel`: Go back to a previous checkpoint
- `list-checkpoints`: View all available checkpoints
- `exit`: Quit the application

## Visualization

To visualize the workflow graph:
```bash
python visualize.py
```

## Implementation Details

This project utilizes LangGraph's StateGraph for building a stateful multi-agent system:

- The state is preserved between interactions using the MemorySaver checkpointer
- LangGraph's ToolNode and tools_condition are used for specialized agent integration
- Human-in-the-loop interactions are implemented using LangGraph's interrupt mechanism
- Time travel allows revisiting previous states and exploring alternative paths

## License

MIT 