# CoCo
LM mediator. Help LMs understand you.

## Quickstart
1. Create virtual env and set up environment 
    * Create a `.env` file with at least `OPENAI_API_KEY` and `WANDB_API_KEY`
    * `pip install requirements.txt` (or use any virtual environment manager like uv)
2. Start Redis DB - [Redis Quickstart](https://redis.io/learn/howtos/quick-start)
    * `docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest`
    * `docker exec -it redis-stack redis-cli`
    * `LRANGE request_queue 0 -1` and `LRANGE response_queue 0 -1` to check the status of the queues on Redis
3. Backend
    * `python backend.py` (use `python backend_fastmcp.py` or `python backend_fastapi.py`)
4. Frontend 
    * `python frontend.py` 

## Cursor Integration

CoCo supports two integration methods with Cursor:

### Method 1: FastMCP Integration (Recommended)

This method uses Anthropic's MCP (Model Context Protocol) for direct integration.

1. Make sure to have Redis DB and gradio frontend running (frontend only necessary for CoCo's collaboration mode)
   
2. Add the CoCo MCP server to your Cursor configuration
    * Edit your `~/.cursor/mcp.json` file to include:
    ```json
    {
      "mcpServers": {
        "CoCo": {
          "command": "/path/to/CoCo/.venv/bin/python",
          "args": [
            "/path/to/CoCo/backend/backend_fastmcp.py"
          ]
        }
      }
    }
    ```
    * Replace `/path/to/CoCo` with your actual CoCo installation path

3. Using CoCo with FastMCP in Cursor
    * Can only be used in agent mode (chat interface)
    * No specific decorators are required, but it helps to specify "CoCo" and whether you want to use it in collaborative mode or delegate mode
    * In-line edits are currently not supported in Cursor's agent mode

### Method 2: FastAPI + ngrok Integration

This method exposes CoCo as an OpenAI-compatible API that can be used with Cursor's OpenAI configuration.

1. Install additional requirements (fastapi, uvicorn, ngrok)
   
2. Make sure to have Redis DB and gradio frontend running (frontend only necessary for CoCo's collaboration mode)

1. Set up ngrok:
    * Run `ngrok http http://localhost:8000` 
    * In Cursor's Model section override OpenAI base URL with `<ngrok-address>/v1` 

2. Using CoCo with FastAPI in Cursor:
    * `@coco_delegate` can be used for in-line editing (command K) and works without the frontend
    * `@coco_collab` should be used in the chat (command L) and calls the mediator UI frontend

## Using CoCo

CoCo has two modes:
* **Delegation Mode**: Hand off a task to the agent and work on something else in the meantime
* **Collaboration Mode**: Jointly co-create and teach the agent/mediator through an interactive process

The idea is to use collaboration to train the agent on your preferences, then delegate similar tasks once the agent has learned how to handle them.