import os
import json
import time
import asyncio
import re
import redis
import weave
from openai import OpenAI, AsyncOpenAI
from mcp.server.fastmcp import FastMCP
from pydantic import Field, BaseModel
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, ".env"))

# Initialize Weave tracing
weave_client = weave.init("integration-tests")

# Connect to Redis (make sure Redis is running on localhost:6379)
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Initialize OpenAI client (ensure OPENAI_API_KEY is set in .env)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize MCP
mcp = FastMCP(
    "CoCo",
    dependencies=[
        "openai",
        "redis",
        "python-dotenv",
        "weave",
    ],
)


# Define models for message validation
class Message(BaseModel):
    role: str
    content: str


# Helper: extract messages and parameters from raw input
def extract_params_from_request(raw_input):
    """
    Extract messages, model, max_tokens, and temperature from the raw input.
    This handles cases where input might not be structured as expected.
    """
    # Default values
    model = "gpt-4o"
    max_tokens = 16000
    temperature = 0.0
    messages = []

    # Try to extract structured data
    if isinstance(raw_input, dict):
        # If it's already a dictionary, extract parameters
        if "messages" in raw_input:
            messages = raw_input.get("messages", [])
        if "model" in raw_input:
            model = raw_input.get("model", model)
        if "max_tokens" in raw_input:
            max_tokens = raw_input.get("max_tokens", max_tokens)
        if "temperature" in raw_input:
            temperature = raw_input.get("temperature", temperature)
    elif isinstance(raw_input, str):
        # If it's a string, treat it as a single user message
        messages = [{"role": "user", "content": raw_input}]

    # Ensure messages is a list of dictionaries with role and content
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]

    # Ensure each message has role and content
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            messages[i] = {"role": "user", "content": str(msg)}
        if "role" not in msg:
            msg["role"] = "user"
        if "content" not in msg:
            msg["content"] = ""

    return messages, model, max_tokens, temperature


# Helper: stream response chunks (simulate streaming)
async def stream_response(response):
    for chunk in response:
        if chunk and chunk.choices:
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                data = {
                    "id": chunk.id,
                    "object": "chat.completion.chunk",
                    "created": chunk.created,
                    "model": chunk.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "delta": {"content": choice.delta.content},
                            "finish_reason": choice.finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"


# Helper: simulate a discovery stream for initial messages
async def create_manual_oai_stream(message_content, model):
    chunk_size = 10  # characters per chunk
    message_id = f"chatcmpl-{int(time.time())}"

    for i in range(0, len(message_content), chunk_size):
        chunk = message_content[i : i + chunk_size]
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": (
                        None if i + chunk_size < len(message_content) else "stop"
                    ),
                }
            ],
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.05)

    yield "data: [DONE]\n\n"


# Helper function to discover intent from user messages
async def discover_intent(messages):
    # Extract the user's messages
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    if not user_messages:
        return {"intent": "unknown", "description": "No user messages found"}

    # Analyze the last user message to determine intent
    last_message = user_messages[-1]["content"]

    # Simple pattern matching for demo purposes
    # In a real implementation, you might use a more sophisticated approach
    if "blog" in last_message.lower() or "article" in last_message.lower():
        return {
            "intent": "blog_writing",
            "description": "User wants to write a blog article based on podcast transcript",
        }
    elif "code" in last_message.lower() or "program" in last_message.lower():
        return {
            "intent": "code_assistance",
            "description": "User needs help with coding or programming",
        }
    elif "question" in last_message.lower() or "help" in last_message.lower():
        return {
            "intent": "general_assistance",
            "description": "User has a general question or needs assistance",
        }
    else:
        return {
            "intent": "conversation",
            "description": "User is engaging in general conversation",
        }


# Helper function to retrieve context based on intent
async def retrieve_context(intent):
    # Return different context based on intent
    if intent == "blog_writing":
        try:
            with open("gd_episode_mike_knoop.txt", "r") as f:
                content = f.read()
            return {
                "context": content,
                "source": "gd_episode_mike_knoop.txt",
                "description": "Transcript from Gradient Dissent podcast with Mike Knoop",
            }
        except FileNotFoundError:
            return {
                "context": "Transcript file not found",
                "source": "error",
                "description": "The requested transcript could not be located",
            }
    elif intent == "code_assistance":
        return {
            "context": "Code assistance context would be provided here",
            "source": "code_examples",
            "description": "Examples and documentation for coding assistance",
        }
    else:
        return {
            "context": "No specific context available for this intent",
            "source": "general",
            "description": "General knowledge base",
        }


# Helper function to retrieve preferences based on intent
async def retrieve_preferences(intent):
    # Return different preferences based on intent
    if intent == "blog_writing":
        return {
            "preferences": {
                "style": "formal and precise",
                "format": "blog article",
                "tone": "informative and professional",
                "length": "medium to long",
                "audience": "technical professionals interested in AI",
            },
            "description": "Preferences for writing a technical blog article",
        }
    elif intent == "code_assistance":
        return {
            "preferences": {
                "style": "clear and concise",
                "format": "code with explanations",
                "language": "python",
                "comments": "detailed",
                "best_practices": True,
            },
            "description": "Preferences for providing code assistance",
        }
    else:
        return {
            "preferences": {
                "style": "conversational",
                "format": "direct answers",
                "tone": "helpful and friendly",
            },
            "description": "General preferences for conversation",
        }


# MCP Tool: Delegate to autonomous processing
@mcp.tool()
async def coco_delegate(
    raw_input: str,
) -> dict:
    """Process a request autonomously without human intervention."""
    print("***MCP Tool: Delegate***")

    # Extract parameters from the raw input
    messages, model, max_tokens, temperature = extract_params_from_request(raw_input)

    try:
        # Discover intent from the messages
        intent_response = await discover_intent(messages)
        intent = intent_response["intent"]

        # Retrieve context based on intent
        context_response = await retrieve_context(intent)

        # Retrieve preferences based on intent
        preference_response = await retrieve_preferences(intent)

        # Create a new message with context and preferences
        enriched_message = {
            "role": "user",
            "content": (
                f"Here is the transcript from the last gradient dissent episode with Mike Knoop. "
                f"Write a detailed blog article based on this transcript.\n\n"
                f"Context: {context_response['context']}\n\n"
                f"Preferences: {json.dumps(preference_response['preferences'], indent=2)}"
            ),
        }

        # Append new message and prepare for LLM call
        messages.append(enriched_message)

        # Call LLM for completion
        response = await async_openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Return formatted response
        return {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": model,
            "system_fingerprint": f"fp_{response.id[-8:]}",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    },
                    "finish_reason": response.choices[0].finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }
    except Exception as e:
        print(f"Error in coco_delegate: {str(e)}")
        return {
            "id": f"error-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Error: {str(e)}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


# MCP Tool: Collaborate with human review
@mcp.tool()
async def coco_collab(
    raw_input: str,
) -> dict:
    """Process a request with human collaboration via Redis queue."""
    print("***MCP Tool: Collaborate***")

    # TODO: currently tool usage difficulties with e.g. sonnet 3.5 when saying: "Can you help me draft my blogpost based on the last podcast episode with Coco collaboratively"
    # Extract parameters from the raw input
    messages, model, max_tokens, temperature = extract_params_from_request(raw_input)

    try:
        # Check if this is for cursor rewrite (which might timeout)
        cursor_rewrite = False
        for msg in messages:
            if (
                msg["role"] == "system"
                and "You are helping a colleague rewrite a piece of code"
                in msg["content"]
            ):
                cursor_rewrite = True
                print(
                    "WARNING: Cursor Re-write doesn't have enough timeout for collaboration."
                )
                break

        # Discover intent from the messages
        intent_response = await discover_intent(messages)
        intent = intent_response["intent"]

        # Retrieve context based on intent
        context_response = await retrieve_context(intent)

        # Retrieve preferences based on intent
        preference_response = await retrieve_preferences(intent)

        # Prepare request data with all context needed for human review
        request_data = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "collaboration_phase": "discovery",
            "intent": intent_response,
            "context": context_response,
            "preferences": preference_response,
        }

        # Push request to request queue
        redis_client.lpush("request_queue", json.dumps(request_data))

        # Retrieve response from response queue
        resp = redis_client.brpop("response_queue", timeout=300)
        if resp is None:
            raise Exception("Timed out waiting for human review")

        edited_data = json.loads(resp[1])
        last_message = edited_data["messages"][-1]["content"]

        # Return formatted response
        return {
            "id": f"chatcmpl-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": f"fp_{int(time.time())}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": last_message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
    except Exception as e:
        print(f"Error in coco_collab: {str(e)}")
        return {
            "id": f"error-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Error: {str(e)}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


if __name__ == "__main__":
    # Run the FastMCP server
    mcp.run(transport="stdio")
