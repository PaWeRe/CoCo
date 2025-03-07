import os
import json
import time
import asyncio
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
import redis
import weave
import threading

# Load environment variables
load_dotenv(".env")

# Initialize Weave tracing
weave_client = weave.init("integration-tests")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Allow all origins
    allow_credentials=True,         # Allow cookies and authentication headers
    allow_methods=["*"],            # Allow all HTTP methods
    allow_headers=["*"],            # Allow all headers
)

# Connect to Redis (make sure Redis is running on localhost:6379)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize OpenAI client (ensure OPENAI_API_KEY is set in .env)
from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define request schemas
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 16000
    temperature: float = 0.0
    stream: bool = True
    stop: list[str] | None = None
    user: str | None = None

# In-memory state for tracking collaboration conversations
collaboration_state = {}

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
                    "choices": [{
                        "index": choice.index,
                        "delta": {"content": choice.delta.content},
                        "finish_reason": choice.finish_reason
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"

# Helper: simulate a discovery stream for initial messages
async def create_discovery_stream(message_content, model):
    chunk_size = 10  # characters per chunk
    message_id = f"chatcmpl-{int(time.time())}"
    
    for i in range(0, len(message_content), chunk_size):
        chunk = message_content[i:i+chunk_size]
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None if i + chunk_size < len(message_content) else "stop"
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.05)
    
    yield "data: [DONE]\n\n"

# OPTIONS route for CORS preflight
@app.options("/{full_path:path}")
@weave.op
async def options_handler(full_path: str):
    response = JSONResponse(content="Preflight OK", status_code=200)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "OPTIONS, GET, POST, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Endpoint: list available models
@app.get("/v1/models")
@weave.op
async def list_models():
    try:
        response = openai_client.models.list()
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint: chat completions with optional human review via Redis queues
@app.post("/v1/chat/completions")
@weave.op
async def chat_completions(request: ChatCompletionRequest):
    # Determine collaboration mode based on message content
    cursor_rewrite = "You are helping a colleague rewrite a piece of code" in request.messages[0].content
    cursor_chat = "You are happy to help answer any questions that the user has" in request.messages[0].content

    # Check intent patterns for delegate or collaborate
    intent_delegate = False
    intent_collaborate = False
    for msg in reversed(request.messages):
        if re.search(r"(^|\s)@coco_delegate(\s|$)", msg.content):
            intent_delegate = True
            break
        elif re.search(r"(^|\s)@coco_collab(\s|$)", msg.content):
            intent_collaborate = True
            break

    # Collaboration branch for re-writing
    if cursor_rewrite and intent_delegate:
        print("***Cursor: Re-write, Intent: Delegate***")
        with open('gd_episode_david_cahn.txt', 'r') as f:
            blog_content = f.read()
        blog_message = Message(
            role="user",
            content=f"Here is the transcript from the last gradient descent episode with David Cahn. Write a detailed blog article from the perspective with key details from the transcript.:\n\n {blog_content}"
        )
        request.messages.append(blog_message)
        model = request.model
        edited_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        max_tokens = request.max_tokens
        temperature = request.temperature

    # Collaboration branch for chat with human review
    elif cursor_chat and intent_collaborate:
        print("***Cursor: Chat, Intent: Collaborate***")
        conversation_id = hash(request.messages[0].content + request.messages[1].content)
        model = request.model
        edited_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        max_tokens = request.max_tokens
        temperature = request.temperature

        if conversation_id not in collaboration_state:
            print(f"***START CONVERSATION, ID: {conversation_id}***")
            print("***DISCOVERY***")
            collaboration_state[conversation_id] = "Discovery"
            discovery_message = {
                "role": "assistant",
                "content": (
                    "I'd like to help you with your question. To better assist you, "
                    "could you please provide more information about:\n\n"
                    "1. What specific problem are you trying to solve?\n"
                    "2. What have you tried so far?\n"
                    "3. Do you have any preferences for the solution approach?\n"
                    "4. Are there any constraints or requirements I should be aware of?"
                )
            }
            if request.stream:
                return StreamingResponse(
                    create_discovery_stream(discovery_message["content"], model),
                    media_type="text/event-stream"
                )
            else:
                formatted_response = {
                    "id": f"chatcmpl-{time.time()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "system_fingerprint": f"fp_{int(time.time())}",
                    "choices": [{
                        "index": 0,
                        "message": discovery_message,
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150
                    }
                }
                return formatted_response
        else:
            print(f"***CONTINUE CONVERSATION, ID: {conversation_id}***")
            current_phase = collaboration_state[conversation_id]
            if current_phase == "Discovery":
                print("***DISCOVERY***")
                collaboration_state[conversation_id] = "Generation"
                try:
                    request_data = {
                        "messages": edited_messages,
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "collaboration_phase": "Generation"
                    }
                    redis_client.lpush("request_queue", json.dumps(request_data))
                    resp = redis_client.brpop("response_queue", timeout=300)
                    if resp is None:
                        raise HTTPException(status_code=408, detail="Timed out waiting for human review")
                    edited_data = json.loads(resp[1])
                    edited_messages = edited_data["messages"]
                    model = edited_data["model"]
                    max_tokens = edited_data["max_tokens"]
                    temperature = edited_data["temperature"]
                except Exception as e:
                    raise HTTPException(status_code=408, detail="Timed out waiting for human review")
            else:  # Generation phase
                print("***GENERATION***")
                del collaboration_state[conversation_id]
                try:
                    request_data = {
                        "messages": edited_messages,
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "collaboration_phase": "Final Review"
                    }
                    redis_client.lpush("request_queue", json.dumps(request_data))
                    resp = redis_client.brpop("response_queue", timeout=300)
                    if resp is None:
                        raise HTTPException(status_code=408, detail="Timed out waiting for human review")
                    edited_data = json.loads(resp[1])
                    edited_messages = edited_data["messages"]
                    model = edited_data["model"]
                    max_tokens = edited_data["max_tokens"]
                    temperature = edited_data["temperature"]
                except Exception as e:
                    raise HTTPException(status_code=408, detail="Timed out waiting for human review")
    # Baseline: no human review intervention
    else:
        print(f"***Baseline, cursor_rewrite: {cursor_rewrite}, cursor_chat: {cursor_chat}, "
              f"intent_delegate: {intent_delegate}, intent_collaborate: {intent_collaborate}***")
        model = request.model
        edited_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        max_tokens = request.max_tokens
        temperature = request.temperature

    # Call the LLM (baseline) and stream or return the full response
    try:
        if request.stream:
            response = openai_client.chat.completions.create(
                model=model,
                messages=edited_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stop=request.stop,
                user=request.user
            )
            return StreamingResponse(
                stream_response(response),
                media_type="text/event-stream"
            )
        response = openai_client.chat.completions.create(
            model=model,
            messages=edited_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=request.stop,
            user=request.user
        )
        formatted_response = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": model,
            "system_fingerprint": f"fp_{response.id[-8:]}",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                },
                "finish_reason": response.choices[0].finish_reason
            }],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        return formatted_response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True)