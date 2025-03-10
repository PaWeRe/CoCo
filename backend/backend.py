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

class IntentRequest(BaseModel):
    messages: list[Message]

class ContextRequest(BaseModel):
    intent: str

class PreferenceRequest(BaseModel):
    intent: str

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
async def create_manual_oai_stream(message_content, model):
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

# Endpoint: discover intent from user messages
@app.post("/v1/intent/discover")
@weave.op
async def discover_intent(request: IntentRequest):
    try:
        # Extract the user's messages
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            return {"intent": "unknown", "description": "No user messages found"}
        
        # Analyze the last user message to determine intent
        last_message = user_messages[-1].content
        
        # Simple pattern matching for demo purposes
        # In a real implementation, you might use a more sophisticated approach
        if "blog" in last_message.lower() or "article" in last_message.lower():
            return {
                "intent": "blog_writing",
                "description": "User wants to write a blog article based on podcast transcript"
            }
        elif "code" in last_message.lower() or "program" in last_message.lower():
            return {
                "intent": "code_assistance",
                "description": "User needs help with coding or programming"
            }
        elif "question" in last_message.lower() or "help" in last_message.lower():
            return {
                "intent": "general_assistance",
                "description": "User has a general question or needs assistance"
            }
        else:
            return {
                "intent": "conversation",
                "description": "User is engaging in general conversation"
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint: get context based on intent
@app.post("/v1/context/retrieve")
@weave.op
async def retrieve_context(request: ContextRequest):
    try:
        intent = request.intent
        
        # Return different context based on intent
        if intent == "blog_writing":
            try:
                with open('gd_episode_mike_knoop.txt', 'r') as f:
                    content = f.read()
                return {
                    "context": content,
                    "source": "gd_episode_mike_knoop.txt",
                    "description": "Transcript from Gradient Dissent podcast with Mike Knoop"
                }
            except FileNotFoundError:
                return {
                    "context": "Transcript file not found",
                    "source": "error",
                    "description": "The requested transcript could not be located"
                }
        elif intent == "code_assistance":
            return {
                "context": "Code assistance context would be provided here",
                "source": "code_examples",
                "description": "Examples and documentation for coding assistance"
            }
        else:
            return {
                "context": "No specific context available for this intent",
                "source": "general",
                "description": "General knowledge base"
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint: get preferences based on intent
@app.post("/v1/preferences/retrieve")
@weave.op
async def retrieve_preferences(request: PreferenceRequest):
    try:
        intent = request.intent
        
        # Return different preferences based on intent
        if intent == "blog_writing":
            return {
                "preferences": {
                    "style": "formal and precise",
                    "format": "blog article",
                    "tone": "informative and professional",
                    "length": "medium to long",
                    "audience": "technical professionals interested in AI"
                },
                "description": "Preferences for writing a technical blog article"
            }
        elif intent == "code_assistance":
            return {
                "preferences": {
                    "style": "clear and concise",
                    "format": "code with explanations",
                    "language": "python",
                    "comments": "detailed",
                    "best_practices": True
                },
                "description": "Preferences for providing code assistance"
            }
        else:
            return {
                "preferences": {
                    "style": "conversational",
                    "format": "direct answers",
                    "tone": "helpful and friendly"
                },
                "description": "General preferences for conversation"
            }
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
    if intent_delegate:
        print("***Intent: Delegate***")
        
        # Discover intent from the messages
        intent_request = IntentRequest(messages=request.messages)
        intent_response = await discover_intent(intent_request)
        intent = intent_response["intent"]
        
        # Retrieve context based on intent
        context_request = ContextRequest(intent=intent)
        context_response = await retrieve_context(context_request)
        
        # Retrieve preferences based on intent
        preference_request = PreferenceRequest(intent=intent)
        preference_response = await retrieve_preferences(preference_request)
        
        # Create a new message with context and preferences
        blog_message = Message(
            role="user",
            content=(
                f"Here is the transcript from the last gradient dissent episode with Mike Knoop. "
                f"Write a detailed blog article based on this transcript.\n\n"
                f"Context: {context_response['context']}\n\n"
                f"Preferences: {json.dumps(preference_response['preferences'], indent=2)}"
            )
        )
        request.messages.append(blog_message)
        model = request.model
        edited_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        max_tokens = request.max_tokens
        temperature = request.temperature

    # Collaboration branch for chat with human review
    elif intent_collaborate:
        print("***Intent: Collaborate***")
        if cursor_rewrite: 
            print("WARNING: Cursor Re-write doesn't have enough timeout for collaboratio. Try @coco_collab.")

        # TODO: considering adding backend-side conversation phase tracking and collaboration phases back in
        # possibly only checking for where the message with the trigger word is makes the most sense
        # conversation_id = hash(request.messages[0].content + request.messages[1].content)

        #try:
        # push request to request queue
        request_data = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "collaboration_phase": "discovery"
        }
        redis_client.lpush("request_queue", json.dumps(request_data))
    
        # retrieve response from response 
        resp = redis_client.brpop("response_queue", timeout=300)
        if resp is None:
            raise HTTPException(status_code=408, detail="Timed out waiting for human review")
        
        edited_data = json.loads(resp[1])
        last_message = edited_data["messages"][-1]["content"]
        
        model = request.model
        edited_messages = edited_data
        max_tokens = request.max_tokens
        temperature = request.temperature
        
        if request.stream:
            return StreamingResponse(
                create_manual_oai_stream(last_message, model),
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
                    "message": {
                        "role": "assistant",
                        "content": last_message
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                }
            }
            return formatted_response
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