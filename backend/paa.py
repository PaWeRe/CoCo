from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os
import uvicorn
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
import weave

# Init Weave tracing
client = weave.init("integration-tests")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Allow all origins
	allow_credentials=True,  # Allow cookies and authentication headers
	allow_methods=["*"],  # Allow all HTTP methods
	allow_headers=["*"],  # Allow all headers
)

# OpenAI API key (Ensure you set this as an environment variable for security)
load_dotenv("utils/.env")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Define request schemas
class Message(BaseModel):
	role: str
	content: str

class ChatCompletionRequest(BaseModel):
	model: str
	messages: List[Message]
	max_tokens: Optional[int] = 16000 #150
	temperature: Optional[float] = 0.0  # Changed default to 0.0 to match example request
	stream: Optional[bool] = True  # Changed default to True to match example request
	stop: Optional[List[str]] = None
	user: Optional[str] = None

@app.options("/{full_path:path}")
@weave.op
async def options_handler(full_path: str):
    response = JSONResponse(content="Preflight OK", status_code=200)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "OPTIONS, GET, POST, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Route for models list
@app.get("/v1/models")
@weave.op
async def list_models():
	try:
		response = client.models.list()
		return response
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

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
						"delta": {
							"content": choice.delta.content
						},
						"finish_reason": choice.finish_reason
					}]
				}
				yield f"data: {json.dumps(data)}\n\n"
	yield "data: [DONE]\n\n"

# Route for chat completions
@app.post("/v1/chat/completions")
@weave.op
async def chat_completions(request: ChatCompletionRequest):
	# Add a system message
	# with open('lukas_website.html', 'r') as f:
	# 	website_content = f.read()
	# website_message = Message(
	# 	role="user", 
	# 	content=f"Here is the current website of Lukas Biewald. Use this as a basis:\n {website_content}")
	# request.messages.append(website_message)


    # test interactive questions next -> for chat
	# what's the actual tech value here?

	with open('gd_episode_david_cahn.txt', 'r') as f:
		blog_content = f.read()
	blog_message = Message(
		role="user", 
		content=f"Here is the transcript from the last gradient descent episode with David Cahn. Write a detailed blog article from the perspective with key details from the transcript.:\n\n {blog_content}")
	request.messages.append(blog_message)
	# request.messages.insert(0, system_message)

	# Prepare messages in the correct format
	messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
	
	try:
		if request.stream:
			response = client.chat.completions.create(
				model=request.model,
				messages=messages_dict,
				max_tokens=request.max_tokens,
				temperature=request.temperature,
				stream=True,
				stop=request.stop,
				user=request.user
			)
			return StreamingResponse(
				stream_response(response),
				media_type="text/event-stream"
			)
		
		response = client.chat.completions.create(
			model=request.model,
			messages=messages_dict,
			max_tokens=request.max_tokens,
			temperature=request.temperature,
			stop=request.stop,
			user=request.user
		)

		# Format response to match required schema
		formatted_response = {
			"id": response.id,
			"object": "chat.completion",
			"created": response.created,
			"model": request.model,
			"system_fingerprint": f"fp_{response.id[-8:]}",
			"choices": [
				{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": response.choices[0].message.content
					},
					"finish_reason": response.choices[0].finish_reason
				}
			],
			"usage": {
				"prompt_tokens": response.usage.prompt_tokens,
				"completion_tokens": response.usage.completion_tokens,
				"total_tokens": response.usage.total_tokens
			}
		}
			
		return formatted_response
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# Main entry point
if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True)
