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
import gradio as gr
import threading
import queue
import time

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
load_dotenv(".env")

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

# Create queues for communication between FastAPI and Gradio
request_queue = queue.Queue()
response_queue = queue.Queue()

# Gradio interface for human review
def review_message(messages_text, model, max_tokens, temperature):
    # Display the messages and allow editing
    return messages_text, model, max_tokens, temperature

def submit_edited_message(messages_text, model, max_tokens, temperature):
    # Parse the edited messages back to the expected format
    try:
        messages = json.loads(messages_text)
        # Put the edited data in the response queue
        response_queue.put({
            "messages": messages,
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature)
        })
        return "Message submitted for processing"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to periodically check the queue
def check_queue():
    if not request_queue.empty():
        req_data = request_queue.get()
        return (
            json.dumps(req_data["messages"], indent=2),
            req_data["model"],
            req_data["max_tokens"],
            req_data["temperature"]
        )
    return gr.update(), gr.update(), gr.update(), gr.update()

# Create and launch Gradio interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Human Review Interface")
                messages_text = gr.Textbox(label="Messages (JSON format)", lines=10)
                model = gr.Textbox(label="Model")
                max_tokens = gr.Number(label="Max Tokens")
                temperature = gr.Number(label="Temperature")
                
                submit_btn = gr.Button("Submit Edited Message")
                status = gr.Textbox(label="Status")
                
                submit_btn.click(
                    fn=submit_edited_message,
                    inputs=[messages_text, model, max_tokens, temperature],
                    outputs=status
                )
                
                # Add a refresh button to manually check for new requests
                refresh_btn = gr.Button("Check for New Requests")
                refresh_btn.click(
                    fn=check_queue,
                    inputs=None,
                    outputs=[messages_text, model, max_tokens, temperature]
                )
        
        # Start a background thread to periodically update the UI
        def polling_thread():
            while True:
                time.sleep(1)  # Check every second
                if not request_queue.empty():
                    # We can't directly update the UI from this thread,
                    # but we can trigger a refresh when the user interacts next
                    pass
    
    # Launch Gradio in a separate thread
    threading.Thread(target=lambda: demo.launch(server_name="0.0.0.0", server_port=7860, share=True), daemon=True).start()
    
    # Start the polling thread
    threading.Thread(target=polling_thread, daemon=True).start()

# Start Gradio interface
create_gradio_interface()

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
		# Send request to Gradio for human review
		request_data = {
			"messages": messages_dict,
			"model": request.model,
			"max_tokens": request.max_tokens,
			"temperature": request.temperature
		}
		request_queue.put(request_data)
		
		# Wait for human to review and edit the message
		# This is a blocking operation - in a production system you might want to use async patterns
		edited_data = response_queue.get(timeout=300)  # 5 minute timeout
		
		# Use the edited data
		edited_messages = edited_data["messages"]
		model = edited_data["model"]
		max_tokens = edited_data["max_tokens"]
		temperature = edited_data["temperature"]
		
		if request.stream:
			response = client.chat.completions.create(
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
		
		response = client.chat.completions.create(
			model=model,
			messages=edited_messages,
			max_tokens=max_tokens,
			temperature=temperature,
			stop=request.stop,
			user=request.user
		)

		# Format response to match required schema
		formatted_response = {
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
	except queue.Empty:
		raise HTTPException(status_code=408, detail="Request timed out waiting for human review")
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# Main entry point
if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True)
