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
import re

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

# Track collaboration state
collaboration_state = {}

# Gradio interface for human review
def review_message(messages_text, model, max_tokens, temperature, collaboration_phase):
    # Display the messages and allow editing
    return messages_text, model, max_tokens, temperature, collaboration_phase

def submit_edited_message(messages_text, model, max_tokens, temperature, collaboration_phase):
    # Parse the edited messages back to the expected format
    try:
        messages = json.loads(messages_text)
        # Put the edited data in the response queue
        response_queue.put({
            "messages": messages,
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "collaboration_phase": collaboration_phase
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
            req_data["temperature"],
            req_data.get("collaboration_phase", "Discovery")
        )
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

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
                collaboration_phase = gr.Radio(
                    ["Discovery", "Generation"], 
                    label="Collaboration Phase", 
                    value="Discovery"
                )
                
                submit_btn = gr.Button("Submit Edited Message")
                status = gr.Textbox(label="Status")
                
                submit_btn.click(
                    fn=submit_edited_message,
                    inputs=[messages_text, model, max_tokens, temperature, collaboration_phase],
                    outputs=status
                )
                
                # Add a refresh button to manually check for new requests
                refresh_btn = gr.Button("Check for New Requests")
                refresh_btn.click(
                    fn=check_queue,
                    inputs=None,
                    outputs=[messages_text, model, max_tokens, temperature, collaboration_phase]
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

# Route for options requests
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

# Function to create a simulated stream for discovery messages
async def create_discovery_stream(message_content, model):
	chunk_size = 10  # Characters per chunk
	message_id = f"chatcmpl-{int(time.time())}"
	
	# Send the message in chunks to simulate streaming
	for i in range(0, len(message_content), chunk_size):
		chunk = message_content[i:i+chunk_size]
		data = {
			"id": message_id,
			"object": "chat.completion.chunk",
			"created": int(time.time()),
			"model": model,
			"choices": [{
				"index": 0,
				"delta": {
					"content": chunk
				},
				"finish_reason": None if i + chunk_size < len(message_content) else "stop"
			}]
		}
		yield f"data: {json.dumps(data)}\n\n"
		await asyncio.sleep(0.05)  # Small delay to simulate real streaming
	
	yield "data: [DONE]\n\n"

# Route for chat completions
@app.post("/v1/chat/completions")
@weave.op
async def chat_completions(request: ChatCompletionRequest):
	# whether user wants to rewrite or chat in cursor
	cursor_rewrite = "You are helping a colleague rewrite a piece of code" in request.messages[0].content
	cursor_chat = "You are happy to help answer any questions that the user has" in request.messages[0].content

	# whether user wants to delegate or collaborate
	# Check for intent patterns in all messages, starting from the most recent
	intent_delegate = False
	intent_collaborate = False
	for msg in reversed(request.messages):
		if re.search(r"(^|\s)@coco_delegate(\s|$)", msg.content):
			intent_delegate = True
			break
		elif re.search(r"(^|\s)@coco_collab(\s|$)", msg.content):
			intent_collaborate = True
			break

	# paa enhancement
	# NOTE: for now we consider collaboration to be possible only in chat bc of timeout 
	# and delegate only in rewrite because no chatting should be required
	if cursor_rewrite and intent_delegate: 
		print("***Cursor: Re-write, Intent: Delegate***")

		# TODO: create weavified ops for intent, preferences, context that are called in here
		with open('gd_episode_david_cahn.txt', 'r') as f:
			blog_content = f.read()
		blog_message = Message(
			role="user", 
			content=f"Here is the transcript from the last gradient descent episode with David Cahn. Write a detailed blog article from the perspective with key details from the transcript.:\n\n {blog_content}")
		request.messages.append(blog_message)
		# request.messages.insert(0, system_message)

		# Prepare messages in the correct format
		model = request.model
		edited_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
		max_tokens = request.max_tokens
		temperature = request.temperature

	if cursor_chat and intent_collaborate:
		print("***Cursor: Chat, Intent: Collaborate***")

		# Generate a unique conversation ID based on the first few messages
		conversation_id = hash(request.messages[0].content + request.messages[1].content)
		
		# Prepare messages in the correct format
		model = request.model
		edited_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
		max_tokens = request.max_tokens
		temperature = request.temperature

		# Check if this is a new conversation or continuing one
		if conversation_id not in collaboration_state:
			print(f"***START CONVERSATION, ID: {conversation_id}***")
			print(f"***DISCOVERY***")	
			# New conversation - start with Discovery phase
			collaboration_state[conversation_id] = "Discovery"
			
			# TODO: do this manully for now - PAA can also generate these
			# Add discovery questions to the messages
			discovery_message = {
				"role": "assistant", 
				"content": "I'd like to help you with your question. To better assist you, could you please provide more information about:\n\n1. What specific problem are you trying to solve?\n2. What have you tried so far?\n3. Do you have any preferences for the solution approach?\n4. Are there any constraints or requirements I should be aware of?"
			}
			
			# Return discovery questions as a stream to match expected format
			if request.stream:
				return StreamingResponse(
					create_discovery_stream(discovery_message["content"], model),
					media_type="text/event-stream"
				)
			else:
				# For non-streaming requests
				formatted_response = {
					"id": f"chatcmpl-{time.time()}",
					"object": "chat.completion",
					"created": int(time.time()),
					"model": model,
					"system_fingerprint": f"fp_{int(time.time())}",
					"choices": [
						{
							"index": 0,
							"message": discovery_message,
							"finish_reason": "stop"
						}
					],
					"usage": {
						"prompt_tokens": 100,  # Estimated
						"completion_tokens": 50,  # Estimated
						"total_tokens": 150  # Estimated
					}
				}
				return formatted_response
			
		else:
			print(f"***CONTINUE CONVERSATION, ID: {conversation_id}***")
			# Continuing conversation - check current phase
			current_phase = collaboration_state[conversation_id]
			
			if current_phase == "Discovery":
				print(f"***DISCOVERY***")
				# Move to Generation phase after discovery
				collaboration_state[conversation_id] = "Generation"
				
				# Send request to Gradio for human review with both phases
				try:
					request_data = {
						"messages": edited_messages,
						"model": model,
						"max_tokens": max_tokens,
						"temperature": temperature,
						"collaboration_phase": "Generation"
					}
					request_queue.put(request_data)
					
					# Wait for human to review and edit the message
					edited_data = response_queue.get(timeout=300)  # 5 minute timeout
					
					# Use the edited data
					edited_messages = edited_data["messages"]
					model = edited_data["model"]
					max_tokens = edited_data["max_tokens"]
					temperature = edited_data["temperature"]
					
				except queue.Empty:
					raise HTTPException(status_code=408, detail="Request timed out waiting for human review")
			
			else:  # Generation phase
				print(f"***GENERATION***")
				# Clean up the state as we're done with this conversation
				del collaboration_state[conversation_id]
				
				# Send request to Gradio for final human review
				try:
					request_data = {
						"messages": edited_messages,
						"model": model,
						"max_tokens": max_tokens,
						"temperature": temperature,
						"collaboration_phase": "Final Review"
					}
					request_queue.put(request_data)
					
					# Wait for human to review and edit the message
					edited_data = response_queue.get(timeout=300)  # 5 minute timeout
					
					# Use the edited data
					edited_messages = edited_data["messages"]
					model = edited_data["model"]
					max_tokens = edited_data["max_tokens"]
					temperature = edited_data["temperature"]
					
				except queue.Empty:
					raise HTTPException(status_code=408, detail="Request timed out waiting for human review")
		
	# baseline without any intervention
	else: 
		print(f"***Baseline, cursor_rewrite: {cursor_rewrite}, cursor_chat: {cursor_chat}, intent_delegate: {intent_delegate}, intent_collaborate: {intent_collaborate}***")
		model = request.model
		edited_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
		max_tokens = request.max_tokens
		temperature = request.temperature
	
	# Baseline LLM call 
	try:
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
	
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))

# TODO: split up into different scripts once frontend is done
if __name__ == "__main__":
	# Start Gradio Frontend
	create_gradio_interface()

	# Start Uvicorn Backend
	uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True)
