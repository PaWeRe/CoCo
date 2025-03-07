import json
import redis
import gradio as gr
from gradio import update

# Connect to Redis (ensure the same connection settings as in backend.py)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def submit_edited_message(messages_text, model, max_tokens, temperature, collaboration_phase):
    try:
        messages = json.loads(messages_text)
        edited_data = {
            "messages": messages,
            "model": model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "collaboration_phase": collaboration_phase
        }
        # Push the human-edited response into the Redis response queue
        redis_client.lpush("response_queue", json.dumps(edited_data))
        return "Message submitted for processing"
    except Exception as e:
        return f"Error: {str(e)}"

def check_queue():
    # Attempt to retrieve a pending request from the Redis request queue
    req_data = redis_client.rpop("request_queue")
    if req_data:
        req_data = json.loads(req_data)
        return (
            json.dumps(req_data.get("messages", []), indent=2),
            req_data.get("model", ""),
            req_data.get("max_tokens", ""),
            req_data.get("temperature", ""),
            req_data.get("collaboration_phase", "Discovery")
        )
    return update(), update(), update(), update(), update()

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Human Review Interface")
        with gr.Row():
            with gr.Column():
                messages_text = gr.Textbox(label="Messages (JSON format)", lines=10)
                model = gr.Textbox(label="Model")
                max_tokens = gr.Number(label="Max Tokens")
                temperature = gr.Number(label="Temperature")
                collaboration_phase = gr.Radio(
                    ["Discovery", "Generation", "Final Review"],
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
                refresh_btn = gr.Button("Check for New Requests")
                refresh_btn.click(
                    fn=check_queue,
                    inputs=None,
                    outputs=[messages_text, model, max_tokens, temperature, collaboration_phase]
                )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    create_gradio_interface()