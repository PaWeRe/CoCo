import json
import redis
import gradio as gr
import openai
import time
import os
from dotenv import load_dotenv

# Set your OpenAI API key (alternatively, use an environment variable)
# Load environment variables
load_dotenv("../backend/.env")
openai.api_key = os.getenv('OPENAI_API_KEY')

# -------------------------
# Redis Connection Logic
# -------------------------
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# -------------------------
# Helper Functions for OpenAI Calls
# -------------------------
def call_openai_api(prompt: str) -> str:
    """
    Calls OpenAI's Chat API using the new interface.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
        )
        output = response.choices[0].message.content.strip()
        return output
    except Exception as e:
        return f"Error calling OpenAI: {str(e)}"

def update_process_visualization(phase: str) -> str:
    """
    Returns a visually enhanced Markdown view of the process steps with colors and arrows.
    """
    steps = ["Discovery", "Execution", "Verification", "Completed"]
    colors = {
        "discovery": "#E6F7FF",  # Light blue
        "execution": "#F6FFED",  # Light green
        "verification": "#FFF7E6", # Light orange
        "completed": "#F9F0FF"   # Light purple
    }
    
    viz = ""
    for i, step in enumerate(steps):
        if step.lower() == phase:
            # Current step with highlighted box
            viz += f"<span style='background-color: {colors[phase.lower()]}; padding: 5px 10px; border: 2px solid #1890FF; border-radius: 4px; font-weight: bold; color: black;'>{step}</span>"
        else:
            # Inactive step
            viz += f"<span style='background-color: #F5F5F5; padding: 5px 10px; border: 1px solid #D9D9D9; border-radius: 4px; color: black;'>{step}</span>"
        
        # Add arrow between steps (except after the last step)
        if i < len(steps) - 1:
            viz += " <span style='color: #1890FF; font-weight: bold;'>→</span> "
    
    return viz

# -------------------------
# Helper Functions for Loading Local Defaults
# -------------------------
def parse_defaults_file(filename: str):
    """
    Parses a local defaults text file and returns a tuple (preferences, context).
    The file must have sections starting with 'Context:' and 'Preferences:'.
    Lines beginning with '-' are treated as items.
    """
    preferences = []
    context = []
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
        mode = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("context:"):
                mode = "context"
                continue
            elif line.lower().startswith("preferences:"):
                mode = "preferences"
                continue
            elif line.startswith("-"):
                item = line.lstrip("-").strip()
                if mode == "context":
                    context.append(item)
                elif mode == "preferences":
                    preferences.append(item)
        return preferences, context
    except Exception as e:
        return [], []

def load_local_defaults(user_input: str):
    """
    Based on keyword matching in the user_input, load the appropriate default preferences and context.
    """
    ui = user_input.lower()
    if "blog" in ui or "website" in ui or "post" in ui:
        filename = "blog_defaults.txt"
    elif "follow" in ui or "mike" in ui or "email" in ui:
        filename = "email_defaults.txt"
    else:
        return [], []
    return parse_defaults_file(filename)

# -------------------------
# Redis Request Fetching
# -------------------------
def fetch_request_action():
    """
    Fetch a request from Redis. The request JSON should contain messages and collaboration_phase.
    """
    # req_data = redis_client.rpop("request_queue")
    req_data = redis_client.lindex("request_queue", -1)
    if req_data:
        req_data = json.loads(req_data)
        messages = req_data.get("messages", [])
        collaboration_phase = req_data.get("collaboration_phase", "")
        model = req_data.get("model", "")
        max_tokens = req_data.get("max_tokens", 0)
        temperature = req_data.get("temperature", 0.0)
        
        # TODO: think about how to get all the cursor message in
        # Extract the last user message as the initial question
        initial_question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                initial_question = msg.get("content", "")
                break
                
        # For now, we'll keep preferences and context empty
        preferences = []
        context_info = []
        
        return initial_question, preferences, context_info
    else:
        return "", [], []

# -------------------------
# Process Function: process_input
# -------------------------
def process_input(user_input, prefs, context, phase, disc_sub, analysis_res, clarified_intent, draft_editor, conversation):
    """
    Processes the user input based on the current phase and subphase.
    
    State Variables:
      - phase: "discovery", "execution", "verification", or "completed"
      - disc_sub: for Discovery, either "analysis" or "clarification"
      - analysis_res: stores the initial analysis output (from Discovery analysis)
      - clarified_intent: final intent determined after clarification
      - draft_editor: contents of the joint editor (draft text)
      - conversation: the accumulated conversation summary
    
    Returns:
      Updated (phase, disc_sub, analysis_res, clarified_intent, draft_editor, conversation, prefill_input, prefs, context)
    """
    # DISCOVERY PHASE
    if phase == "discovery":
        if disc_sub == "analysis":
            # Load defaults if prefs or context are empty.
            if not prefs or not context:
                default_p, default_c = load_local_defaults(user_input)
                if not prefs:
                    prefs = default_p
                if not context:
                    context = default_c
            prompt = (
                f"Discovery Analysis:\n"
                f"User says: '{user_input}'\n"
                f"Context: {', '.join(context) if context else 'None'}\n"
                f"Preferences: {', '.join(prefs) if prefs else 'None'}\n\n"
                "Analyze the intent and determine if more details are needed. "
                "Provide a short summary of the intent and list a few follow-up questions - only if needed."
            )
            analysis_output = call_openai_api(prompt)
            new_conversation = conversation + f"User (Discovery): {user_input}\nAgent (Analysis): {analysis_output}\n\n"
            # Set discovery subphase to clarification.
            new_disc_sub = "clarification"
            prefill = "Please clarify your intent and update context/preferences if needed."
            # Return updated state; remain in discovery phase.
            return phase, new_disc_sub, analysis_output, clarified_intent, draft_editor, new_conversation, prefill, prefs, context

        elif disc_sub == "clarification":
            # Use the stored analysis_res and the user's clarification to produce a final clarified intent.
            prompt = (
                f"Combine the following analysis:\n{analysis_res}\n\n"
                f"with the user's clarification:\n{user_input}\n\n"
                "Produce a final, clear statement of the user's intent."
            )
            final_intent = call_openai_api(prompt)
            new_conversation = conversation + f"User (Clarification): {user_input}\nAgent (Final Intent): {final_intent}\n\n"
            new_phase = "execution"
            # Save the final clarified intent.
            new_clarified_intent = final_intent
            prefill = final_intent  # pre-fill the input with the clarified intent for drafting
            return new_phase, "", analysis_res, new_clarified_intent, draft_editor, new_conversation, prefill, prefs, context

    # EXECUTION PHASE
    elif phase == "execution":
        if draft_editor == "":
            # Generate an initial draft based on the clarified intent.
            prompt = (
                f"Based on the clarified intent:\n'{clarified_intent}'\n"
                f"with Preferences: {', '.join(prefs) if prefs else 'None'} and Context: {', '.join(context) if context else 'None'}\n\n"
                "Generate an initial draft."
            )
            draft = call_openai_api(prompt)
            new_conversation = conversation + f"Agent (Initial Draft): {draft}\n\n"
            new_draft = draft
            prefill = "Please provide feedback to refine the draft."
            return phase, disc_sub, analysis_res, clarified_intent, new_draft, new_conversation, prefill, prefs, context
        else:
            # User provides feedback to refine the draft.
            prompt = (
                f"Refine the following draft:\n'{draft_editor}'\n"
                f"Using the feedback: '{user_input}'\n"
                f"while maintaining the clarified intent: '{clarified_intent}',\n"
                f"Preferences: {', '.join(prefs) if prefs else 'None'}, and Context: {', '.join(context) if context else 'None'}.\n\n"
                "Produce a refined draft."
            )
            refined_draft = call_openai_api(prompt)
            new_conversation = conversation + f"User Feedback: {user_input}\nAgent (Refined Draft): {refined_draft}\n\n"
            new_draft = refined_draft
            new_phase = "verification"
            prefill = refined_draft
            return new_phase, disc_sub, analysis_res, clarified_intent, new_draft, new_conversation, prefill, prefs, context

    # VERIFICATION PHASE
    elif phase == "verification":
        prompt = (
            f"Review the following draft:\n'{draft_editor}'\n"
            f"and explain how it incorporates the clarified intent: '{clarified_intent}',\n"
            f"Preferences: {', '.join(prefs) if prefs else 'None'}, and Context: {', '.join(context) if context else 'None'}.\n\n"
            "Produce a final verified output."
        )
        final_output = call_openai_api(prompt)
        new_conversation = conversation + f"Agent (Final Output): {final_output}\n\n"
        new_phase = "completed"
        prefill = final_output
        new_draft = final_output
        return new_phase, disc_sub, analysis_res, clarified_intent, new_draft, new_conversation, prefill, prefs, context

    # COMPLETED PHASE
    else:
        return phase, disc_sub, analysis_res, clarified_intent, draft_editor, conversation, user_input, prefs, context

# Function to submit final result to Redis
def submit_final_result(final_content):
    try:
        # Get the original request data from Redis
        req_data = redis_client.rpop("request_queue")
        if req_data:
            req_data = json.loads(req_data)
            
            # Update the messages with the final content
            messages = req_data.get("messages", [])
            messages.append({"role": "assistant", "content": final_content})
            
            # Prepare the response data with all original fields and updated collaboration_phase
            response_data = {
                "messages": messages,
                "model": req_data.get("model", ""),
                "max_tokens": req_data.get("max_tokens", 0),
                "temperature": req_data.get("temperature", 0.0),
                "collaboration_phase": "completed"  # Update the collaboration phase to the last phase
            }
            
            # Push the response back to Redis
            redis_client.lpush("response_queue", json.dumps(response_data))
            return "Final result submitted successfully!"
        else:
            return "No request data found in Redis."
    except Exception as e:
        return f"Error submitting final result: {str(e)}"

# -------------------------
# Gradio UI Definition
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("# LLM-Based Dynamic Workflow")
    
    # Redis buttons and Process visualization in one row
    with gr.Row():
        fetch_btn = gr.Button("Fetch Request", size="sm")
        process_viz = gr.HTML(update_process_visualization("discovery"))
        submit_final_btn = gr.Button("Submit Result", size="sm")
    
    # Main content area with two columns
    with gr.Row():
        # Left column: User input and draft editor
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="User Input",
                placeholder="Enter your message here...",
                lines=3
            )
            
            # Joint Editor for Drafting (shows the current draft output)
            draft_editor_box = gr.TextArea(
                label="Joint Editor (Draft)",
                lines=12
            )
        
        # Right column: Conversation summary, preferences, and context
        with gr.Column(scale=1):
            conversation_box = gr.TextArea(
                label="Conversation Summary",
                lines=8,
                interactive=False
            )
            
            preferences = gr.Dropdown(
                choices=["Formal", "Casual", "Detailed", "Brief", "Technical"],
                label="Personal Preferences",
                multiselect=True,
                allow_custom_value=True
            )
            
            context_box = gr.Dropdown(
                choices=["Tech Savvy", "Manager", "Developer", "Designer"],
                label="Personal Context",
                multiselect=True,
                allow_custom_value=True
            )
    
    # Status message for submission
    submission_status = gr.Textbox(label="Submission Status", visible=False)
    
    # Hidden state variables
    phase_state = gr.State("discovery")
    disc_sub_state = gr.State("analysis")       # Discovery subphase: "analysis" then "clarification"
    analysis_res_state = gr.State("")            # To store the analysis result
    clarified_intent_state = gr.State("")
    draft_editor_state = gr.State("")
    conversation_state = gr.State("")
    
    # -------------------------
    # Event Bindings
    # -------------------------
    # Fetch request from Redis to pre-fill the User Input, Preferences, and Context panels.
    fetch_btn.click(
        fn=fetch_request_action,
        inputs=None,
        outputs=[user_input, preferences, context_box]
    )
    
    # When the user hits enter in the User Input box, process the input.
    user_input.submit(
        fn=process_input,
        inputs=[user_input, preferences, context_box, phase_state, disc_sub_state, analysis_res_state, clarified_intent_state, draft_editor_state, conversation_state],
        outputs=[phase_state, disc_sub_state, analysis_res_state, clarified_intent_state, draft_editor_state, conversation_state, user_input, preferences, context_box]
    )
    
    # Update the Process Visualization.
    phase_state.change(
        lambda ph: update_process_visualization(ph),
        inputs=[phase_state],
        outputs=[process_viz]
    )
    
    # Update the Conversation Summary panel.
    conversation_state.change(
        lambda conv: conv,
        inputs=[conversation_state],
        outputs=[conversation_box]
    )
    
    # Update the Joint Editor panel.
    draft_editor_state.change(
        lambda d: d,
        inputs=[draft_editor_state],
        outputs=[draft_editor_box]
    )
    
    # Submit the final result (contents of the Joint Editor) to Redis.
    submit_final_btn.click(
        fn=lambda final: submit_final_result(final),
        inputs=[draft_editor_state],
        outputs=gr.Textbox(label="Submission Status")
    )
    
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)