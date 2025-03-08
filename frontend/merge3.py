import gradio as gr
from gradio import update
import time
import random
import json
import redis

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Mock LLM providers (now as preference tags instead of separate dropdown)
LLM_MODELS = ["ChatGPT", "Claude", "Perplexity", "Gemini"]

# Mock context and preference tags
CONTEXT_TAGS = [
    "Mike Knoop Podcast",
    "GD Episode #42",
    "Weights & Biases Dinner",
    "Podcast Guest",
    "ML Engineer",
    "Recent Connection",
    "Investor Relations",
    "Tech Conference",
    "Partnership Opportunity",
    "Product Demo",
]

PREFERENCE_TAGS = [
    "Formal",
    "Casual",
    "Brief",
    "Detailed",
    "Technical",
    "Business-focused",
    "Include Call-to-Action",
    "Request Meeting",
    "Share Information",
    "Ask Questions",
    "Follow-up Timeline",
    "Sensitive Data",
]

# Add LLM models to preference tags
PREFERENCE_TAGS.extend([f"Use {model}" for model in LLM_MODELS])

# Initial questions to ask the user
INITIAL_QUESTIONS = [
    "Are you referring to your latest GD episode with Mike Knoop?",
    "Would you like to invite him to the upcoming M'n'M networking?",
    "Let's create a first draft of the e-mail to Mike.",
]

# Intent messages to display
INTENT_MSGS = [
    "Lukas wants to follow up with a contact. Searching in Social Graph for 'Mike'",
    "Following up with Mike Knoop for last GD episode",
    "Following up with Mike Knoop for last GD episode with invitation to M'n'M",
    "Drafting follow up e-mail to Mike. De-identifying and calling ChatGPT...",
]


def generate_mock_tags(category, count=3):
    """Generate random mock tags from the predefined lists"""
    if category == "context":
        return [
            "Mike Knoop",
            "GD Episode #42",
            "Weights & Biases Dinner",
            "Recent Connection",
        ]
    else:
        return [
            "Sensitive Data",
            "ChatGPT",
            "Business-focused",
            "Include Call-to-Action",
        ]


def push_to_redis_queue(
    messages, model, max_tokens=1000, temperature=0.7, collaboration_phase="discovery"
):
    """Push request to Redis queue for processing by backend"""
    try:
        request_data = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "collaboration_phase": collaboration_phase,
        }
        redis_client.lpush("request_queue", json.dumps(request_data))
        return True
    except Exception as e:
        print(f"Error pushing to Redis: {str(e)}")
        return False


def check_redis_response():
    """Check for response from Redis response queue"""
    try:
        response_data = redis_client.rpop("response_queue")
        if response_data:
            return json.loads(response_data)
        return None
    except Exception as e:
        print(f"Error checking Redis response: {str(e)}")
        return None


# Chat function with Redis integration
def chat(
    message,
    history,
    context_tags,
    preference_tags,
    question_count,
    current_phase,
    editor_content,
):
    history = history or []
    context_tags = context_tags or []
    preference_tags = preference_tags or []

    # Add user message to history
    history.append([message, None])
    yield history, context_tags, preference_tags, question_count, current_phase, editor_content

    # Determine which LLM to use from preference tags
    llm_provider = "ChatGPT"  # Default
    for tag in preference_tags:
        if any(f"Use {model}" in tag for model in LLM_MODELS):
            llm_provider = tag.replace("Use ", "")
            break

    # Handle initial questions flow (Discovery Phase)
    if current_phase == "discovery":
        if question_count < len(INITIAL_QUESTIONS):
            # Update intent based on user's answer
            if question_count == 0:
                editor_content = INITIAL_QUESTIONS[0]
                # Update history with AI response
                history[-1][1] = INITIAL_QUESTIONS[0]
            elif question_count == 1:
                # Add context tags after second question
                new_context_tags = generate_mock_tags("context")
                for tag in new_context_tags:
                    if tag not in context_tags:
                        context_tags.append(tag)

                editor_content = INITIAL_QUESTIONS[1]
                # Update history with AI response
                history[-1][1] = INITIAL_QUESTIONS[1]
            elif question_count == 2:
                # Add preference tags after third question
                new_preference_tags = generate_mock_tags("preference")
                for tag in new_preference_tags:
                    if tag not in preference_tags:
                        preference_tags.append(tag)
                editor_content = INITIAL_QUESTIONS[2]
                # Update history with AI response
                history[-1][1] = INITIAL_QUESTIONS[2]
            elif question_count == 3:
                # Transition to the generation phase
                editor_content = "Ok, let's draft this follow-up email now. Moving to the generation phase now."
                # Update history with AI response
                history[-1][
                    1
                ] = "Ok, let's draft this follow-up email now. Moving to the generation phase now."
                current_phase = "generation"

            question_count += 1
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content
            return

    # Generation Phase
    if current_phase == "generation":
        # Check if message is requesting a draft email
        if any(
            keyword in message.lower()
            for keyword in ["email", "draft", "write", "message", "follow", "create"]
        ):
            # Show thinking message in history
            thinking_msg = f"I'll help you draft an email based on your requirements. Let me use {llm_provider} to generate a draft."
            history[-1][1] = thinking_msg
            editor_content = thinking_msg
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            # Show tool call
            tool_call_msg = f"Calling {llm_provider} API to generate an email draft..."
            history[-1][1] = tool_call_msg
            editor_content = tool_call_msg
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            # Prepare data for Redis queue
            formatted_messages = []
            for msg in history:
                if msg[0] is not None:
                    formatted_messages.append({"role": "user", "content": msg[0]})
                if msg[1] is not None:
                    formatted_messages.append({"role": "assistant", "content": msg[1]})

            # Add context and preferences as system message
            context_str = ", ".join(context_tags)
            preferences_str = ", ".join(
                [
                    tag
                    for tag in preference_tags
                    if not any(f"Use {model}" in tag for model in LLM_MODELS)
                ]
            )

            system_message = {
                "role": "system",
                "content": f"Context: {context_str}\nPreferences: {preferences_str}\nTask: Write a follow-up email.",
            }

            formatted_messages.insert(0, system_message)

            # Push to Redis for processing
            success = push_to_redis_queue(
                formatted_messages,
                llm_provider,
                max_tokens=1000,
                temperature=0.7,
                collaboration_phase="generation",
            )

            if not success:
                # Fallback to mock response if Redis push fails
                draft = """Subject: Follow-up from our recent conversation

Dear Mike,

I hope this email finds you well. I wanted to follow up after our recent conversation on the Gradient Dissent podcast (episode #42). It was great having you as a guest!

Based on our discussion, I thought you might be interested in attending our upcoming Weights & Biases dinner event next month.

The event will bring together leaders in the ML space to discuss the latest trends and opportunities for collaboration. I believe your insights would be valuable to the group.

Would you be available to join us on March 20th? Please let me know if you have any questions.

Best regards,
Lukas"""
            else:
                # Try to get response from Redis (with timeout)
                attempts = 0
                draft = None
                while attempts < 10 and draft is None:
                    response_data = check_redis_response()
                    if response_data and "content" in response_data:
                        draft = response_data["content"]
                        break
                    time.sleep(0.5)
                    attempts += 1

                # Fallback if no response received
                if draft is None:
                    draft = """Subject: Follow-up from our recent conversation

Dear Mike,

I hope this email finds you well. I wanted to follow up after our recent conversation on the Gradient Dissent podcast (episode #42). It was great having you as a guest!

Based on our discussion, I thought you might be interested in attending our upcoming Weights & Biases dinner event next month.

The event will bring together leaders in the ML space to discuss the latest trends and opportunities for collaboration. I believe your insights would be valuable to the group.

Would you be available to join us on March 20th? Please let me know if you have any questions.

Best regards,
Lukas"""

            # Update editor content
            editor_content = draft

            # Show completed tool call (but keep brief in chat)
            completed_msg = "I've generated an email draft based on your preferences. You can see and edit it in the shared editor."
            history[-1][1] = completed_msg
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            # Add a new message to history
            history.append(
                [
                    None,
                    "You can edit the draft directly in the editor. Would you like me to modify anything specific? You can also adjust the context and preference tags to refine the message.",
                ]
            )
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content
        elif (
            "adapt" in message.lower()
            or "modify" in message.lower()
            or "change" in message.lower()
            or "revise" in message.lower()
        ):
            # Handle adaptation requests
            adapt_msg = f"I'll adapt the current draft based on your feedback. Let me make those changes for you."
            history[-1][1] = adapt_msg
            editor_content = adapt_msg
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            # Show tool call for adaptation
            tool_call_msg = f"Calling adaptation tool to modify the draft..."
            history[-1][1] = tool_call_msg
            editor_content = tool_call_msg
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            # Prepare data for Redis queue
            formatted_messages = []
            for msg in history:
                if msg[0] is not None:
                    formatted_messages.append({"role": "user", "content": msg[0]})
                if msg[1] is not None:
                    formatted_messages.append({"role": "assistant", "content": msg[1]})

            # Add current draft and request as system message
            system_message = {
                "role": "system",
                "content": f"Current draft:\n{editor_content}\n\nRequest: Modify this draft according to the user's request.",
            }

            formatted_messages.insert(0, system_message)

            # Push to Redis for processing
            success = push_to_redis_queue(
                formatted_messages,
                llm_provider,
                max_tokens=1000,
                temperature=0.7,
                collaboration_phase="final_review",
            )

            if not success:
                # Fallback to mock modification if Redis push fails
                if "more formal" in message.lower():
                    editor_content = editor_content.replace(
                        "I hope this email finds you well",
                        "I trust this communication finds you in good health",
                    )
                    if "Best regards" in editor_content:
                        editor_content = editor_content.replace(
                            "Best regards", "Sincerely"
                        )
                elif "more casual" in message.lower():
                    editor_content = editor_content.replace(
                        "I hope this email finds you well", "Hope you're doing great"
                    )
                    if "Sincerely" in editor_content:
                        editor_content = editor_content.replace("Sincerely", "Cheers")
                elif "shorter" in message.lower():
                    lines = editor_content.split("\n")
                    if len(lines) > 6:  # Remove some middle content
                        editor_content = "\n".join(
                            lines[:4]
                            + ["\nWould you be available to join us on March 20th?\n"]
                            + lines[-2:]
                        )
                else:
                    # Generic change - add a sentence
                    if "dinner event" in editor_content:
                        editor_content = editor_content.replace(
                            "dinner event next month.",
                            "dinner event next month. We'll have several industry leaders attending, and your expertise would be a valuable addition to our discussions.",
                        )
            else:
                # Try to get response from Redis (with timeout)
                attempts = 0
                modified_draft = None
                while attempts < 10 and modified_draft is None:
                    response_data = check_redis_response()
                    if response_data and "content" in response_data:
                        modified_draft = response_data["content"]
                        break
                    time.sleep(0.5)
                    attempts += 1

                # Update if response received
                if modified_draft is not None:
                    editor_content = modified_draft

            # Show completed adaptation
            history[-1][
                1
            ] = "I've updated the draft based on your feedback. You can see the changes in the shared editor."
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content
        else:
            # General response for other queries
            history[-1][
                1
            ] = "I can help you draft emails and messages. Just let me know what you'd like to write or how you'd like to modify the current draft in the editor."
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content


# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Mediator")

    # State variables
    current_context_tags = gr.State([])
    current_preference_tags = gr.State([])
    question_counter = gr.State(0)
    intent_counter = gr.State(0)
    app_phase = gr.State("discovery")  # "discovery" or "generation"
    current_editor_content = gr.State("")

    with gr.Tabs() as tabs:
        with gr.Tab("AI Mediator"):
            with gr.Row():
                # Column 1: Shared Editor and Chat (always visible)
                with gr.Column(scale=2) as editor_column:
                    editor = gr.TextArea(
                        label="Shared Editor",
                        placeholder="",
                        lines=20,
                        show_copy_button=True,
                    )
                    chat_history = gr.Chatbot(
                        label="Conversation",
                        height=300,
                        show_copy_button=True,
                    )
                    chat_input = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=4,
                        container=False,
                    )

                # Column 2: Context and Preferences (always visible)
                with gr.Column(scale=1) as context_column:
                    intent_display = gr.Textbox(
                        label="Agent Reasoning",
                        placeholder="Agent's thoughts will be displayed here...",
                        lines=10,
                    )

                    context_tags_component = gr.Dropdown(
                        multiselect=True,
                        label="Context Tags",
                        info="Tags are added automatically, but you can also edit them",
                        allow_custom_value=True,
                    )

                    preference_tags_component = gr.Dropdown(
                        choices=PREFERENCE_TAGS,
                        multiselect=True,
                        label="Preference Tags",
                        info="Tags are added automatically, but you can also edit them",
                        allow_custom_value=True,
                    )

        # Memory Tab (empty for now)
        with gr.Tab("Memory"):
            gr.Markdown("## User Memory")
            gr.Markdown(
                "This tab will store context and background information about users for future interactions."
            )

            with gr.Accordion("Available Information", open=False):
                gr.Markdown("- No stored memory available yet")
                gr.Markdown("- User information will be stored here")
                gr.Markdown("- Past interactions will be accessible from this tab")

            gr.Button("Refresh Memory", variant="secondary")

        # Admin Tab for Redis Queue Management
        with gr.Tab("Admin"):
            gr.Markdown("## Queue Management")

            with gr.Row():
                with gr.Column():
                    queue_status = gr.TextArea(label="Queue Status", lines=5)
                    refresh_queue_btn = gr.Button("Refresh Queue Status")

                    def get_queue_status():
                        try:
                            request_queue_len = redis_client.llen("request_queue")
                            response_queue_len = redis_client.llen("response_queue")
                            return f"Redis Queue Status:\n- Pending Requests: {request_queue_len}\n- Pending Responses: {response_queue_len}\n\nLast checked: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                        except Exception as e:
                            return f"Error connecting to Redis: {str(e)}"

                    refresh_queue_btn.click(get_queue_status, outputs=[queue_status])

                with gr.Column():
                    clear_queue_btn = gr.Button("Clear All Queues", variant="stop")
                    queue_action_status = gr.Textbox(label="Action Status")

                    def clear_all_queues():
                        try:
                            redis_client.delete("request_queue")
                            redis_client.delete("response_queue")
                            return "All Redis queues cleared successfully."
                        except Exception as e:
                            return f"Error clearing queues: {str(e)}"

                    clear_queue_btn.click(
                        clear_all_queues, outputs=[queue_action_status]
                    )

    # Event handlers
    def update_intent(editor_content, intent_counter):
        if not editor_content:
            return "Waiting for user input...", intent_counter

        # Check if the editor content matches any of the initial questions
        if editor_content == INITIAL_QUESTIONS[0]:
            return INTENT_MSGS[0], 0
        elif editor_content == INITIAL_QUESTIONS[1]:
            return INTENT_MSGS[1], 1
        elif editor_content == INITIAL_QUESTIONS[2]:
            return INTENT_MSGS[2], 2
        elif "generation phase" in editor_content.lower():
            return "Transitioning to generation phase...", 3
        elif "draft" in editor_content.lower() and "email" in editor_content.lower():
            # This is for the final email draft
            return "Email draft completed and ready for review", 3
        else:
            # For other editor content changes
            if intent_counter < len(INTENT_MSGS):
                return INTENT_MSGS[intent_counter], intent_counter
            else:
                return "Lukas is exploring communication options", intent_counter

    # Update UI based on the current phase
    def update_ui_for_phase(phase):
        if phase == "discovery":
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        else:  # "generation"
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

    # Handle chat submission
    chat_input.submit(
        chat,
        inputs=[
            chat_input,
            chat_history,
            current_context_tags,
            current_preference_tags,
            question_counter,
            app_phase,
            current_editor_content,
        ],
        outputs=[
            chat_history,
            current_context_tags,
            current_preference_tags,
            question_counter,
            app_phase,
            current_editor_content,
        ],
    ).then(lambda: "", outputs=[chat_input])

    # Update editor content
    current_editor_content.change(
        lambda x: x, inputs=[current_editor_content], outputs=[editor]
    )

    # Update intent when editor content changes
    current_editor_content.change(
        update_intent,
        inputs=[current_editor_content, intent_counter],
        outputs=[intent_display, intent_counter],
    )

    # Phase change triggers UI update
    app_phase.change(
        update_ui_for_phase,
        inputs=[app_phase],
        outputs=[editor_column, context_column],
    )

    # Context tags changes
    context_tags_component.change(
        lambda x: x, inputs=[context_tags_component], outputs=[current_context_tags]
    )

    current_context_tags.change(
        lambda x: x, inputs=[current_context_tags], outputs=[context_tags_component]
    )

    # Preference tags changes
    preference_tags_component.change(
        lambda x: x,
        inputs=[preference_tags_component],
        outputs=[current_preference_tags],
    )

    current_preference_tags.change(
        lambda x: x,
        inputs=[current_preference_tags],
        outputs=[preference_tags_component],
    )

# Launch the app
if __name__ == "__main__":
    try:
        # Test Redis connection
        redis_client.ping()
        print("Successfully connected to Redis")
    except Exception as e:
        print(f"Warning: Could not connect to Redis: {str(e)}")
        print("App will run with mocked responses instead of Redis backend")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
