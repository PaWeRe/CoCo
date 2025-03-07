import gradio as gr
from gradio import ChatMessage
import time
import random

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
]

# Add LLM models to preference tags
PREFERENCE_TAGS.extend([f"Use {model}" for model in LLM_MODELS])


def generate_mock_tags(category, count=3):
    """Generate random mock tags from the predefined lists"""
    if category == "context":
        # return random.sample(CONTEXT_TAGS, min(count, len(CONTEXT_TAGS)))
        return [
            "Mike Knoop",
            "GD Episode #42",
            "Weights & Biases Dinner",
            "Recent Connection",
        ]
    else:
        # return random.sample(PREFERENCE_TAGS[:11], min(count, 11))  # Only use original preferences, not LLM models
        return [
            "Sensitive Data",
            "ChatGPT",
            "Business-focused",
            "Include Call-to-Action",
        ]


# Initial questions to ask the user
INITIAL_QUESTIONS = [
    "Are you referring to your latest GD episode with Mike Knoop?",
    "Would you like to invite him to the upcoming M'n'M networking?",
    "Let's create a first draft of the e-mail to Mike.",
]

# Intent quetsions to display
INTENT_MSGS = [
    "Lukas wants to follow up with a contact. Searching in Social Graph for 'Mike'",
    "Following up with Mike Knoop for last GD episode",
    "Following up with Mike Knoop for last GD episode with invitation to M'n'M",
    "Drafting follow up e-mail to Mike. De-identifying and calling ChatGPT...",
]


# Chat function with tool call simulation
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

    # Add user message
    history.append(ChatMessage(role="user", content=message))
    yield history, context_tags, preference_tags, question_count, current_phase, editor_content

    time.sleep(0.5)

    # Handle initial questions flow (Discovery Phase)
    if current_phase == "discovery":
        if question_count < len(INITIAL_QUESTIONS):
            # Update intent based on user's answer
            if question_count == 0:
                editor_content = INITIAL_QUESTIONS[0]
            elif question_count == 1:
                # Add context tags after second question
                new_context_tags = generate_mock_tags("context")
                for tag in new_context_tags:
                    if tag not in context_tags:
                        context_tags.append(tag)

                editor_content = INITIAL_QUESTIONS[1]
            elif question_count == 2:
                # Add preference tags after third question
                new_preference_tags = generate_mock_tags("preference")
                for tag in new_preference_tags:
                    if tag not in preference_tags:
                        preference_tags.append(tag)
                editor_content = INITIAL_QUESTIONS[2]
            elif question_count == 3:
                # Transition to the generation phase
                editor_content = "Ok, let's draft this follow-up email now. Moving to the generation phase now."
                current_phase = "generation"

            question_count += 1
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content
            return

    # Generation Phase
    if current_phase == "generation":
        # Determine which LLM to use from preference tags
        llm_provider = "ChatGPT"  # Default
        for tag in preference_tags:
            if any(f"Use {model}" in tag for model in LLM_MODELS):
                llm_provider = tag.replace("Use ", "")
                break

        if any(
            keyword in message.lower()
            for keyword in ["email", "draft", "write", "message", "follow", "create"]
        ):
            # Show thinking message
            editor_content = f"I'll help you draft an email based on your requirements. Let me use {llm_provider} to generate a draft."
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            time.sleep(1)

            # Show tool call
            editor_content = f"Calling {llm_provider} API to generate an email draft..."
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            time.sleep(1.5)

            # Generate draft based on context and preferences
            context_str = ", ".join(context_tags[:3])
            preferences_str = ", ".join(
                [
                    tag
                    for tag in preference_tags
                    if not any(f"Use {model}" in tag for model in LLM_MODELS)
                ][:3]
            )

            draft = f"""Subject: Follow-up from our recent conversation

Dear Mike,

I hope this email finds you well. I wanted to follow up after our recent conversation on the Gradient Dissent podcast (episode #42). It was great having you as a guest!

Based on our discussion about {context_str}, I thought you might be interested in attending our upcoming Weights & Biases dinner event next month.

The event will bring together leaders in the ML space to discuss the latest trends and opportunities for collaboration. I believe your insights would be valuable to the group.

Would you be available to join us on March 20th? Please let me know if you have any questions.

Best regards,
Lukas"""

            # Update editor content instead of showing in chat
            editor_content = draft

            # Show completed tool call (but keep brief in chat)
            history.append(
                ChatMessage(
                    role="assistant",
                    content="I've generated an email draft based on your preferences. You can see and edit it in the shared editor.",
                    metadata={"title": f"✅ {llm_provider} Draft Generated"},
                )
            )
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            # Final assistant message
            history.append(
                ChatMessage(
                    role="assistant",
                    content="You can edit the draft directly in the editor. Would you like me to modify anything specific? You can also adjust the context and preference tags to refine the message.",
                )
            )
        elif (
            "adapt" in message.lower()
            or "modify" in message.lower()
            or "change" in message.lower()
            or "revise" in message.lower()
        ):
            # Handle adaptation requests
            editor_content = f"I'll adapt the current draft based on your feedback. Let me make those changes for you."
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            time.sleep(1)

            # Show tool call for adaptation
            editor_content = f"Calling adaptation tool to modify the draft..."
            yield history, context_tags, preference_tags, question_count, current_phase, editor_content

            time.sleep(1.5)

            # Modify the draft based on user's request
            # For demonstration, we'll just make a simple modification
            if "more formal" in message.lower():
                editor_content = editor_content.replace(
                    "I hope this email finds you well",
                    "I trust this communication finds you in good health",
                )
                if "Best regards" in editor_content:
                    editor_content = editor_content.replace("Best regards", "Sincerely")
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

            # Show completed adaptation
            history.append(
                ChatMessage(
                    role="assistant",
                    content="I've updated the draft based on your feedback. You can see the changes in the shared editor.",
                    metadata={"title": "✅ Text Adaptation Complete"},
                )
            )
        else:
            # General response for other queries
            history.append(
                ChatMessage(
                    role="assistant",
                    content="I can help you draft emails and messages. Just let me know what you'd like to write or how you'd like to modify the current draft in the editor.",
                )
            )

    yield history, context_tags, preference_tags, question_count, current_phase, editor_content


# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Mediator MVP")

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
                # Column 1: Shared Editor (always visible)
                with gr.Column(scale=2) as editor_column:
                    # gr.Markdown("<center><h3>Shared Editor</h3></center>")
                    editor = gr.TextArea(
                        label="Shared Editor",
                        placeholder="",
                        lines=20,
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
                    # gr.Markdown("<center><h3>Agent Brain</h3></center>")
                    intent_display = gr.Textbox(
                        label="Agent Reasoning",
                        placeholder="Agent's thoughts will be displayed here...",
                        lines=10,
                    )

                    # gr.Markdown("<center><h3>Context</h3></center>")
                    context_tags_component = gr.Dropdown(
                        multiselect=True,
                        label="Context Tags",
                        info="Tags are added automatically, but you can also edit them",
                        allow_custom_value=True,
                    )

                    # gr.Markdown("<center><h3>Preferences</h3></center>")
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
        elif editor_content == INITIAL_QUESTIONS[3]:
            return INTENT_MSGS[3], 3
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
            gr.State([]),
            current_context_tags,
            current_preference_tags,
            question_counter,
            app_phase,
            current_editor_content,
        ],
        outputs=[
            gr.State([]),
            current_context_tags,
            current_preference_tags,
            question_counter,
            app_phase,
            current_editor_content,
        ],
    )

    # Update intent when chat history changes
    # chat_input.change(update_intent, inputs=[chat_input], outputs=[intent_display])

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
    demo.launch(share=False)
