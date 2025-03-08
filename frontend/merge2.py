import gradio as gr
from gradio import ChatMessage
import time
import random

# Mock LLM providers
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

# Initial questions for discovery phase
INITIAL_QUESTIONS = [
    "Are you referring to your latest GD episode with Mike Knoop?",
    "Would you like to invite him to the upcoming M'n'M networking?",
    "Drafting follow up e-mail to Mike...",
]

# Agent reasoning steps for execution phase
EXECUTION_STEPS = [
    "Writing prompt for e-mail to Mike",
    "Removing all possible PII",
    "Calling ChatGPT",
    "Return e-mail draft",
]

# Intent messages to display
INTENT_MSGS = [
    "Lukas wants to follow up with a contact. Searching in Social Graph for 'Mike'",
    "Following up with Mike Knoop for last GD episode",
    "Following up with Mike Knoop for last GD episode with invitation to M'n'M",
    "Drafting e-mail to Mike: - ..., - ..., -...",
]


def generate_mock_tags(category, count=3):
    """Generate specific mock tags"""
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


# Chat function with three phases
def chat(
    message,
    history,
    context_tags,
    preference_tags,
    question_count,
    current_phase,
    editor_content,
    execution_step,
):
    history = history or []
    context_tags = context_tags or []
    preference_tags = preference_tags or []

    # Add user message
    history.append(ChatMessage(role="user", content=message))
    yield history, context_tags, preference_tags, question_count, current_phase, editor_content, execution_step

    time.sleep(0.5)

    # DISCOVERY PHASE
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

            question_count += 1

            # Transition to execution phase after all questions
            if question_count >= 3:
                current_phase = "execution"
                editor_content = "E-Mail draft writing in progress..."
                execution_step = 0

            yield history, context_tags, preference_tags, question_count, current_phase, editor_content, execution_step
            return

    # EXECUTION PHASE
    elif current_phase == "execution":
        # Determine which LLM to use from preference tags
        llm_provider = "ChatGPT"  # Default
        for tag in preference_tags:
            if any(f"Use {model}" in tag for model in LLM_MODELS):
                llm_provider = tag.replace("Use ", "")
                break

        # Start execution process automatically
        editor_content = "E-Mail draft writing in progress..."

        # Generate draft based on context and preferences
        draft = f"""Subject: Follow-up from our recent conversation

Dear Mike,

I hope this email finds you well. I wanted to follow up after our recent conversation on the Gradient Dissent podcast (episode #42). It was great having you as a guest!

Based on our discussion about machine learning applications, I thought you might be interested in attending our upcoming Weights & Biases dinner event next month.

The event will bring together leaders in the ML space to discuss the latest trends and opportunities for collaboration. I believe your insights would be valuable to the group.

Would you be available to join us on March 20th? Please let me know if you have any questions.

Best regards,
Lukas"""

        # After the process is complete, show the draft and transition to verification
        if execution_step >= len(EXECUTION_STEPS):
            editor_content = draft
            current_phase = "verification"

            # Add assistant message about draft completion
            history.append(
                ChatMessage(
                    role="assistant",
                    content="I've generated an email draft for your review. You can see it in the shared editor. Is there anything you'd like me to modify?",
                )
            )

        yield history, context_tags, preference_tags, question_count, current_phase, editor_content, execution_step

    # VERIFICATION PHASE
    elif current_phase == "verification":
        # Handle verification and editing requests
        if any(
            keyword in message.lower()
            for keyword in ["looks good", "approve", "send", "fine"]
        ):
            history.append(
                ChatMessage(
                    role="assistant",
                    content="Great! The email has been approved and is ready to send.",
                )
            )
        else:
            history.append(
                ChatMessage(
                    role="assistant",
                    content="I understand. You can make direct edits in the editor, or I can help make specific changes if you describe what you'd like modified.",
                )
            )

        yield history, context_tags, preference_tags, question_count, current_phase, editor_content, execution_step


# Function to generate execution steps display with checkboxes
def generate_execution_display(step):
    result = ""
    for i, s in enumerate(EXECUTION_STEPS):
        if i < step:
            result += f"✅ {s} - Completed\n"
        elif i == step:
            result += f"⏳ {s} - In progress...\n"
        else:
            result += f"⬜ {s} - Pending\n"
    return result


# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Mediator MVP")

    # State variables
    current_context_tags = gr.State([])
    current_preference_tags = gr.State([])
    question_counter = gr.State(0)
    app_phase = gr.State("discovery")  # "discovery", "execution", or "verification"
    current_editor_content = gr.State("")
    current_execution_step = gr.State(0)

    with gr.Tabs() as tabs:
        with gr.Tab("AI Mediator"):
            # Phase visualization - all phases shown vertically
            with gr.Column():
                # Phase indicators with progress visualization
                with gr.Row():
                    phase_boxes = gr.HighlightedText(
                        label="Process Phases",
                        value=[
                            ("Discovery", "discovery"),
                            ("Execution", "execution"),
                            ("Verification", "verification"),
                        ],
                        color_map={
                            "discovery": "green",
                            "execution": "blue",
                            "verification": "purple",
                            "inactive": "gray",
                        },
                        interactive=False,
                    )

            # Main content area
            with gr.Row():
                # Column 1: Context and Preferences + Shared Editor
                with gr.Column(scale=2) as editor_column:
                    # Context and Preference tags at the top
                    with gr.Row():
                        # Context tags
                        context_tags_component = gr.Dropdown(
                            multiselect=True,
                            label="Context Tags",
                            info="Tags are added automatically, but you can also edit them",
                            allow_custom_value=True,
                        )

                        # Preference tags
                        preference_tags_component = gr.Dropdown(
                            choices=PREFERENCE_TAGS,
                            multiselect=True,
                            label="Preference Tags",
                            info="Tags are added automatically, but you can also edit them",
                            allow_custom_value=True,
                        )

                    # Shared Editor below the tags
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

                # Column 2: Agent Reasoning
                with gr.Column(scale=1) as reasoning_column:
                    # Agent reasoning display
                    agent_reasoning = gr.TextArea(
                        label="Agent Reasoning",
                        placeholder="Agent's thoughts will be displayed here...",
                        lines=20,
                    )

        # Memory Tab
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
    def update_intent(editor_content, execution_step, current_phase):
        # For discovery phase
        if current_phase == "discovery":
            if editor_content == INITIAL_QUESTIONS[0]:
                return INTENT_MSGS[0]
            elif editor_content == INITIAL_QUESTIONS[1]:
                return INTENT_MSGS[1]
            elif editor_content == INITIAL_QUESTIONS[2]:
                return INTENT_MSGS[2]
            else:
                return "Gathering information about your request..."

        # For execution phase
        elif current_phase == "execution":
            return generate_execution_display(execution_step)

        # For verification phase
        elif current_phase == "verification":
            return "Email draft ready for your review. Please approve or suggest modifications."

        return "Waiting for your input..."

    # Update phase visualization
    def update_phase_visualization(phase):
        phases = [
            ("Discovery", "inactive"),
            ("Execution", "inactive"),
            ("Verification", "inactive"),
        ]

        if phase == "discovery":
            phases[0] = ("Discovery", "discovery")
        elif phase == "execution":
            phases[1] = ("Execution", "execution")
        elif phase == "verification":
            phases[2] = ("Verification", "verification")

        return phases

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
            current_execution_step,
        ],
        outputs=[
            gr.State([]),
            current_context_tags,
            current_preference_tags,
            question_counter,
            app_phase,
            current_editor_content,
            current_execution_step,
        ],
    )

    # Update editor content
    current_editor_content.change(
        lambda x: x, inputs=[current_editor_content], outputs=[editor]
    )

    # FIX: Update agent reasoning when execution step changes
    current_execution_step.change(
        update_intent,
        inputs=[current_editor_content, current_execution_step, app_phase],
        outputs=[agent_reasoning],
    )

    # Update agent reasoning when editor content changes
    current_editor_content.change(
        update_intent,
        inputs=[current_editor_content, current_execution_step, app_phase],
        outputs=[agent_reasoning],
    )

    # Update agent reasoning when phase changes
    app_phase.change(
        update_intent,
        inputs=[current_editor_content, current_execution_step, app_phase],
        outputs=[agent_reasoning],
    )

    # Update phase visualization when phase changes
    app_phase.change(
        update_phase_visualization,
        inputs=[app_phase],
        outputs=[phase_boxes],
    )

    # FIX: Properly implement execution step progression with a timer
    def auto_execution_step(phase, step):
        if phase == "execution" and step < len(EXECUTION_STEPS):
            return step + 1
        return step

    # Create a hidden component to trigger the timer
    timer = gr.HTML(visible=False)

    # Function to check and update execution step
    def check_execution_phase(phase, step):
        if phase == "execution" and step < len(EXECUTION_STEPS):
            # Return a timestamp to force the component to update
            return f"<div>{time.time()}</div>"
        return ""

    # Set up the timer to check execution phase every second
    demo.load(
        lambda: gr.update(every=2),
        inputs=None,
        outputs=[timer],
    )

    # When the timer updates, check if we need to increment the execution step
    timer.change(
        check_execution_phase,
        inputs=[app_phase, current_execution_step],
        outputs=[timer],
    ).then(
        auto_execution_step,
        inputs=[app_phase, current_execution_step],
        outputs=[current_execution_step],
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
