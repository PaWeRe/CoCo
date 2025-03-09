import gradio as gr
from gradio import ChatMessage
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

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

# Mock social graph data
MOCK_SOCIAL_GRAPH = {
    "Lukas": {
        "attributes": {
            "role": "W&B CEO",
            "interests": ["ML", "Startups", "Podcasting"],
        },
        "connections": {
            "Mike Knoop": {
                "type": "professional",
                "strength": 7,
                "last_contact": "2 weeks ago",
            },
            "Seema Gajwani": {
                "type": "mentor",
                "strength": 9,
                "last_contact": "1 month ago",
            },
            "Sarah Hooker": {
                "type": "colleague",
                "strength": 8,
                "last_contact": "5 days ago",
            },
            "Andrej Karpathy": {
                "type": "acquaintance",
                "strength": 5,
                "last_contact": "3 months ago",
            },
        },
    },
    "Mike Knoop": {
        "attributes": {
            "role": "Founder at Runway",
            "interests": ["AI", "Video Generation", "Design"],
        },
        "connections": {
            "Lukas": {
                "type": "podcast guest",
                "strength": 7,
                "last_contact": "2 weeks ago",
            },
        },
    },
    "Seema Gajwani": {
        "attributes": {
            "role": "Legal Expert",
            "interests": ["AI Ethics", "Policy", "Regulation"],
        },
        "connections": {
            "Lukas": {"type": "mentee", "strength": 9, "last_contact": "1 month ago"},
        },
    },
    "Sarah Hooker": {
        "attributes": {
            "role": "ML Researcher",
            "interests": ["ML Robustness", "Education", "AI Safety"],
        },
        "connections": {
            "Lukas": {
                "type": "research partner",
                "strength": 8,
                "last_contact": "5 days ago",
            },
        },
    },
    "Andrej Karpathy": {
        "attributes": {
            "role": "AI Researcher",
            "interests": ["Deep Learning", "Vision", "Education"],
        },
        "connections": {
            "Lukas": {
                "type": "conference connection",
                "strength": 5,
                "last_contact": "3 months ago",
            },
        },
    },
}

# Mock active memory data
MOCK_ACTIVE_MEMORY = [
    {
        "source": "Cursor",
        "timestamp": "2 hours ago",
        "type": "code",
        "content": """
def train_model(X_train, y_train, epochs=100):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=0.2,
        verbose=0
    )
    
    return model, history
""",
        "query": "Can you optimize this model for better performance with imbalanced data?",
    },
    {
        "source": "Slack",
        "timestamp": "Yesterday",
        "type": "message",
        "content": "Team meeting scheduled for tomorrow at 10am PT to discuss Q1 roadmap.",
        "query": "Can you remind me what we need to prepare for the roadmap meeting?",
    },
]

# Memory chat responses
MEMORY_CHAT_RESPONSES = {
    "default": "Hi Lukas, anything to update?",
    "new_person": "I'll add Mike Knoop to your social graph. What's your relationship with him?",
    "person_details": "Thanks! What are Mike's interests and current role?",
    "person_confirmed": "Added Mike Knoop to your social graph. I've noted he's a founder at Runway with interests in AI video generation and design, and you met him during a podcast recording.",
    "memory_query": "Based on your social graph, I see you haven't connected with Andrej in 3 months. Would you like to draft a follow-up message?",
}


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


# Create a social graph visualization
def create_social_graph():
    G = nx.Graph()

    # Add nodes
    for person, data in MOCK_SOCIAL_GRAPH.items():
        G.add_node(person)

    # Add edges
    for person, data in MOCK_SOCIAL_GRAPH.items():
        for connection, conn_data in data.get("connections", {}).items():
            G.add_edge(
                person,
                connection,
                weight=conn_data["strength"],
                type=conn_data["type"],
                last_contact=conn_data["last_contact"],
            )

    # Create the plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    node_sizes = []
    node_colors = []

    for node in G.nodes():
        if node == "Lukas":
            node_sizes.append(1500)
            node_colors.append("lightblue")
        else:
            node_sizes.append(1000)
            node_colors.append("lightgreen")

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8
    )

    # Draw edges with varying widths
    edge_widths = [G[u][v]["weight"] / 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # Add edge labels (relationship types)
    edge_labels = {(u, v): G[u][v]["type"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.axis("off")
    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Convert to base64 for displaying
    img = Image.open(buf)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    img_str = base64.b64encode(img_byte_arr).decode("utf-8")
    img_html = f'<img src="data:image/png;base64,{img_str}" alt="Social Graph" style="width:100%">'

    plt.close()
    return img_html


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
        draft = f"""Subject: Follow up e-mail Mike Knoop:

Mike,

just wanted to say thanks again for joining the podcast—really enjoyed your unique vision for the path to AGI and good luck for ndea! Also, wanted to invite you to our next Meet&Mingle event, would be great to catch up and connect with more folks. It’s on Jan 17th, 5 PM at Freiheitshalle—hope you can make it!

PS: attaching our lunch group photo, figured you might like it.

Best,
Lukas


"""

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


# Memory chat function
def memory_chat(message, history):
    history = history or []

    # Add user message
    history.append(ChatMessage(role="user", content=message))

    # Generate response based on specific keywords
    response = ""
    if "add" in message.lower() and "mike" in message.lower():
        response = MEMORY_CHAT_RESPONSES["new_person"]
    elif "podcast" in message.lower() and "recorded" in message.lower():
        response = MEMORY_CHAT_RESPONSES["person_details"]
    elif "founder" in message.lower() and "runway" in message.lower():
        response = MEMORY_CHAT_RESPONSES["person_confirmed"]
    elif "andrej" in message.lower() or "karpathy" in message.lower():
        response = MEMORY_CHAT_RESPONSES["memory_query"]
    else:
        response = "I've noted that. Anything else you'd like to share or update?"

    # Add assistant response
    history.append(ChatMessage(role="assistant", content=response))

    return history


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


# Function to load active memory content
def load_active_memory(index):
    if 0 <= index < len(MOCK_ACTIVE_MEMORY):
        entry = MOCK_ACTIVE_MEMORY[index]
        content = f"Source: {entry['source']}\nTimestamp: {entry['timestamp']}\n\n{entry['content']}\n\nQuery: {entry['query']}"
        return content
    return "No memory content available"


# Connect external source
def connect_external_source(source):
    return f"Successfully connected to {source}. Data will appear here."


# Function to load active memory content and trigger workflow
def load_active_memory_and_trigger_workflow(
    memory_source,
    editor_content,
    phase,
    question_count,
    context_tags,
    preference_tags,
    execution_step,
):
    # Convert the radio selection to an index
    index = 0 if memory_source == "Cursor Code Snippet" else 1

    # Get the memory content
    entry = MOCK_ACTIVE_MEMORY[index] if 0 <= index < len(MOCK_ACTIVE_MEMORY) else None

    if entry:
        # Get the query from the memory entry
        query = entry.get("query", "No query available")

        # Reset the workflow state
        new_phase = "discovery"
        new_question_count = 0
        new_execution_step = 0

        # Set the query to the editor
        new_editor_content = query

        # Clear tags for new workflow
        new_context_tags = []
        new_preference_tags = []

        # Get the memory content for display
        memory_content = f"Source: {entry['source']}\nTimestamp: {entry['timestamp']}\n\n{entry['content']}\n\nQuery: {query}"

        # Return all the updated values
        return (
            memory_content,
            new_editor_content,
            new_phase,
            new_question_count,
            new_context_tags,
            new_preference_tags,
            new_execution_step,
            gr.update(selected=0),
        )  # Switch to first tab (AI Mediator)

    return (
        "No memory content available",
        editor_content,
        phase,
        question_count,
        context_tags,
        preference_tags,
        execution_step,
        gr.update(),
    )


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

            # Main content area - restructured to have tags in one row and editor+reasoning in another
            with gr.Column():
                # Context and Preference tags in the same row
                with gr.Row():
                    # Context tags
                    context_tags_component = gr.Dropdown(
                        multiselect=True,
                        label="Context Tags",
                        info="Tags are added automatically, but you can also edit them",
                        allow_custom_value=True,
                        scale=1,
                    )

                    # Preference tags
                    preference_tags_component = gr.Dropdown(
                        choices=PREFERENCE_TAGS,
                        multiselect=True,
                        label="Preference Tags",
                        info="Tags are added automatically, but you can also edit them",
                        allow_custom_value=True,
                        scale=1,
                    )

                # Shared Editor and Agent Reasoning in the same row
                with gr.Row():
                    # Shared Editor
                    with gr.Column(scale=3):
                        editor = gr.TextArea(
                            label="Shared Editor",
                            placeholder="",
                            lines=20,
                            show_copy_button=True,
                        )

                    # Agent Reasoning
                    with gr.Column(scale=2):
                        agent_reasoning = gr.TextArea(
                            label="Agent Reasoning",
                            placeholder="Agent's thoughts will be displayed here...",
                            lines=20,
                        )

                # Chat input at the bottom
                chat_input = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False,
                )

        # Memory Tab (New Implementation)
        with gr.Tab("Memory"):
            gr.Markdown("## Your Personal Memory Hub")

            with gr.Tabs() as memory_tabs:
                # Social Graph Section
                with gr.Tab("Social Graph"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            social_graph_html = gr.HTML(
                                create_social_graph(), label="Your Social Network"
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("### Person Details")
                            selected_person = gr.Dropdown(
                                choices=list(MOCK_SOCIAL_GRAPH.keys()),
                                label="Select Person",
                                value="Mike Knoop",
                            )

                            person_details = gr.JSON(
                                {
                                    "role": "Founder at Runway",
                                    "interests": ["AI", "Video Generation", "Design"],
                                    "relationship": "podcast guest",
                                    "strength": 7,
                                    "last_contact": "2 weeks ago",
                                    "notes": "Met during GD Episode #42, discussed AI video tools",
                                },
                                label="Details",
                            )

                            gr.Button("Update Details").click(
                                lambda: gr.update(visible=True), outputs=person_details
                            )

                # External Knowledge Sources
                with gr.Tab("External Knowledge Sources"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Available Sources")

                            with gr.Row():
                                gmail_btn = gr.Button("Connect Gmail")
                                calendar_btn = gr.Button("Connect Calendar")

                            with gr.Row():
                                slack_btn = gr.Button("Connect Slack")
                                whatsapp_btn = gr.Button("Connect WhatsApp")

                            external_source_display = gr.Markdown(
                                "No external sources connected"
                            )

                            # Connect button actions
                            gmail_btn.click(
                                lambda: connect_external_source("Gmail"),
                                outputs=external_source_display,
                            )
                            calendar_btn.click(
                                lambda: connect_external_source("Calendar"),
                                outputs=external_source_display,
                            )
                            slack_btn.click(
                                lambda: connect_external_source("Slack"),
                                outputs=external_source_display,
                            )
                            whatsapp_btn.click(
                                lambda: connect_external_source("WhatsApp"),
                                outputs=external_source_display,
                            )

                # Active Memory
                with gr.Tab("Active Memory"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Recent Activities")

                            memory_source = gr.Radio(
                                ["Cursor Code Snippet", "Slack Message"],
                                label="Select Memory Source",
                                value="Cursor Code Snippet",
                            )

                            fetch_btn = gr.Button("Fetch Memory")

                            active_memory_display = gr.TextArea(
                                label="Memory Content",
                                placeholder="Select a source and fetch memory...",
                                lines=15,
                                interactive=False,
                            )

                            # Fetch button action
                            fetch_btn.click(
                                load_active_memory_and_trigger_workflow,
                                inputs=[
                                    memory_source,  # Pass the actual component, not a lambda function
                                    current_editor_content,
                                    app_phase,
                                    question_counter,
                                    current_context_tags,
                                    current_preference_tags,
                                    current_execution_step,
                                ],
                                outputs=[
                                    active_memory_display,  # Display memory content
                                    current_editor_content,  # Set new query in editor
                                    app_phase,  # Reset to discovery phase
                                    question_counter,  # Reset question counter
                                    current_context_tags,  # Reset context tags
                                    current_preference_tags,  # Reset preference tags
                                    current_execution_step,  # Reset execution step
                                    tabs,  # Switch to AI Mediator tab
                                ],
                            )

                # Memory Chat
                with gr.Tab("Memory Chat"):
                    memory_chatbot = gr.Chatbot(
                        value=[[None, MEMORY_CHAT_RESPONSES["default"]]],
                        height=400,
                        avatar_images=(None, "https://freesvg.org/img/1538298822.png"),
                    )

                    memory_chat_input = gr.Textbox(
                        placeholder="Chat with your memory assistant...",
                        show_label=False,
                        container=False,
                    )

                    # Memory chat submission
                    memory_chat_input.submit(
                        memory_chat,
                        inputs=[memory_chat_input, memory_chatbot],
                        outputs=[memory_chatbot],
                    )

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

    # Update agent reasoning when execution step changes
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

    # Auto-progress execution steps
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

    # Update person details when selection changes
    selected_person.change(
        lambda person: MOCK_SOCIAL_GRAPH[person]["attributes"]
        | {
            "relationship": MOCK_SOCIAL_GRAPH["Lukas"]["connections"]
            .get(person, {})
            .get("type", ""),
            "strength": MOCK_SOCIAL_GRAPH["Lukas"]["connections"]
            .get(person, {})
            .get("strength", 0),
            "last_contact": MOCK_SOCIAL_GRAPH["Lukas"]["connections"]
            .get(person, {})
            .get("last_contact", ""),
            "notes": (
                "Met during GD Episode #42, discussed AI video tools"
                if person == "Mike Knoop"
                else ""
            ),
        },
        inputs=[selected_person],
        outputs=[person_details],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)
