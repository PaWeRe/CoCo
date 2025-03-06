import gradio as gr
import textwrap
from datetime import datetime


def generate_email(command, history):
    # Simulate the generation process
    # In a real implementation, this would connect to your AI model
    # Get current timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Generate content plan based on input
    content_plan = """
    Proposed Content:
    - Say thanks for being on PC
    - Mention program synth
    - Invite him to the M&M

    Proposed Preferences:
    - business casual
    """

    # Generate email based on the plan
    email_output = """
    Mike,

    just wanted to say thanks again for joining the podcast—really enjoyed your unique vision for the path to AGI and good luck for ndad! Also, wanted to invite you to our next Meet&Mingle event, would be great to catch up and connect with more folks. It's on Jan 17th, 5 PM at Freiheitshalle—hope you can make it!

    PS: attaching our lunch group photo, figured you might like it.

    Best,
    Lukas
    """
    # Update history
    new_history = (
        f"{history}\n{timestamp}: {command}" if history else f"{timestamp}: {command}"
    )
    return content_plan.strip(), email_output.strip(), new_history


def process_input(command, history):
    content_plan, email, new_history = generate_email(command, history)
    return content_plan, email, new_history


# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Mediator MVP")

    with gr.Row():
        # Timeline column (new)
        with gr.Column(scale=1):
            gr.Markdown("### Timeline")
            history_output = gr.Textbox(
                label="Generation History",
                lines=20,
                interactive=False,
                placeholder="Timeline will appear here...",
            )

        # Planning column
        with gr.Column(scale=1):
            gr.Markdown("### User")
            command_input = gr.Textbox(
                label="Command line",
                placeholder="Enter your command here...",
                value="Follow up with Mike",
            )
            gr.Markdown("### Plan")
            plan_output = gr.Textbox(
                label="Recommended generation plan", lines=10, interactive=True
            )
            # Button to generate
            generate_btn = gr.Button("Generate")

        # Output column
        with gr.Column(scale=1):
            gr.Markdown("### Generation")
            email_output = gr.Textbox(
                label="Output verification", lines=15, interactive=True
            )
            # Button to generate
            generate_btn = gr.Button("Re-Generate")

    # Handle generation
    generate_btn.click(
        fn=process_input,
        inputs=[command_input, history_output],
        outputs=[plan_output, email_output, history_output],
    )

    # Set initial state
    demo.load(
        fn=process_input,
        inputs=[command_input, history_output],
        outputs=[plan_output, email_output, history_output],
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
