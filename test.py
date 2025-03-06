import gradio as gr
import textwrap
from datetime import datetime


def generate_email(command, history):
    # Get current timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Generate content plan and email as before
    content_plan = """
    Proposed Content:
    - Say thanks for being on PC
    - Mention program synth
    - Invite him to the M&M

    Proposed Preferences:
    - business casual
    """

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
    gr.Markdown("# Mediator MVP")

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

        # Main content columns
        with gr.Column(scale=2):
            with gr.Row():
                # Command and plan column
                with gr.Column():
                    command_input = gr.Textbox(
                        label="Command line",
                        placeholder="Enter your command here...",
                        value="Follow up with Mike",
                    )

                    plan_output = gr.Textbox(
                        label="Recommended generation plan", lines=10, interactive=False
                    )

                # Email output column
                with gr.Column():
                    email_output = gr.Textbox(
                        label="Output verification", lines=15, interactive=False
                    )

    # Button to generate
    generate_btn = gr.Button("Generate")

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
