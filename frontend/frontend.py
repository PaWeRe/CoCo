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
                "Provide a short summary of the intent and list two follow-up questions - only if needed."
            )
            analysis_output = call_openai_api(prompt)
            new_conversation = conversation + f"User (Discovery): {user_input}\nAgent (Analysis): {analysis_output}\n\n"
            # Set discovery subphase to clarification.
            new_disc_sub = "clarification"
            prefill = "Hi Lukas, we're talking about your latest GD episode with Mike Knoop, right?"
            # Return updated state; remain in discovery phase.
            return phase, new_disc_sub, analysis_output, clarified_intent, draft_editor, new_conversation, prefill, prefs, context

        elif disc_sub == "clarification":
            # Use the stored analysis_res and the user's clarification to produce a final clarified intent.
            prompt = (
                f"Combine the following messages:\n{clarified_intent + ": " + analysis_res}\n\n"
                f"with the user's clarification:\n{user_input}\n\n"
                f"Give three suggestions for good blog titles"
                #"Produce a final, clear statement of the user's intent."
            )
            final_intent = call_openai_api(prompt)
            new_conversation = conversation + f"User (Clarification): {user_input}\nAgent (Final Intent): {final_intent}\n\n"
            new_phase = "execution"
            # Save the final clarified intent.
            new_clarified_intent = final_intent
            prefill = """Cool, I thought about these three blog titles, what do you think?
            - **The New AI Paradigm: From Scaling Laws to Reasoning Systems**
            - **Benchmarking AI Intelligence: What the ARC Prize Taught Us**
            - **The Reliability Problem: Why AI Agents Still Struggle in Production**"""

            # user ansers: yes but make it about Weave too
            prefs += ["Promote Weave"]
            return new_phase, "", analysis_res, new_clarified_intent, draft_editor, new_conversation, prefill, prefs, context

    # EXECUTION PHASE
    elif phase == "execution":
        # TODO: manual for now (also code snippet from the beginning should be handled better, state machine with condisitons)
        with open("gd_episode_mike_knoop.txt", "r", encoding="utf-8") as file:
            podcast_transcript = file.read()

        code_template_cursor = """<article id="post-500" class="post-500 post type-post status-publish format-standard hentry category-uncategorized">
<header class="entry-header">
    <h1 class="entry-title"><a href="https://lukasbiewald.com/2019/06/24/starting-a-second-machine-learning-tools-company-ten-years-later/" rel="bookmark">Starting a Second Machine Learning Tools Company, Ten Years&nbsp;Later</a></h1>	</header><!-- .entry-header -->
            <div class="entry-meta">
        <span class="posted-on"><a href="https://lukasbiewald.com/2019/06/24/starting-a-second-machine-learning-tools-company-ten-years-later/" rel="bookmark"><time class="entry-date published" datetime="2019-06-24T00:30:50+00:00">June 24, 2019</time><time class="updated" datetime="2019-06-24T00:30:58+00:00">June 24, 2019</time></a></span><span class="byline"><span class="author vcard"><span class="sep"> ~ </span><a class="url fn n" href="https://lukasbiewald.com/author/lbiewald/">lbiewald</a></span></span>								</div><!-- .entry-meta -->
            <div class="entry-content">
    <p id="6e5f" class="graf graf--p graf-after--h3">I’ve spent the last six months heads down building a new machine learning tool called <a class="markup--anchor markup--p-anchor" href="http://wandb.com/" target="_blank" rel="nofollow noopener">Weights and Biases</a> with my longtime cofounder Chris Van Pelt, my new cofounder and friend Shawn Lewis and brave early users at Open AI, Toyota Research, Uber and others. Now that it’s public I wanted to talk a little bit about why I’m (still) so excited about building machine learning tools.</p>
<p id="1eae" class="graf graf--p graf-after--p">I remember the magic I felt training my first machine learning algorithm. It was 2002 and I was taking Stanford’s 221 class from Daphne Koller. I had procrastinated so I spent 72 hours straight in the computer lab building a reinforcement learning algorithm that played game after game of Othello against itself. The algorithm started off incredibly dumb, but I kept fiddling and watching the computer struggle to play on my little ASCII terminal. In the middle of the night, something clicked and it started getting better and better, blowing past my own skill level. It felt like breathing life into a machine. I was hooked.</p>
<p id="4766" class="graf graf--p graf-after--p">When I worked as a TA in Daphne’s lab a few years later during grad school, it seemed like nothing in ML was working. The now famous NIPS conference had just a few hundred attendees. I remember Mike Montemerlo and Sebastian Thrun had to work to get skeptical grad students excited about a self-driving car project. Out in the world, AI was mostly being used to rank ads.</p>
<figure id="4bfd" class="graf graf--figure graf-after--p">
<div class="aspectRatioPlaceholder is-locked">
<div class="aspectRatioPlaceholder-fill"></div>
<div class="progressiveMedia js-progressiveMedia graf-image is-canvasLoaded is-imageLoaded"><img class="progressiveMedia-image js-progressiveMedia-image" src="https://lukasbiewald.com/wp-content/uploads/2019/06/45a6d-1s4l3hwkupbyryjp3nacmxw.png?w=656" /></div>
</div><figcaption class="imageCaption">Unveiling CrowdFlower at Tech Crunch in 2009</figcaption></figure>
<p id="0376" class="graf graf--p graf-after--figure">After working on search for a few years, by 2007 it was clear to me that the biggest problem in Machine Learning in every company and lab was access to training data. I left my job to start CrowdFlower (now Figure Eight) to solve that problem. Every researcher knew that access to training data was a major problem, but outside of research it wasn’t yet clear at all. We made tens of millions of dollars creating training data sets for everything from eBay’s product recommendations to instagram’s support ticket classification but until around 2016, nearly all VCs were adamant that machine learning wasn’t a legitimate vertical worth targeting.</p>
<p id="0a23" class="graf graf--p graf-after--p">Ten years later the company is thriving and spawned a field full of competitors. But it turned out that one of our core doctrines was wrong. <span class="markup--quote markup--p-quote is-other">My strong bias was always that algorithms don’t matter. Over and over I had worked with people who promised a magic new breakthrough algorithm that would fundamentally change the way AI worked. It was never true.</span> It was painful watching companies pour resources into improving algorithms <span class="markup--quote markup--p-quote is-other">when simply collecting more training data would have had a much bigger impact.</span></p>
<figure id="2edb" class="graf graf--figure graf-after--p">
<div class="aspectRatioPlaceholder is-locked">
<div class="aspectRatioPlaceholder-fill"></div>
<div class="progressiveMedia js-progressiveMedia graf-image is-canvasLoaded is-imageLoaded"><img class="progressiveMedia-image js-progressiveMedia-image" src="https://cdn-images-1.medium.com/max/1067/0*MybdbjYDvNMb255W." /></div>
</div><figcaption class="imageCaption">Training data has become a mainstream concept (apparently I find that painful?)</figcaption></figure>
<p id="fd12" class="graf graf--p graf-after--figure">The first sign something had changed came in 2012, when I heard from Anthony Goldbloom that neural nets — the darling of 70s-era AI professors — were winning Kaggle competitions. In 2013 and 2014 we started seeing an explosion of image labeling tasks at CrowdFlower. It became undeniable that these “new” algorithms that people were calling deep learning were working in practical ways on applications where ML had never worked before.</p>
<figure id="b5c5" class="graf graf--figure graf-after--p">
<div class="aspectRatioPlaceholder is-locked">
<div class="aspectRatioPlaceholder-fill"></div>
<div class="progressiveMedia js-progressiveMedia graf-image is-canvasLoaded is-imageLoaded"><img class="progressiveMedia-image js-progressiveMedia-image" src="https://lukasbiewald.com/wp-content/uploads/2019/06/4f05c-1zpxxvfm-huir7i-msr54nw.png?w=656" /></div>
</div><figcaption class="imageCaption">These cheap robots do object recognition better than any supercomputer on the planet just a few years ago.</figcaption></figure>
<p id="7076" class="graf graf--p graf-after--figure">I stepped down as CEO of Figure Eight and <span class="markup--quote markup--p-quote is-other">went about building my technical chops in deep learning. I spent days in my garage, building robots running TensorFlow on a Raspberry Pi. My friend Adrien Treuille and I locked ourselves in an Airbnb and i</span><span class="markup--quote markup--p-quote is-other">mplemented backpropagation for perceptrons and then Convolutional Neural Nets and then more complicated models</span><span class="markup--quote markup--p-quote is-other">. To sharpen my thinking, I taught “introduction to deep learning classes” to thousands of engineers. I somehow got myself an internship at OpenAI and got to work with some of the best people in the world. I pair programmed with twenty-four year old grad students who intimidated the hell out of me.</span></p>
<p id="5d07" class="graf graf--p graf-after--p">Stepping back into being a practitioner gave me a view on a new set of problems. When you write (non-AI/ML) code directly, you can walk through what it does. You can diff it and version it in a meaningful way. Debugging is never easy, but we have seventy years of debugging expertise behind us and we’ve built an amazing array of tools and best practices to do it well.</p>
<figure id="29ae" class="graf graf--figure graf-after--p">
<div class="aspectRatioPlaceholder is-locked">
<div class="aspectRatioPlaceholder-fill"></div>
<div class="progressiveMedia js-progressiveMedia graf-image is-canvasLoaded is-imageLoaded"><img class="progressiveMedia-image js-progressiveMedia-image" src="https://cdn-images-1.medium.com/max/1067/0*-O9x4LkZA8yZcRRU." /></div>
</div><figcaption class="imageCaption">Machine Learning classes for engineers are super popular.</figcaption></figure>
<p id="fbe8" class="graf graf--p graf-after--figure"><span class="markup--quote markup--p-quote is-other">With machine learning, we’re starting over.</span><span class="markup--quote markup--p-quote is-other"> Instead of programming the computer directly, we write code that guides the computer to create a model. We can’t modify the model directly or even easily understand how it does what it does. Diffs between versions of the model don’t make sense to humans: if I change the functionality even slightly, every single bit in the model will likely be different.</span> From my experience at Figure Eight, I knew all the machine learning teams were having the same problem. All of the problems machine learning always had are becoming worse with deep learning. Training data is still critically important, but because of this poor tooling, many teams that should be deploying a new model every day are lucky if they deploy twice a month.</p>
<p id="8574" class="graf graf--p graf-after--p">I started Weights and Biases because, for the second time in my career, I have deep conviction about what the AI field needs. <span class="markup--quote markup--p-quote is-other">Ten years ago training data was the biggest problem holding back real world machine learning. Today, the biggest pain is a lack of basic software and best practices to manage a completely new style of coding</span>. <a class="markup--anchor markup--p-anchor" href="https://medium.com/@karpathy" target="_blank" rel="noopener">Andrej Karpathy</a> describes machine learning as the new kind of programming that needs a reinvented IDE. <a class="markup--anchor markup--p-anchor" href="https://petewarden.com/" target="_blank" rel="nofollow noopener noopener">Pete Warden </a>writes about AI’s reproducibility crisis — there’s no version control for machine learning models and it’s incredibly hard to reproduce one’s own work let alone some else’s. As machine learning rapidly evolves from research projects to critical real-world deployed software we suddenly have an acute need for a new set of developer tools.</p>
<figure id="3e3b" class="graf graf--figure graf-after--p">
<div class="aspectRatioPlaceholder is-locked">
<div class="aspectRatioPlaceholder-fill"></div>
<div class="progressiveMedia js-progressiveMedia graf-image is-canvasLoaded is-imageLoaded"><img class="progressiveMedia-image js-progressiveMedia-image" src="https://lukasbiewald.com/wp-content/uploads/2019/06/6894d-1r_e-de4biafya6s_aauvaa.png?w=656" /></div>
</div><figcaption class="imageCaption">Face Recognizing drone tracks down Chris</figcaption></figure>
<p id="8289" class="graf graf--p graf-after--figure">Working on deep learning, I had that same sense of wonder — that I was breathing life into a machine — that had first hooked me on to machine learning. Machine learning has the potential to solve the world’s biggest problems. In just the past couple of years, image recognition went from unsolvable to solved, voice recognition became a household applianc<span class="markup--quote markup--p-quote is-other">e. Like Pete Warden said, software is eating the world and deep learning is eating software.</span></p>
<p id="87e5" class="graf graf--p graf-after--p">I love working with people working on machine learning. In my view the work they do has the highest potential to impact the world and I want to build them tools to help them do that. Like every powerful technology machine learning will create lots of problems to wrestle with. <span class="markup--quote markup--p-quote is-other">Every machine learning practitioner I know wants their models to be safe, fair and reliable.</span> Today, that’s really hard to do.</p>
<p id="3ff2" class="graf graf--p graf-after--p">You can’t paint well with a crappy paintbrush, you can’t write code well in a crappy IDE, and you can’t build and deploy great deep learning models with the tools we have now. I can’t think of any more important goal than changing that.</p>
<p id="e9e9" class="graf graf--p graf-after--p"><em class="markup--em markup--p-em">Check out Weights &amp; Biases at </em><a class="markup--anchor markup--p-anchor" href="http://wandb.com/" target="_blank" rel="nofollow noopener">wandb.com</a><em class="markup--em markup--p-em">.</em></p>
<p id="62b9" class="graf graf--p graf-after--p graf--trailing"><em class="markup--em markup--p-em">Thanks Noga Leviner, Michael E. Driscoll, </em><a class="markup--anchor markup--p-anchor" href="http://yanda.com/" target="_blank" rel="nofollow noopener"><em class="markup--em markup--p-em">Yanda Erlich</em></a><em class="markup--em markup--p-em">,Will Smith and James Cham for feedback on early drafts.</em></p>
        </div><!-- .entry-content -->
</article><!-- #post-## -->"""
            
        if draft_editor == "":
            # Generate an initial draft based on the clarified intent.
            prompt = (
                f"Based on the clarified intent:\n'{clarified_intent}'\n"
                f"with Preferences: {', '.join(prefs) if prefs else 'None'} and Context: {', '.join(context) if context else 'None'}\n\n"
                f"Transcript from the podcast: {podcast_transcript}"
                "Generate an initial draft."
            )
            draft = call_openai_api(prompt)
            new_conversation = conversation + f"Agent (Initial Draft): {draft}\n\n"
            new_draft = draft
            prefill = "Great, I wrote up a first draft with OpenAI on focusing on the shift to reasoning models and bringing in Weave:\n\n 'Enhancing AI Reliability: How W&B Weave Supports the Shift Toward Reasoning Models'\n\n Any feedback?"
            return phase, disc_sub, analysis_res, clarified_intent, new_draft, new_conversation, prefill, prefs, context
        else:
            # User provides feedback to refine the draft.
            prompt = (
                f"Refine the following draft:\n'{draft_editor}'\n"
                f"Using the feedback: '{user_input}'\n"
                f"while maintaining the clarified intent: '{clarified_intent}',\n"
                f"Preferences: {', '.join(prefs) if prefs else 'None'}, and Context: {', '.join(context) if context else 'None'}.\n\n"
                f"The template code to put the blog post into: {code_template_cursor}"
                "Produce a refined draft in case user gives feedback."
            )
            refined_draft = call_openai_api(prompt)
            new_conversation = conversation + f"User Feedback: {user_input}\nAgent (Refined Draft): {refined_draft}\n\n"
            new_draft = refined_draft
            new_phase = "verification"
            prefill = "Got it, I added the blog post into the code template that you gave me in the beginning."
            return new_phase, disc_sub, analysis_res, clarified_intent, new_draft, new_conversation, prefill, prefs, context

    # VERIFICATION PHASE
    elif phase == "verification":
        prompt = (
            f"Review the following draft:\n'{draft_editor}'\n"
            f"last user feedback: '{user_input}',\n"
            #f"Preferences: {', '.join(prefs) if prefs else 'None'}, and Context: {', '.join(context) if context else 'None'}.\n\n"
            "Incorporate last feedback if necessary."
        )
        #final_output = call_openai_api(prompt)
        final_output = draft_editor
        new_conversation = conversation + f"Agent (Final Output): {final_output}\n\n"
        new_phase = "completed"
        prefill = "Cool, great working with you Lukas! I can send it back to Cursor or you can just copy it."
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
    gr.Markdown("# Personal AI Mediator")
    
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
                lines=4,
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