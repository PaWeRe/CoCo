import os
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.types import interrupt, Command
from langchain_core.tools import InjectedToolCallId, tool
from dotenv import load_dotenv
from typing import Annotated

# from schemas import Message as SchemaMessage

# Try to load environment variables from multiple possible locations
env_paths = ["../backend/.env", "./.env", "../.env", "../../.env"]

env_loaded = False
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        env_loaded = True
        break

# Check for required API keys
if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    print("Please set this in your environment or in a .env file.")
    print("You can create a .env file with the following content:")
    print("OPENAI_API_KEY=your_api_key_here")

# Initialize LLM model
try:
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
    )
except Exception as e:
    print(f"Error initializing LLM model: {e}")
    print("Please check your API key and internet connection.")

    # Create a fallback model that will print helpful error messages
    class FallbackModel:
        def invoke(self, *args, **kwargs):
            return AIMessage(
                content="Error: Could not initialize LLM model. Please check your API key and internet connection."
            )

        def bind_tools(self, *args, **kwargs):
            return self

    model = FallbackModel()

# Define system prompts for different phases
DISCOVERY_PROMPT = """You are in the DISCOVERY phase of creating a personalized blog post.
Your goal is to understand what kind of blog post the user wants to create.

Ask clarifying questions about:
1. The main topic of the blog post
2. The target audience
3. The tone and style preferences
4. Any specific context or information to include

Be conversational and helpful. Gather enough information to move to the EXECUTION phase.
You can use the retrieval_agent tool to gather relevant context from available documents.
"""

EXECUTION_PROMPT = """You are in the EXECUTION phase of creating a personalized blog post.
Your goal is to coordinate the creation of a high-quality blog post draft based on the information gathered.

You can:
1. Use the drafting_agent tool to create or revise the blog post draft
2. Ask the user for more information or clarification if needed
3. Use the retrieval_agent tool to gather additional context

Make sure the blog post meets the user's requirements before proceeding to the VERIFICATION phase.
"""

VERIFICATION_PROMPT = """You are in the VERIFICATION phase of creating a personalized blog post.
Your goal is to ensure the blog post meets the user's requirements and expectations.

You can:
1. Use the reviewing_agent tool to analyze and provide feedback on the current draft
2. Ask the user for feedback on the draft
3. Decide whether to return to the EXECUTION phase for revisions or proceed to finalizing the post

Help the user make the final decisions on the blog post content.
"""

FINALIZATION_PROMPT = """You are in the FINALIZATION phase of creating a personalized blog post.
Your goal is to prepare the finalized blog post for integration into the user's website.

You can:
1. Use the integration_agent tool to prepare the blog post for integration
2. Get final approval from the user
3. Confirm all requirements have been met

Ensure everything is ready for the final integration of the blog post into the website.
"""


def create_mediator_agent(tools):
    """Creates a mediator agent with the provided tools"""
    # Bind the tools to the llm
    llm_with_tools = model.bind_tools(tools)

    def mediator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
        """Mediator agent that manages the overall process"""
        # Get the current system prompt based on the phase
        if state["phase"] == "discovery":
            system_prompt = DISCOVERY_PROMPT
        elif state["phase"] == "execution":
            system_prompt = EXECUTION_PROMPT
        elif state["phase"] == "verification":
            system_prompt = VERIFICATION_PROMPT
        elif state["phase"] == "finalization":
            system_prompt = FINALIZATION_PROMPT
        else:
            system_prompt = DISCOVERY_PROMPT

        # Update the system prompt in the state
        state["sys_prompt"] = system_prompt

        # Create the messages for the LLM
        messages = [SystemMessage(content=system_prompt)]

        # Add all the previous messages
        for msg in state["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "tool":
                messages.append(
                    ToolMessage(
                        content=msg["content"], tool_call_id=msg.get("tool_call_id", "")
                    )
                )

        # Invoke the LLM
        response = llm_with_tools.invoke(messages)

        # Return the response to update the state
        return {"messages": [{"role": "assistant", "content": response.content}]}

    return mediator_agent


def retrieval_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent for retrieving contextual information"""
    query = state["messages"][-1]["content"]

    # Create the system prompt for the retrieval agent
    system_prompt = """You are a specialized retrieval agent. Your task is to find and provide relevant context 
    based on the user's query. Access documents like 'lukas_website.html' and 'gd_episode_david_cahn.txt' 
    when needed, and provide the most pertinent information for blog post creation."""

    # Simulate retrieving context (in a real implementation, you would search actual documents)
    # For this example, we'll just have the LLM generate some representative content
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"I need to find relevant context for this query: {query}. Please simulate retrieving information from appropriate documents."
        ),
    ]

    # Get response from LLM
    response = model.invoke(messages)

    # Return the tool response
    return {
        "messages": [
            {
                "role": "tool",
                "content": response.content,
                "tool_call_id": "retrieval_agent",
            }
        ]
    }


def drafting_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent for drafting blog content"""
    # Extract intent and preferences from state
    intent = state["intent"]
    draft = state["draft"]

    # Create system prompt for drafting
    system_prompt = """You are a specialized blog post writer. 
    Your task is to draft high-quality blog content based on the user's intent, context, and preferences.
    Write in a clear, engaging style and format the content appropriately for a blog post."""

    # Create content prompt with existing draft if available
    human_content = f"""
    Write a blog post with the following specifications:
    
    INTENT: {intent.get('description', '')}
    CONTEXT: {', '.join(intent.get('context', []))}
    PREFERENCES: {', '.join(intent.get('preferences', []))}
    
    Previous draft content (if any):
    {draft.get('content', '')}
    
    Write or refine this blog post draft.
    """

    # Invoke the LLM
    response = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
    )

    # Process the response to extract title and content
    content = response.content

    # Extract a title if none exists
    title = draft.get("title", "")
    if not title and content:
        lines = content.split("\n")
        # Look for a line that could be a title (first non-empty line)
        for line in lines:
            if line.strip():
                title = line.strip("# ").strip()
                content = "\n".join(lines[1:]).strip()
                break

    # Update the draft in the state
    updated_draft = {
        "title": title,
        "content": content,
        "feedback": draft.get("feedback", []),
    }

    # Return the draft update and tool response
    return {
        "draft": updated_draft,
        "messages": [
            {
                "role": "tool",
                "content": f"Draft created/updated: '{title}'",
                "tool_call_id": "drafting_agent",
            }
        ],
    }


def reviewing_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent for reviewing and providing feedback on blog drafts"""
    draft = state["draft"]

    # Create system prompt for reviewing
    system_prompt = """You are a specialized blog post reviewer and editor.
    Your task is to critically analyze the draft and provide constructive feedback for improvement.
    Consider content quality, organization, tone, style, and adherence to the specified intent and preferences."""

    # Create the review request
    human_content = f"""
    Please review this blog post draft:
    
    TITLE: {draft.get('title', '')}
    
    CONTENT:
    {draft.get('content', '')}
    
    INTENT: {state['intent'].get('description', '')}
    PREFERENCES: {', '.join(state['intent'].get('preferences', []))}
    
    Provide thorough feedback focusing on:
    1. Content quality and accuracy
    2. Organization and flow
    3. Style and tone
    4. Grammar and clarity
    5. Adherence to the stated intent and preferences
    
    Be specific and constructive in your critique.
    """

    # Invoke the LLM
    response = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
    )

    # Add the feedback to the draft
    feedback = draft.get("feedback", [])
    feedback.append(response.content)

    updated_draft = {
        "title": draft.get("title", ""),
        "content": draft.get("content", ""),
        "feedback": feedback,
    }

    # Return the updated draft and tool response
    return {
        "draft": updated_draft,
        "messages": [
            {
                "role": "tool",
                "content": response.content,
                "tool_call_id": "reviewing_agent",
            }
        ],
    }


def integration_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent for integrating the blog post into the website"""
    draft = state["draft"]

    # Create system prompt for integration
    system_prompt = """You are a web integration specialist. 
    Your task is to integrate a new blog post into the existing HTML structure of a website.
    Identify the appropriate section for blog posts and add the new content with the correct HTML formatting."""

    # Try to load a sample of the website if available
    website_sample = ""
    try:
        with open("../backend/lukas_website.html", "r", encoding="utf-8") as f:
            website_sample = f.read(5000)  # First 5000 chars as reference
    except:
        website_sample = "<This is a placeholder for the website HTML structure>"

    # Create the integration request
    human_content = f"""
    I need to integrate this blog post into the existing website HTML.
    
    BLOG TITLE: {draft.get('title', '')}
    BLOG CONTENT: {draft.get('content', '')}
    
    Here is a portion of the existing website HTML structure:
    ```html
    {website_sample}
    ```
    
    Generate HTML code that I can use to add this blog post to the website. 
    Make sure it matches the style and format of other blog entries.
    Only provide the HTML for the new blog post section, formatted to fit into the existing structure.
    """

    # Invoke the LLM
    response = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
    )

    # Return the integration proposal
    return {
        "website_integration": response.content,
        "messages": [
            {
                "role": "tool",
                "content": "Website integration proposal generated.",
                "tool_call_id": "integration_agent",
            }
        ],
    }


def human_assistance_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """Tool to get human input when needed"""
    query = state["messages"][-1]["content"]

    # Create a message asking for human input
    human_query = f"I need your input on: {query}"

    # Return the interrupt command
    return interrupt(Command(name="human_feedback", args={"query": human_query}))


def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Use this to ask the human for assistance."""
    human_response = interrupt(
        {"query": "I need your input on:", "name": name, "birthday": birthday}
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        human_response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)


# Define all the available tools
def get_tools():
    """Returns the list of tools available to the mediator agent"""
    return [
        {
            "type": "function",
            "function": {
                "name": "retrieval_agent",
                "description": "Use this tool to retrieve relevant context and information for the blog post.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query or context you're looking for",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "drafting_agent",
                "description": "Use this tool to create or update the blog post draft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instructions": {
                            "type": "string",
                            "description": "Specific instructions for creating or updating the draft",
                        }
                    },
                    "required": ["instructions"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reviewing_agent",
                "description": "Use this tool to review and provide feedback on the current blog post draft.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus_areas": {
                            "type": "string",
                            "description": "Specific areas to focus on during the review",
                        }
                    },
                    "required": ["focus_areas"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "integration_agent",
                "description": "Use this tool to prepare the blog post for integration into the website.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "placement": {
                            "type": "string",
                            "description": "Information about where to place the blog post",
                        }
                    },
                    "required": ["placement"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "human_assistance",
                "description": "Use this tool when you need human input or clarification.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question or query for the human",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
    ]
