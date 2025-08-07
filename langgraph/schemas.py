from typing import TypedDict, List, Optional, Annotated, Dict, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class Intent(BaseModel):
    """User intent with context and preferences"""

    description: str = ""
    context: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)


class BlogDraft(BaseModel):
    """Draft of a blog post"""

    title: str = ""
    content: str = ""
    feedback: List[str] = Field(default_factory=list)


class State(TypedDict):
    """The state of the blog post collaboration process"""

    # Messages will be appended rather than overwritten
    messages: Annotated[list, add_messages]

    # Current phase of the process
    phase: str

    # Intent, context and preferences
    intent: Intent

    # Current version of the blog post
    draft: BlogDraft
