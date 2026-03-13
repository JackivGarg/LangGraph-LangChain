from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing_extensions import Literal, Optional
from langgraph.graph.message import add_messages
from typing import Annotated


class Route1(BaseModel):
    action: Literal[
        "TAVILY_SEARCH",
        "VECTOR_STORE",
        "SEND_EMAIL",
        "STOP"
    ]
    category: Literal[
        None,
        "admissions",
        "programs",
        "hostel",
        "placements",
        "policies",
        "general"
    ] = None


class State(TypedDict):
    messages: Annotated[List, add_messages]
    input: str
    decision: str
    output: str
    action: str
    category: str
