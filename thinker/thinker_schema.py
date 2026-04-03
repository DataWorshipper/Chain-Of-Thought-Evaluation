from typing import Annotated, TypedDict, List, Optional,NotRequired
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser

class ThinkerState(TypedDict):
    id: int
    question: str
    true_answer: str
    thinker_response: NotRequired[str]
    thinker_cot: NotRequired[str]
    thinker_answer: NotRequired[str]
    is_correct: NotRequired[int]

class ThinkerOutput(BaseModel):
    thinker_cot: str = Field(
        description="Step-by-step reasoning for solving the problem"
    )
    thinker_answer: str = Field(
        description="Final numeric answer only (no explanation)"
    )

thinker_parser = PydanticOutputParser(
    pydantic_object=ThinkerOutput
)


