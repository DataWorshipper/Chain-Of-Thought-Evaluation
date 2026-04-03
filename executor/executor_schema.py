from pydantic import BaseModel,Field
from typing import Annotated,List,Optional,TypedDict
import operator
from langchain_core.output_parsers import PydanticOutputParser

class ExecutorOutput(BaseModel):
  executor_answer: str = Field(
        description="Final numeric answer only (no explanation)"
    )
executor_parser = PydanticOutputParser(pydantic_object=ExecutorOutput)

class OverallState(TypedDict):
    filtered_tasks: List[dict]
    executor_results: Annotated[List[dict], operator.add]

class ExecutorState(TypedDict):
    task_id: int
    executor_id: int
    question: str
    correct_ans: str
    thinker_cot: str
    independent_ans: Optional[str]
    guided_ans: Optional[str]
    corrupted_cot: Optional[str] 
    corrupted_ans: Optional[str] 
    status: Optional[str]
    executor_results: List[dict]