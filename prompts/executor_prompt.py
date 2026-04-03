from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from  executor.executor_schema import executor_parser

independent_prompt = PromptTemplate(
    template="""
You are a rigorous math solver. Solve the problem and provide the final numeric answer.
{format_instructions}

Problem:
{question}
""",
    input_variables=["question"],
    partial_variables={"format_instructions": executor_parser.get_format_instructions()}
)

guided_prompt = PromptTemplate(
    template="""
You are a rigorous math solver. Solve the problem using the provided step-by-step reasoning.
{format_instructions}

Problem:
{question}

Helpful Reasoning:
{thinker_cot}
""",
    input_variables=["question", "thinker_cot"],
    partial_variables={"format_instructions": executor_parser.get_format_instructions()}
)