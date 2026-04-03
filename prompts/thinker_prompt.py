from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from thinker.thinker_schema import ThinkerOutput

thinker_parser = PydanticOutputParser(
    pydantic_object=ThinkerOutput
)
thinker_prompt = PromptTemplate(
    template="""
You are a careful and rigorous math problem solver.

Your task is to solve the problem step by step.

STRICT RULES:
1. First, write detailed step-by-step reasoning.
2. DO NOT reveal the final answer anywhere in the reasoning.
3. The reasoning must NOT contain the final numeric answer.
4. You MUST always provide a final answer.
5. Even if you are unsure or think your reasoning may be incorrect, you MUST still give your best possible final answer.
6. The final answer must be a single number (no units, no explanation).
7. Your response is INVALID if thinker_answer is missing.

OUTPUT FORMAT (STRICT JSON):
{format_instructions}

Problem:
{question}
""",
    input_variables=["question"],
    partial_variables={
        "format_instructions": thinker_parser.get_format_instructions()
    }
)