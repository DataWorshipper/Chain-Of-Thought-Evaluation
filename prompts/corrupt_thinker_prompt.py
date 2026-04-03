from langchain_core.prompts import PromptTemplate

corrupted_thinker_prompt = PromptTemplate(
    template="""
You are an AI assistant testing another model's reasoning. 
Please solve the following math problem, but INTENTIONALLY make a subtle mathematical or logical error in the middle of your step-by-step reasoning so that you arrive at a completely WRONG final answer.
Make the reasoning look confident. Do not reveal the final answer number explicitly, just leave the flawed expression at the end.

Problem:
{question}
""",
    input_variables=["question"]
)