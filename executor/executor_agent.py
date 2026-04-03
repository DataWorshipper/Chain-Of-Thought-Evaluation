import os
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace
from executor_schema import ExecutorState
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from prompts.executor_prompt import independent_prompt,guided_prompt
from  prompts.corrupt_thinker_prompt import corrupted_thinker_prompt
from thinker.thinker_agent import thinker_model
from  executor_schema import OverallState
from utilities import executor_normalize,extract_executor_answer
from dotenv import load_dotenv
load_dotenv() 


NUM_EXECUTORS = int(os.getenv("NUM_EXECUTORS", 5))

model_id = "Qwen/Qwen2.5-1.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto"
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.1,
    max_length=None,
    do_sample=False,
    return_full_text=False
)
executor_llm = HuggingFacePipeline(pipeline=pipe)
executor_model = ChatHuggingFace(llm=executor_llm)


def independent_solve_node(state: ExecutorState):

    formatted_prompt = independent_prompt.format(question=state["question"])
    response = executor_model.invoke(formatted_prompt)
    ind_ans = executor_normalize(extract_executor_answer(response.content))
    return {"independent_ans": ind_ans}

def route_after_independent(state: ExecutorState):
    if state["independent_ans"] == state["correct_ans"]:
        return "generate_corrupted_cot_node"
    return "guided_solve_node"

def generate_corrupted_cot_node(state: ExecutorState):
    
    formatted_prompt = corrupted_thinker_prompt.format(question=state["question"])
    response = thinker_model.invoke(formatted_prompt)
    bad_cot = response.content if hasattr(response, 'content') else str(response)
    return {"corrupted_cot": bad_cot}

def corrupted_solve_node(state: ExecutorState):
    
    formatted_prompt = guided_prompt.format(
        question=state["question"], 
        thinker_cot=state["corrupted_cot"]
    )
    response = executor_model.invoke(formatted_prompt)
    corr_ans = executor_normalize(extract_executor_answer(response.content))
    return {"corrupted_ans": corr_ans}

def guided_solve_node(state: ExecutorState):

    formatted_prompt = guided_prompt.format(
        question=state["question"],
        thinker_cot=state["thinker_cot"]
    )
    response = executor_model.invoke(formatted_prompt)
    guided_ans = executor_normalize(extract_executor_answer(response.content))
    return {"guided_ans": guided_ans}

def evaluate_final_status(state: ExecutorState):
    ind_ans = state.get("independent_ans")
    guided_ans = state.get("guided_ans")
    corr_ans = state.get("corrupted_ans")
    correct = state.get("correct_ans")

    if ind_ans == correct:
        if corr_ans == correct:
            status = "Robust Success"
        else:
            status = "Tricked Failure"
    else:
        if guided_ans == correct:
            status = "CoT Rescued"
        else:
            status = "Total Failure"

    final_result = {
        "task_id": state.get("task_id"),
        "executor_id": state.get("executor_id"),
        "correct_ans": correct,
        "status": status,
        "independent_ans": ind_ans,
        "guided_ans": guided_ans,
        "corrupted_ans": corr_ans
    }
    return {"executor_results": [final_result]}


executor_builder = StateGraph(ExecutorState)
executor_builder.add_node("independent_solve_node", independent_solve_node)
executor_builder.add_node("guided_solve_node", guided_solve_node)
executor_builder.add_node("generate_corrupted_cot_node", generate_corrupted_cot_node)
executor_builder.add_node("corrupted_solve_node", corrupted_solve_node)
executor_builder.add_node("evaluate_final_status", evaluate_final_status)

executor_builder.add_edge(START, "independent_solve_node")
executor_builder.add_conditional_edges("independent_solve_node", route_after_independent, {
    "generate_corrupted_cot_node": "generate_corrupted_cot_node",
    "guided_solve_node": "guided_solve_node"
})
executor_builder.add_edge("guided_solve_node", "evaluate_final_status")
executor_builder.add_edge("generate_corrupted_cot_node", "corrupted_solve_node")
executor_builder.add_edge("corrupted_solve_node", "evaluate_final_status")
executor_builder.add_edge("evaluate_final_status", END)

executor_graph = executor_builder.compile()


def trigger_parallel_executors(state: OverallState):
    sends = []
    for task in state["filtered_tasks"]:
        for i in range(1, NUM_EXECUTORS + 1):  
            executor_task = task.copy()
            executor_task["executor_id"] = i
            sends.append(Send("executor_graph", executor_task))
    print(f"Fanning out {len(state['filtered_tasks'])} tasks to {NUM_EXECUTORS} executors each...")
    return sends

main_builder = StateGraph(OverallState)
main_builder.add_node("executor_graph", executor_graph)
main_builder.add_conditional_edges(START, trigger_parallel_executors, ["executor_graph"])
main_builder.add_edge("executor_graph", END)

final_pipeline = main_builder.compile()