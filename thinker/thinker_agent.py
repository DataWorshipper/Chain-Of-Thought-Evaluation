import os
import re
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END

from utilities import normalize, extract_json, extract_last_number_strong, is_invalid_answer, remove_answer_from_cot
from load_dataset import dataset
from thinker_schema import ThinkerState, ThinkerOutput, thinker_parser
from prompts import thinker_prompt

load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

thinker_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.2,
    max_new_tokens=768,
    do_sample=False,
    huggingfacehub_api_token=hf_api_key
)
thinker_model = ChatHuggingFace(llm=thinker_llm)

def thinker_agent(state: ThinkerState):
    q = state["question"]
    gt = state["true_answer"]

    formatted_prompt = thinker_prompt.format(question=q)
    response = thinker_model.invoke(formatted_prompt)
    
    if isinstance(response, str):
        raw_text = response.strip()
    else:
        raw_text = response.content.strip()

    cot, answer = None, None

    try:
        json_text = extract_json(raw_text)
        if json_text:
            parsed = thinker_parser.parse(json_text)
            cot = parsed.thinker_cot
            answer = parsed.thinker_answer
    except Exception:
        pass

    candidate = extract_last_number_strong(raw_text)

    if is_invalid_answer(answer):
        match = re.search(r"####\s*([-0-9.,]+)", raw_text)
        if match:
            answer = match.group(1)

    if is_invalid_answer(answer):
        answer = candidate

    if cot is None:
        cot = raw_text

    cot = remove_answer_from_cot(cot, answer)

    norm_pred = normalize(answer)
    norm_gt = normalize(gt)

    is_correct = int(norm_pred == norm_gt) if norm_pred and norm_gt else 0

    return {
        "thinker_response": raw_text,
        "thinker_cot": cot,
        "thinker_answer": answer,
        "is_correct": is_correct
    }

builder = StateGraph(ThinkerState)
builder.add_node("thinker_node", thinker_agent)
builder.add_edge(START, "thinker_node")
builder.add_edge("thinker_node", END)
thinker_graph = builder.compile()

