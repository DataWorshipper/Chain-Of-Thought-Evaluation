import re
import os
from  executor.executor_schema import executor_parser
from executor.executor_schema import OverallState

NUM_EXECUTORS=os.getenv["NUM_EXECUTORS"]

def normalize(x):
    if x is None:
        return None
    return re.sub(r"[^\d\.-]", "", str(x))


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def is_invalid_answer(ans):
    if ans is None:
        return True
    ans = str(ans).strip().lower()
    return ans in ["", "thinker_answer", "missing"]


def remove_answer_from_cot(cot, answer):
    if cot and answer:
        cot = re.sub(rf"\b{re.escape(str(answer))}\b", "", cot)
    return cot.strip() if cot else cot


def extract_last_number_strong(text):
    tail = text[-200:]
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", tail)
    return nums[-1] if nums else None

def extract_answer(example):
    ans = example["answer"]
    match = re.search(r"####\s*([-0-9.,]+)", ans)
    example["final_answer"] = match.group(1) if match else None
    return example

def extract_executor_answer(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = executor_parser.parse(match.group(0))
            return parsed.executor_answer
    except Exception:
        pass

    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text[-100:])
    return nums[-1] if nums else None

def executor_normalize(x):
    if x is None:
        return None
    return re.sub(r"[^\d\.-]", "", str(x))

