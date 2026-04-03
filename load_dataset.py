from datasets import load_dataset
from utilities import extract_answer

dataset = load_dataset("gsm8k", "main")
dataset = dataset.map(extract_answer)

