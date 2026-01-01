import os
import json
from datasets import Dataset, DatasetDict, load_dataset

hf_path = "zzy1123/sudoku_sft"
dataset = load_dataset(hf_path, split="train")
target_path = "data/sudoku/train.txt"

with open(target_path, "w") as f:
    for example in dataset:
        input_text = example["input"]