from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("zzy1123/Diff_LLaMA_336M_sudoku_sft_v2_720", trust_remote_code=True)

text="Answer:4321123434122143"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs['input_ids'][0]
print("Input IDs:", input_ids)
print(f"Length of input IDs: {input_ids.shape[-1]}")