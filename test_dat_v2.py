from lit_gpt.config import Config
from lit_gpt.datmodel_v2 import DualStreamGPTV2
from safetensors.torch import load_file

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

CKPT_PATH = "workdir/finetune/dat-336M-v2-reverse/iter-003200-ckpt.safetensors"
PARAM=336
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = f"Diff_LLaMA_{PARAM}M"
config = Config.from_name(model_name)
# tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                        #   padding_side="right", use_fast=True)
tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', padding_side="left", use_fast=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = 1

model = DualStreamGPTV2(config).to(DEVICE)
model.load_state_dict(load_file(CKPT_PATH))

dataset = load_dataset("zzy1123/ptr_follow_reverse_order_rl", split="train")

def get_question(prompt):
    return prompt[0]['content']

def get_reward(reward_dict):
    return reward_dict['ground_truth']

questions = [get_question(q) for q in dataset["prompt"]]
# right_answers = data["target"]
right_answers = [get_reward(r) for r in dataset["reward_model"]]

idx = 3
q = questions[idx]
ans = right_answers[idx]
ans = ans[:6]+'D'+ans[7:]
question_ids = tokenizer([q], padding="max_length", max_length=256, truncation=False, return_tensors="pt")['input_ids'].to('cuda')
resp_ids = tokenizer([ans], truncation=False, return_tensors="pt")['input_ids'].to('cuda')
# print(len(question_ids[0]), len(resp_ids[0]))
# print(resp_ids)
resp_ids = resp_ids[:, 1:]  # remove bos token
eos_ids = torch.full((resp_ids.shape[0], 1), tokenizer.eos_token_id, dtype=resp_ids.dtype, device=resp_ids.device)
resp_ids = torch.cat((resp_ids, eos_ids), dim=-1)

full_input_ids = torch.cat((question_ids, resp_ids), dim=-1)
with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
    logits = model(full_input_ids)
resp_logits = logits[:, -17:-1, :]
pred_ids = torch.argmax(resp_logits, dim=-1)
# print(resp_ids[:,:-1], pred_ids)

print(q)
print(ans)

pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
print("Predicted answer:", pred_text)