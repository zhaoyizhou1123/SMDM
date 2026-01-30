import torch
import argparse
import re

from lit_gpt.model_cache import Config
from lit_gpt.trimodel import TriGPT
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_file

from eval.gen_model_answer import trigpt_sample
from evaluate_diff import set_seed
from eval.math_normalization import normalize_final_answer, check_sympy_equivalence


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        default=16,
        type=int,
    )
    parser.add_argument(
        "--model",
        default=1028,
        type=int,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
    )
    parser.add_argument(
        "--length",
        type=int,
        default=512
    )
    parser.add_argument(
        "--cfg1",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--cfg2",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--temperature",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        '--alg',
        type=str,
        default='greddy'
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default='zzy1123/ptr_follow_reverse_order_rl'
    )
    args = parser.parse_args()
    return args


def get_trigpt_sample(args, question, model, tokenizer):
    print(f"Question: {question}")
    question_ids = tokenizer(question, padding="max_length", max_length=256, truncation=False, return_tensors="pt")['input_ids'].to('cuda')
    answer_ids, history = trigpt_sample(model,
                             tokenizer,
                             question_ids,
                             alg=args.alg,
                             steps=args.steps,
                             temperature=args.temperature,
                             cfg_scale=args.cfg1,
                             response_length=args.length,
                             device='cuda')
    answer = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
    history = torch.concatenate(history, dim=0)
    print(f"History shape: {history.shape}")
    history = tokenizer.batch_decode(history, skip_special_tokens=False)
    # print(f"Answer length: {answer_ids}")

    # prefix_ids = tokenizer(prefix, padding="longest", truncation=True, return_tensors="pt")['input_ids'].to('cuda')
    # answer_ids = diff_sample(model,
    #                          tokenizer,
    #                          prefix_ids,
    #                          alg='greddy',
    #                          steps=args.steps,
    #                          temperature=args.temperature,
    #                          cfg_scale=args.cfg2,
    #                          context_length=args.length,
    #                          device='cuda')
    # answer = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)

    return answer, history


# def get_acc(pred, right_answer):
#     pattern = "#### (.*)$"

#     preds = re.findall(pattern, pred)
#     pred = preds[-1] if len(preds) >= 1 else ""

#     pred = normalize_final_answer(pred)
#     right_answer = normalize_final_answer(right_answer)
#     return check_sympy_equivalence(pred, right_answer)

def get_acc(pred, right_answer):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, pred, re.DOTALL)
    if matches:
        extracted_sol = matches[-1].strip()
    else:
        extracted_sol = ""
    return extracted_sol == right_answer

def get_question(prompt):
    return prompt[0]['content']

def get_reward(reward_dict):
    return reward_dict['ground_truth']

def get_dataset(data_path):
    dataset = load_dataset(data_path, split="train")
    return dataset



if __name__ == "__main__":
    args = get_args()
    set_seed(1234)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = f"Diff_LLaMA_{args.model}M"
    config = Config.from_name(model_name)
    # tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                            #   padding_side="right", use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', padding_side="left", use_fast=True)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = 0
    tokenizer.add_special_tokens({'mask_token': ' m'})

    model = TriGPT(config).to(device)
    if args.ckpt_path.endswith('.safetensors'):
        model.load_state_dict(load_file(args.ckpt_path))
    else:
        with open(args.ckpt_path, 'rb') as f:
            state_dict = torch.load(f)['model']
        model.load_state_dict(state_dict)

    acc = 0
    num = 0
    batch_size = 1

    # dataset = load_dataset('json', data_files='data/gsm8k/test.jsonl')
    dataset = get_dataset(args.data_path)
    # length = len(dataset['train'])
    length = len(dataset)
    iter = length // batch_size if length % batch_size == 0 else length // batch_size + 1

    for i in range(iter):
        end_index = (i + 1) * batch_size if (i + 1) * batch_size < length else length
        data = dataset[i * batch_size: end_index]
        # questions = ['Question: ' + q for q in data["question"]]
        questions = [get_question(q) for q in data["prompt"]]
        # right_answers = data["target"]
        right_answers = [get_reward(r) for r in data["reward_model"]]

        preds, history = get_trigpt_sample(args, questions, model, tokenizer)

        for index in range(len(questions)):
            print(preds[index])
            print()
            print(f'Ground truth answers: {right_answers[index]}')
            for step, h in enumerate(history):
                print(f'Step {step}: {h}')
            print(f'***************************************************')

        for pred, right_answer in zip(preds, right_answers):
            if get_acc(pred, right_answer):
                acc += 1
        num += len(questions)

    print(f'Acc: {acc/num}')

