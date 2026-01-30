"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import re

import shortuuid
import torch
from tqdm import tqdm
from typing import Optional

from fastchat.model.model_adapter import get_conversation_template
from fastchat.utils import str_to_torch_dtype
import torch.nn.functional as F

import sys
from pathlib import Path
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from transformers import AutoTokenizer
from lit_gpt.model_cache import GPTCache, Config
from lit_gpt.diffmodel import TransEncoder
from safetensors.torch import load_file


def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


@ torch.no_grad()
def ar_sample_kvcache(gpt, tokenizer, prompt, temperature=1., context_length=2048, device='cuda'):
    gpt.eval()
    gpt.reset_cache()

    prev_pos = 0
    for cur_pos in range(prompt.shape[1], context_length):
        input_pos = torch.arange(cur_pos, dtype=torch.long, device=device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = gpt(prompt[:, prev_pos:cur_pos], input_pos=input_pos)[:, -1]

        logits_with_noise = add_gumbel_noise(logits, temperature)
        next_token = torch.argmax(logits_with_noise, dim=-1, keepdim=True)

        prompt = torch.cat([prompt, next_token], dim=-1)
        prev_pos = cur_pos
        if next_token[0] == torch.tensor([tokenizer.eos_token_id], device=device):
            break
    return prompt


@ torch.no_grad()
def diff_sample(model, tokenizer, prompt=None, batch_size=1, alg='origin', steps=512, temperature=1., cfg_scale=2.,
                context_length=2048, eps=1e-5, dim=32000, device='cuda'):
    batch_size = batch_size if prompt is None else prompt.shape[0]
    x = torch.full((batch_size, context_length), dim, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    for i in range(steps):
        mask_index = (x == dim)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[:, :prompt.shape[1]] = dim
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits, un_logits = logits[mask_index], un_logits[mask_index]
            else:
                logits = model(x)[mask_index]

        if cfg_scale > 0.:
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == 'origin':
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + dim
            transfer_index_t_s = torch.rand(*x0.shape, device='cuda') < p_transfer
            logits_with_noise = add_gumbel_noise(logits[transfer_index_t_s], temperature=temperature)
            x0[transfer_index_t_s] = torch.argmax(logits_with_noise, dim=-1)
            x[mask_index] = x0.clone()
        elif alg == 'greddy':
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            logits = logits.to(torch.float64)
            p = F.softmax(logits, dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            num_mask_token = mask_index.sum()
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
            if number_transfer_tokens > 0:
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + dim
                x0_[transfer_index] = x0[transfer_index].clone()
                x[mask_index] = x0_
        else:
            raise NotImplementedError(alg)

    return x

@ torch.no_grad()
def masked_ar_sample(model_l2r, model_r2l, tokenizer, prompt=None, batch_size=1, alg='origin', steps=512, temperature=1., response_length=16, eps=1e-5, mask_token_id = 0, device='cuda'):
    # print("Prompt shape", prompt.shape)
    batch_size = batch_size if prompt is None else prompt.shape[0]
    prompt_length = prompt.shape[1]
    context_length = prompt_length + response_length
    x_l2r = torch.full((batch_size, context_length), mask_token_id, dtype=torch.long).to(device)
    x_l2r[:, :prompt_length] = prompt.clone()
    x_r2l = x_l2r.clone()

    # timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    # We simply decode 1 token at each step
    order = []
    gen_model = []
    gt_order = [7,8,6,9,5,10,4,11,3,12,2,13,1,14,0,15]  # for response_length=16
    for i in range(response_length):
        resp_ids = x_l2r[:, prompt_length:]
        resp_mask_index = (resp_ids == mask_token_id) # (b, resp_len)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            resp_logits_l2r = model_l2r(x_l2r)[:, prompt_length-1:-1] # (b, resp_len, vocab_size)
            resp_logits_r2l = model_r2l(x_r2l)[:, prompt_length-1:-1]
            resp_logits_r2l = resp_logits_r2l.flip(dims=[1]) # reverse the response logits for r2l

        # t = timesteps[i]
        # s = timesteps[i + 1]
        if alg == 'low_confidence':

            def sample_from_logits(logits):
                if temperature < 1e-4:
                    logits_with_noise = logits
                else:
                    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                logits = logits.to(torch.float64)
                p = F.log_softmax(logits, dim=-1)
                confidence = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(dim=-1)
                return x0, confidence
            x0_l2r, confidence_l2r = sample_from_logits(resp_logits_l2r)
            x0_r2l, confidence_r2l = sample_from_logits(resp_logits_r2l)
            number_transfer_tokens = 1
            if number_transfer_tokens > 0:
                # Get confidence by taking max from both directions
                confidence = torch.max(confidence_l2r, confidence_r2l)
                # Get x0 according to which direction has higher confidence
                x0 = torch.where(confidence_l2r >= confidence_r2l, x0_l2r, x0_r2l)
                model_id = torch.argmax(torch.stack([confidence_l2r, confidence_r2l], dim=0), dim=0) # (b, resp_len)

                # confidence = confidence_r2l # (b, resp_len)
                # x0 = x0_r2l # (b, resp_len)

                # confidence = confidence_r2l if i % 2 == 0 else confidence_l2r
                # x0 = x0_r2l if i % 2 == 0 else x0_l2r

                # modify confidence of unmasked tokens to -inf so they won't be selected
                confidence = torch.where(resp_mask_index, confidence, float('-inf'))

                _, transfer_index = torch.topk(confidence, number_transfer_tokens) # (b, number_transfer_tokens)
                # transfer_index = torch.tensor([[gt_order[i]] for _ in range(batch_size)], device='cuda')  # (b, 1)

                order.append(transfer_index[0,0].item())
                model_used = model_id[0, transfer_index[0,0]].item()
                gen_model.append('L2R' if model_used == 0 else 'R2L')
                transfer_ids = torch.gather(x0, dim=-1, index=transfer_index) # (b, number_transfer_tokens)
                row_index = torch.arange(batch_size).unsqueeze(-1) # (b, 1)
                x_l2r[row_index, transfer_index + prompt_length] = transfer_ids
                # print("Step", i, "Transfer index L2R:", x_l2r[0,-16:].to('cpu').tolist())
                transfer_index_r2l = response_length - 1 - transfer_index
                x_r2l[row_index, transfer_index_r2l + prompt_length] = transfer_ids
        else:
            raise NotImplementedError(alg)
    print(f"Order: {order}")
    print(f"Model used: {gen_model}")
    return x_l2r

@ torch.no_grad()
def trigpt_sample(model, tokenizer, prompt=None, batch_size=1, alg='origin', steps=512, temperature=1., cfg_scale=2.,
                response_length=16, eps=1e-5, dim=32000, device='cuda'):
    batch_size = batch_size if prompt is None else prompt.shape[0]
    prompt_length = prompt.shape[1]
    context_length = prompt_length + response_length
    x = torch.full((batch_size, context_length), dim, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    history = []
    conf_record = torch.full((batch_size, response_length), -1e10, device='cuda', dtype=torch.float64)
    for i in range(steps):
        mask_index = (x == dim)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x)
            masked_logits = logits[mask_index]

        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == 'origin':
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + dim
            transfer_index_t_s = torch.rand(*x0.shape, device='cuda') < p_transfer
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x_decode = torch.argmax(logits_with_noise, dim=-1) # full length
            history.append(x_decode[:, prompt_length:].cpu())
            x0[transfer_index_t_s] = x_decode[mask_index][transfer_index_t_s].clone()
            x[mask_index] = x0.clone()
        elif alg == 'greddy':
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x_decode = torch.argmax(logits_with_noise, dim=-1) # full length
            # history.append(x_decode[:, prompt_length:].cpu())

            x0 = x_decode[mask_index]

            masked_logits = masked_logits.to(torch.float64)
            p = F.softmax(masked_logits, dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            num_mask_token = mask_index.sum()
            # number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
            number_transfer_tokens = 1
            if number_transfer_tokens > 0:
                _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                x0_ = torch.zeros_like(x0, device=device, dtype=torch.long) + dim
                x0_[transfer_index] = x0[transfer_index].clone()
                x[mask_index] = x0_
            history.append(x[:, prompt_length:].cpu())
        elif alg == 'dyn_greedy':
            resp_logits = logits[:, prompt_length:, :] # only response part
            logits_with_noise = add_gumbel_noise(resp_logits, temperature=temperature)
            x_decode = torch.argmax(logits_with_noise, dim=-1) # (b,resp_len)
            # history.append(x_decode.cpu())

            resp_logits = resp_logits.to(torch.float64)
            p = F.softmax(resp_logits, dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x_decode, -1)).squeeze(dim=-1)
            # num_mask_token = mask_index.sum()
            # number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
            x0 = x[:, prompt_length:]
            
            # modify tokens with higher confidence
            transfer_indices = torch.logical_and(confidence > conf_record, x_decode != x0)
            if transfer_indices.sum() == 0:
                print(f"Step {i}: No more tokens to transfer, stopping early.")
                break
            # # choose the most confident token among transfer_indices to transfer
            # confidence_masked = torch.where(transfer_indices, confidence, float('-inf'))
            # transfer_index = torch.argmax(confidence_masked, dim=-1)  # (b,)
            # row_index = torch.arange(batch_size) # (b,)
            # conf_record[row_index, transfer_index] = confidence[row_index, transfer_index]
            # # update x
            # x[row_index, transfer_index + prompt_length] = x_decode[row_index, transfer_index]

            # transfer all tokens that meet the criteria
            conf_record = torch.where(transfer_indices, confidence, conf_record)
            x0_ = torch.where(transfer_indices, x_decode, x0)
            x[:, prompt_length:] = x0_.clone()

            history.append(x[:, prompt_length:].cpu())
        else:
            raise NotImplementedError(alg)

    return x, history


def generate(model, tokenizer, input_ids, max_new_tokens=16):
    # append input_ids with mask token id of length max_new_tokens
    prompt_length = input_ids.shape[1]
    mask_token_id = 0
    mask_tokens = torch.full((input_ids.size(0), max_new_tokens), mask_token_id, dtype=input_ids.dtype, device=input_ids.device)
    input_ids = torch.cat([input_ids, mask_tokens], dim=1)
    # 2. Inference Loop
    decode_orders = []
    for i in range(max_new_tokens):
        gen_ids = input_ids[:, prompt_length:]
        mask = (gen_ids == mask_token_id) # only decode mask tokens
        with torch.no_grad():
            # Get logits from the model
            outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        logprobs = torch.log_softmax(logits[:, -max_new_tokens-1:-1, :], dim=-1)
        
        # Focus on the response tokens
        resp_token_logits = logits[:, -max_new_tokens-1:-1, :]
        
        # Simple Greedy Decoding (argmax)
        # You can replace this with top-k or top-p sampling if needed
        resp_token_id = torch.argmax(resp_token_logits, dim=-1) # (b, max_new_tokens)
        token_logprobs = torch.gather(logprobs, -1, resp_token_id.unsqueeze(-1)).squeeze(-1)
        token_logprobs = torch.where(mask, token_logprobs, float('-inf'))
        # print("Probability", token_logprobs)
        decode_pos = torch.argmax(token_logprobs, dim=-1) # (b,)

        for j in range(decode_pos.shape[0]):
            p = decode_pos[j]
            decode_orders.append(p.item())
            # use a predefined order
            input_ids[j, prompt_length + p] = resp_token_id[j, p]
    print("Decode Order:", decode_orders)
    return input_ids


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    steps,
    model_type,
    cfg_scale,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                steps=steps,
                model_type=model_type,
                cfg_scale=cfg_scale,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    steps,
    model_type,
    cfg_scale,
):
    model_name = f"Diff_LLaMA_{model_id}M"
    config = Config.from_name(model_name)
    if model_type == 'arm':
        model = GPTCache(config).to('cuda')
    elif model_type == 'mdm':
        model = TransEncoder(config).to('cuda')
    else:
        raise ValueError(f"{model_type}")
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                              padding_side="right", use_fast=True)

    ckpt_dic = load_file(model_path)
    model.load_state_dict(ckpt_dic)

    for question in tqdm(questions):
        temperature = 1.

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("models/vicuna-7b-v1.5")
            turns = []
            for j in range(len(question["turns"])):
                if j > 0:
                    output = ''
                else:
                    qs = question["turns"][j]
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to('cuda')

                    if temperature < 1e-4:
                        do_sample = False
                    else:
                        do_sample = True

                    # some models may error out when generating long outputs
                    try:
                        if model_type == 'arm':
                            output_ids = ar_sample_kvcache(model,
                                                           tokenizer,
                                                           input_ids,
                                                           temperature=temperature,
                                                           context_length=max_new_token,
                                                           device='cuda')
                        else:
                            output_ids = diff_sample(model,
                                                     tokenizer,
                                                     input_ids,
                                                     steps=steps,
                                                     temperature=temperature,
                                                     cfg_scale=cfg_scale,
                                                     context_length=max_new_token,
                                                     device='cuda')

                        output_ids = output_ids[0][len(input_ids[0]) :]

                        # be consistent with the template's stop_token_ids
                        if conv.stop_token_ids:
                            stop_token_ids_index = [
                                i
                                for i, id in enumerate(output_ids)
                                if id in conv.stop_token_ids
                            ]
                            if len(stop_token_ids_index) > 0:
                                output_ids = output_ids[: stop_token_ids_index[0]]

                        output = tokenizer.decode(
                            output_ids,
                            spaces_between_special_tokens=False,
                        )
                        if conv.stop_str and isinstance(conv.stop_str, list):
                            stop_str_indices = sorted(
                                [
                                    output.find(stop_str)
                                    for stop_str in conv.stop_str
                                    if output.find(stop_str) > 0
                                ]
                            )
                            if len(stop_str_indices) > 0:
                                output = output[: stop_str_indices[0]]
                        elif conv.stop_str and output.find(conv.stop_str) > 0:
                            output = output[: output.find(conv.stop_str)]

                        for special_token in tokenizer.special_tokens_map.values():
                            if isinstance(special_token, list):
                                for special_tok in special_token:
                                    output = output.replace(special_tok, "")
                            else:
                                output = output.replace(special_token, "")
                        output = output.strip()

                        if conv.name == "xgen" and output.startswith("Assistant:"):
                            output = output.replace("Assistant:", "", 1).strip()
                    except RuntimeError as e:
                        print("ERROR question ID: ", question["question_id"])
                        output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="arm",
        help="arm or mdm",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="sampling steps.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=0.,
        help="classfier-free guidance.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    # We measure the inference time following:
    # https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
    # GPU warmup
    model = torch.nn.Sequential(
        torch.nn.Linear(10000, 10000),
        torch.nn.ReLU(),
        torch.nn.Linear(10000, 10000),
        torch.nn.ReLU(),
        torch.nn.Linear(10000, 1)
    ).to('cuda')
    for _ in range(10):
        x = torch.randn(1024, 10000, device='cuda')
        y = model(x)
    del model

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        steps=args.steps,
        model_type=args.model_type,
        cfg_scale=args.cfg_scale,
    )

    ender.record()
    torch.cuda.synchronize()
    consume_times = starter.elapsed_time(ender)
    message = f'{answer_file}, inference time: {consume_times / 1000} s'
    print(message)
    reorg_answer_file(answer_file)
