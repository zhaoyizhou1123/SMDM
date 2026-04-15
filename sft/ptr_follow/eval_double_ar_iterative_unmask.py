import glob
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from lit_gpt import FusedCrossEntropyLoss
from transformers import AutoTokenizer
import random
import argparse
from safetensors.torch import load_file

import numpy as np


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, help='model parameters')
    parse.add_argument('--bs', type=int, default=256, help='batch size')
    parse.add_argument('--epoch', type=int, default=1, help='training epoch')
    parse.add_argument('--task', type=str, default='ptr_follow')
    parse.add_argument('--order', type=str, default='reverse', help='reverse, middle')
    parse.add_argument('--n_gpu', type=int, default=4, help='number of gpu')
    parse.add_argument('--l2r_path', type=str, default='', help='l2r model ckpt path')
    parse.add_argument('--r2l_path', type=str, default='', help='r2l model ckpt path')
    parse.add_argument('--ckpt_path', type=str, default='', help='combined double-AR checkpoint (from finetune_ar_masked_full_loop.py)')
    parse.add_argument('--postfix', type=str, default='')
    parse.add_argument('--num_val', type=int, default=100, help='number of validation samples')
    parse.add_argument('--num_data', type=int, default=-1, help='max evaluation samples (-1 = all)')
    parse.add_argument('--debug_n', type=int, default=0, help='print intermediate sequences for first N samples')
    args = parse.parse_args()
    if not args.ckpt_path and not (args.l2r_path and args.r2l_path):
        parse.error("Provide either --ckpt_path or both --l2r_path and --r2l_path")
    return args

args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'  # config
out_dir = Path('workdir')

# Hyperparameters
num_of_devices = args.n_gpu
global_batch_size = args.bs
learning_rate = 2e-4
if args.model <= 50:
    micro_batch_size = 16
else:
    micro_batch_size = 8
max_step = int(10000 / global_batch_size)
warmup_steps = 200
log_step_interval = 1

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = learning_rate / 10

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)


def extract_number(filename):
    match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
    return int(match.group(1)) if match else 0


@torch.no_grad()
def double_ar_sample_iterative_unmask(model_l2r, model_r2l, prompt, response_length=16, device='cuda',
                                      debug=False, tokenizer=None):
    """Iterative unmasking sampler using two AR models (L2R and R2L).

    Starts with all response positions filled with a mask token (id=0).
    At each step, scores every still-masked position using both models and
    permanently fills the single most-confident position across both directions.
    Repeats until all response_length positions are filled.

    Args:
        model_l2r: left-to-right AR model
        model_r2l: right-to-left AR model (trained on reversed sequences)
        prompt: (B, prompt_len) token ids
        response_length: number of tokens to generate
        device: target device
        debug: if True, print intermediate unmasking steps for the first sample
        tokenizer: optional tokenizer for decoding tokens in debug output
    """
    MASK_TOKEN_ID = 0

    prompt = prompt.to(device)
    batch_size = prompt.shape[0]

    response = torch.full((batch_size, response_length), MASK_TOKEN_ID, dtype=torch.long, device=device)
    # True = position already filled, should be skipped
    filled = torch.zeros(batch_size, response_length, dtype=torch.bool, device=device)

    def _tok_str(tok_id):
        if tok_id == MASK_TOKEN_ID:
            return "[M]"
        if tokenizer is not None:
            return repr(tokenizer.decode([tok_id]))
        return str(tok_id)

    if debug:
        prompt_str = tokenizer.decode(prompt[0].cpu().tolist()) if tokenizer else str(prompt[0].cpu().tolist())
        print(f"\n[DEBUG] prompt: {prompt_str!r}")

    for step in range(response_length):
        x = torch.cat([prompt, response], dim=-1)

        # L2R pass: logits over response positions
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l2r_logits = model_l2r(x)[:, -(response_length+1):-1]  # (B, resp_len, vocab); shift: logit i predicts token i+1

        # R2L pass: flip response, run model, flip logits back
        flipped_response = response.flip(dims=[1])
        flipped_x = torch.cat([prompt, flipped_response], dim=-1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            r2l_logits_flipped = model_r2l(flipped_x)[:, -(response_length+1):-1]  # same shift correction
        r2l_logits = r2l_logits_flipped.flip(dims=[1])  # (B, resp_len, vocab)

        # Max-probability confidence and predicted token per position
        l2r_probs = torch.softmax(l2r_logits.float(), dim=-1)
        r2l_probs = torch.softmax(r2l_logits.float(), dim=-1)
        confidence_l2r, pred_l2r = torch.max(l2r_probs, dim=-1)  # (B, resp_len)
        confidence_r2l, pred_r2l = torch.max(r2l_probs, dim=-1)  # (B, resp_len)

        # For each position, use whichever direction is more confident
        use_l2r = confidence_l2r >= confidence_r2l                          # (B, resp_len)
        confidence_combined = torch.where(use_l2r, confidence_l2r, confidence_r2l)
        pred_combined = torch.where(use_l2r, pred_l2r, pred_r2l)

        # Exclude already-filled positions from selection
        confidence_combined = torch.where(filled, torch.full_like(confidence_combined, float('-inf')), confidence_combined)

        # Pick the single most-confident unfilled position per batch element
        best_pos = torch.argmax(confidence_combined, dim=-1)  # (B,)

        # Fill that position
        batch_idx = torch.arange(batch_size, device=device)
        response[batch_idx, best_pos] = pred_combined[batch_idx, best_pos]
        filled[batch_idx, best_pos] = True

        if debug:
            resp_toks = response[0].cpu().tolist()
            resp_display = " ".join(_tok_str(t) for t in resp_toks)
            pos = best_pos[0].item()
            tok = pred_combined[0, pos].item()
            src = "l2r" if use_l2r[0, pos].item() else "r2l"
            conf = confidence_combined[0, pos].item() if not filled[0, pos - 1 if pos > 0 else 0].item() else confidence_l2r[0, pos].item()
            conf_val = max(confidence_l2r[0, pos].item(), confidence_r2l[0, pos].item())
            print(f"  step {step+1:2d}: [{resp_display}]  <- pos {pos} = {_tok_str(tok)} (via {src}, conf={conf_val:.3f})")

    return torch.cat([prompt, response], dim=-1)


def setup(
    devices: int = 8,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'double_arm-{args.model}M-iterative-unmask-{args.order}'
    if args.postfix != '':
        hp_name += f'-{args.postfix}'
    out_dir = Path('workdir/finetune') / hp_name

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                auto_wrap_policy={Block},
                activation_checkpointing_policy=None,
                state_dict_type="full",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger])
    fabric.print(hparams)
    main(fabric, resume)


def main(fabric, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                              padding_side="right", use_fast=True)

    if args.task == 'ptr_follow':
        from ptr_follow_data import preprocess_ptr_follow_ar_split
        train_set, val_set = preprocess_ptr_follow_ar_split(tokenizer, r2l=False, order=args.order, num_val=args.num_val)
    else:
        raise NotImplementedError(f"Task {args.task} not implemented")

    fabric.seed_everything(3407)
    val_dataloader = DataLoader(val_set, batch_size=micro_batch_size, shuffle=False, drop_last=False,
                                num_workers=8, pin_memory=True, persistent_workers=True)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()

    def load_ckpt(path, state_key=None):
        model = GPT(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))
        if path.endswith('.safetensors'):
            ckpt_dic = load_file(path)
        else:
            raw = torch.load(path, map_location='cpu')
            if state_key is not None:
                # Combined checkpoint saved by finetune_ar_masked_full_loop.py via fabric.save
                ckpt_dic = raw[state_key]
            elif "model" in raw:
                ckpt_dic = raw["model"]
            else:
                ckpt_dic = raw
        model.load_state_dict(ckpt_dic)
        fabric.print(f"Loaded {state_key or 'model'} from {path}")
        return model

    with fabric.init_module(empty_init=False):
        if args.ckpt_path:
            l2r_model = load_ckpt(args.ckpt_path, state_key="l2r_model")
            r2l_model = load_ckpt(args.ckpt_path, state_key="r2l_model")
        else:
            l2r_model = load_ckpt(args.l2r_path)
            r2l_model = load_ckpt(args.r2l_path)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(l2r_model):,}")

    l2r_model = fabric.setup(l2r_model)
    r2l_model = fabric.setup(r2l_model)

    validate(fabric, l2r_model, r2l_model, val_dataloader, tokenizer=tokenizer,
             num_data=args.num_data, debug_n=args.debug_n, step=0)


def validate(fabric, l2r_model, r2l_model, val_dataloader, step,
             tokenizer=None, num_data=-1, debug_n=0):
    correct_list = []
    seen = 0
    for data in val_dataloader:
        input_ids = data['data']  # [prompt + answer + padding]

        # Respect num_data limit
        if num_data > 0:
            remaining = num_data - seen
            if remaining <= 0:
                break
            input_ids = input_ids[:remaining]

        resp_length = 16
        prompt_ids = input_ids[:, :-resp_length]
        target_ids = input_ids[:, -resp_length:]

        # Enable debug printing for the first debug_n samples (first batch only)
        debug_this_batch = debug_n > 0 and seen == 0
        output_ids = double_ar_sample_iterative_unmask(
            l2r_model, r2l_model, prompt_ids,
            response_length=resp_length, device=l2r_model.device,
            debug=debug_this_batch, tokenizer=tokenizer,
        )[:, -resp_length:]
        output_ids = output_ids.cpu()

        if debug_this_batch:
            n_print = min(debug_n, output_ids.size(0))
            for i in range(n_print):
                pred_str = tokenizer.decode(output_ids[i].tolist()) if tokenizer else str(output_ids[i].tolist())
                tgt_str = tokenizer.decode(target_ids[i].tolist()) if tokenizer else str(target_ids[i].tolist())
                match = (output_ids[i] == target_ids[i]).all().item()
                fabric.print(f"[DEBUG sample {i}] pred={pred_str!r}  target={tgt_str!r}  correct={match}")

        correct = (output_ids == target_ids).all(dim=-1).float().tolist()
        correct_list.extend(correct)
        seen += input_ids.size(0)

    acc = np.mean(correct_list)
    fabric.print(f"Step {step}, validation accuracy: {acc * 100:.2f}%  (n={seen})")
    fabric.log("val/val_acc", acc, step=step)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup(devices=num_of_devices)
