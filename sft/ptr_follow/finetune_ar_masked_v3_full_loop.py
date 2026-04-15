# Masked autoregressive fine-tuning of L2R and R2L models simultaneously,
# with periodic validation using the iterative-unmask dual-AR strategy.
#
# v2 differences from finetune_ar_masked_full_loop.py:
#   - L2R model: same masked forward process (randomly mask a prefix of the response)
#   - R2L model: standard autoregressive training (no masking; loss on full response)

import glob
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
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
from pytorch_lightning.loggers import WandbLogger
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
    parse.add_argument('--epoch', type=int, default=100, help='training epoch')
    parse.add_argument('--l2r_pretrain_path', type=str, default='', help='l2r pretrain ckpt path')
    parse.add_argument('--r2l_pretrain_path', type=str, default='', help='r2l pretrain ckpt path')
    parse.add_argument('--task', type=str, default='ptr_follow')
    parse.add_argument('--order', type=str, default='reverse', help='reverse, middle')
    parse.add_argument('--n_gpu', type=int, default=4, help='number of gpu')
    parse.add_argument('--save_freq', type=int, default=10)
    parse.add_argument('--postfix', type=str, default='')
    parse.add_argument('--num_val', type=int, default=100, help='number of validation samples')
    parse.add_argument('--val_freq', type=int, default=10, help='validation frequency in steps')
    args = parse.parse_args()
    return args

args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'
out_dir = Path('workdir')

# Hyperparameters
num_of_devices = args.n_gpu
global_batch_size = args.bs
learning_rate = 2e-4
if args.model <= 50:
    micro_batch_size = 128
else:
    micro_batch_size = 8
max_step = int(10000 * args.epoch / global_batch_size)
warmup_steps = 200
log_step_interval = 1
save_step_interval = args.save_freq

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


# ---------------------------------------------------------------------------
# Masked forward process for L2R (from sft/finetune_ar_masked.py)
# ---------------------------------------------------------------------------

def forward_process_ar(batch, mask_token_id: int, prompt_length: torch.Tensor, response_length: torch.Tensor):
    """Randomly mask a prefix of each response, returning the noisy batch and the
    mask boundary (mask_id_high). Loss is computed only on the unmasked suffix."""
    resp_length = response_length[0].item()
    assert torch.all(response_length == resp_length), "All response lengths in the batch must be the same"
    b, l = batch.shape
    n_mask = torch.randint(low=0, high=resp_length, size=(b,), device=batch.device)

    mask_id_high = (prompt_length + n_mask).unsqueeze(1).repeat(1, l)
    mask_id_low = prompt_length.unsqueeze(1).repeat(1, l)

    pos_indices = torch.arange(l, device=batch.device).unsqueeze(0).repeat(b, 1)
    mask_indices = (pos_indices < mask_id_high) & (pos_indices >= mask_id_low)
    noisy_batch = torch.where(mask_indices, mask_token_id, batch)
    return noisy_batch, mask_id_high


# ---------------------------------------------------------------------------
# Dual-AR iterative-unmask sampler (from eval_double_ar_iterative_unmask.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def double_ar_sample_anchor_then_expand(model_l2r, model_r2l, prompt, response_length=16, device='cuda'):
    """Anchor-then-expand sampler using two AR models (L2R and R2L).

    Phase 1: Run both models on the fully-masked response, pick the single
    position with the highest combined confidence as the anchor, reveal it.

    Phase 2 (L2R): Fill positions to the right of the anchor left-to-right
    using the L2R model, each step conditioning on previously revealed tokens.

    Phase 3 (R2L): Fill positions to the left of the anchor right-to-left
    using the R2L model, each step conditioning on previously revealed tokens.
    """
    MASK_TOKEN_ID = 0

    prompt = prompt.to(device)
    batch_size = prompt.shape[0]
    batch_idx = torch.arange(batch_size, device=device)

    response = torch.full((batch_size, response_length), MASK_TOKEN_ID, dtype=torch.long, device=device)

    # Phase 1: find anchor
    x = torch.cat([prompt, response], dim=-1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        l2r_logits = model_l2r(x)[:, -(response_length + 1):-1]

    flipped_response = response.flip(dims=[1])
    flipped_x = torch.cat([prompt, flipped_response], dim=-1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        r2l_logits = model_r2l(flipped_x)[:, -(response_length + 1):-1].flip(dims=[1])

    l2r_probs = torch.softmax(l2r_logits.float(), dim=-1)
    r2l_probs = torch.softmax(r2l_logits.float(), dim=-1)
    confidence_l2r, pred_l2r = torch.max(l2r_probs, dim=-1)
    confidence_r2l, pred_r2l = torch.max(r2l_probs, dim=-1)

    # Anchor is determined solely by L2R max confidence
    anchor_pos = torch.argmax(confidence_l2r, dim=-1)  # (B,)
    response[batch_idx, anchor_pos] = pred_l2r[batch_idx, anchor_pos]

    # Phase 2: L2R expansion — fill positions right of anchor
    for j in range(response_length):
        needs_fill = j > anchor_pos  # (B,) bool
        if not needs_fill.any():
            continue
        x = torch.cat([prompt, response], dim=-1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model_l2r(x)[:, -(response_length + 1):-1]
        pred = logits[:, j].float().argmax(dim=-1)
        response[:, j] = torch.where(needs_fill, pred, response[:, j])

    # Phase 3: R2L expansion — fill positions left of anchor
    for j in range(response_length - 1, -1, -1):
        needs_fill = j < anchor_pos  # (B,) bool
        if not needs_fill.any():
            continue
        flipped_response = response.flip(dims=[1])
        flipped_x = torch.cat([prompt, flipped_response], dim=-1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            r2l_logits = model_r2l(flipped_x)[:, -(response_length + 1):-1].flip(dims=[1])
        pred = r2l_logits[:, j].float().argmax(dim=-1)
        response[:, j] = torch.where(needs_fill, pred, response[:, j])

    return torch.cat([prompt, response], dim=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_number(filename):
    match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
    return int(match.group(1)) if match else 0


def load_ckpt_into_model(model, path, fabric):
    if path == '':
        fabric.print("No pretrain path provided — starting from scratch.")
        return
    if path.endswith('.safetensors'):
        ckpt_dic = load_file(path)
    else:
        ckpt_dic = torch.load(path, map_location='cpu')["model"]
    model.load_state_dict(ckpt_dic)
    fabric.print(f"Loaded checkpoint from {path}")


# ---------------------------------------------------------------------------
# Setup / main
# ---------------------------------------------------------------------------

def setup(
    devices: int = 8,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'arm-{args.model}M-masked-v3-{args.order}-double'
    if args.postfix != '':
        hp_name += f'-{args.postfix}'
    out_dir = Path('workdir/finetune') / hp_name
    wandb_logger = WandbLogger(name=hp_name, save_dir=out_dir, project=f'18786_{args.task}_{args.order}')

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

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
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
        train_set_l2r, val_set = preprocess_ptr_follow_ar_split(
            tokenizer, r2l=False, order=args.order, num_val=args.num_val)
        train_set_r2l, _ = preprocess_ptr_follow_ar_split(
            tokenizer, r2l=True, order=args.order, num_val=args.num_val)
    else:
        raise NotImplementedError(f"Task {args.task} not implemented")

    fabric.seed_everything(3407)

    train_dl_l2r = DataLoader(train_set_l2r, batch_size=micro_batch_size, shuffle=True, drop_last=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    train_dl_r2l = DataLoader(train_set_r2l, batch_size=micro_batch_size, shuffle=True, drop_last=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)
    train_dl_l2r = fabric.setup_dataloaders(train_dl_l2r)
    train_dl_r2l = fabric.setup_dataloaders(train_dl_r2l)

    # val_dataloader is intentionally NOT wrapped — validation runs on rank-0 only
    val_dataloader = DataLoader(val_set, batch_size=micro_batch_size, shuffle=False, drop_last=False,
                                num_workers=8, pin_memory=True, persistent_workers=True)

    fabric.print(f"Loading models with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        l2r_model = GPT(config)
        l2r_model.apply(partial(l2r_model._init_weights, n_layer=config.n_layer))
        load_ckpt_into_model(l2r_model, args.l2r_pretrain_path, fabric)

        r2l_model = GPT(config)
        r2l_model.apply(partial(r2l_model._init_weights, n_layer=config.n_layer))
        load_ckpt_into_model(r2l_model, args.r2l_pretrain_path, fabric)

    fabric.print(f"Time to instantiate models: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters per model: {num_parameters(l2r_model):,}")

    l2r_model = fabric.setup(l2r_model)
    r2l_model = fabric.setup(r2l_model)

    l2r_optimizer = torch.optim.AdamW(
        l2r_model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False)
    r2l_optimizer = torch.optim.AdamW(
        r2l_model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False)
    l2r_optimizer = fabric.setup_optimizers(l2r_optimizer)
    r2l_optimizer = fabric.setup_optimizers(r2l_optimizer)

    state = {
        "l2r_model": l2r_model, "r2l_model": r2l_model,
        "l2r_optimizer": l2r_optimizer, "r2l_optimizer": r2l_optimizer,
        "hparams": hparams, "iter_num": 0, "step_count": 0,
    }

    if resume is True:
        try:
            resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
        except Exception:
            resume = False
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dl_l2r, train_dl_r2l, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(fabric, state, train_dl_l2r, train_dl_r2l, val_dataloader, monitor, resume):
    l2r_model = state["l2r_model"]
    r2l_model = state["r2l_model"]
    l2r_optimizer = state["l2r_optimizer"]
    r2l_optimizer = state["r2l_optimizer"]

    with torch.device("meta"):
        meta_model = GPT(l2r_model.config)
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs per model: {estimated_flops * fabric.world_size / 1e12:.2f}")
        del meta_model

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = 0

    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    loss_func = FusedCrossEntropyLoss()

    for l2r_data, r2l_data in zip(infinite(train_dl_l2r), infinite(train_dl_r2l)):
        # Resume: skip batches already processed
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break

        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in l2r_optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in r2l_optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0

        # ---- L2R forward / backward (masked training) ----------------------
        input_ids_l2r = l2r_data['data']
        prompt_len_l2r = l2r_data['input_length']
        length_l2r = l2r_data['length']
        input_ids_l2r = input_ids_l2r[:, :length_l2r.max().item()]

        noisy_l2r, mask_high_l2r = forward_process_ar(
            input_ids_l2r, mask_token_id=0,
            prompt_length=prompt_len_l2r,
            response_length=length_l2r - prompt_len_l2r,
        )

        with fabric.no_backward_sync(l2r_model, enabled=is_accumulating):
            logits_l2r = l2r_model(noisy_l2r)
            tmp = torch.arange(logits_l2r.size(1), device=input_ids_l2r.device).expand(logits_l2r.size(0), -1)
            logits_l2r = logits_l2r[(tmp >= mask_high_l2r - 1) & (tmp < (length_l2r - 1).unsqueeze(1))]
            targets_l2r = input_ids_l2r[(tmp >= mask_high_l2r) & (tmp < length_l2r.unsqueeze(1))]
            loss_l2r = loss_func(logits_l2r, targets_l2r)
            fabric.backward(loss_l2r / gradient_accumulation_steps)

        # ---- R2L forward / backward ----------------------------------------
        input_ids_r2l = r2l_data['data']
        prompt_len_r2l = r2l_data['input_length']
        length_r2l = r2l_data['length']
        input_ids_r2l = input_ids_r2l[:, :length_r2l.max().item()]

        noisy_r2l, mask_high_r2l = forward_process_ar(
            input_ids_r2l, mask_token_id=0,
            prompt_length=prompt_len_r2l,
            response_length=length_r2l - prompt_len_r2l,
        )

        with fabric.no_backward_sync(r2l_model, enabled=is_accumulating):
            logits_r2l = r2l_model(noisy_r2l)
            tmp = torch.arange(logits_r2l.size(1), device=input_ids_r2l.device).expand(logits_r2l.size(0), -1)
            logits_r2l = logits_r2l[(tmp >= mask_high_r2l - 1) & (tmp < (length_r2l - 1).unsqueeze(1))]
            targets_r2l = input_ids_r2l[(tmp >= mask_high_r2l) & (tmp < length_r2l.unsqueeze(1))]
            loss_r2l = loss_func(logits_r2l, targets_r2l)
            fabric.backward(loss_r2l / gradient_accumulation_steps)

        # ---- Optimizer step ------------------------------------------------
        if not is_accumulating:
            fabric.clip_gradients(l2r_model, l2r_optimizer, max_norm=grad_clip)
            l2r_optimizer.step()
            l2r_optimizer.zero_grad()

            fabric.clip_gradients(r2l_model, r2l_optimizer, max_norm=grad_clip)
            r2l_optimizer.step()
            r2l_optimizer.zero_grad()

            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        state["iter_num"] += 1
        total_lengths += input_ids_l2r.size(1)
        t1 = time.perf_counter()

        loss = (loss_l2r.item() + loss_r2l.item()) / 2
        fabric.print(
            f"iter {state['iter_num']} step {state['step_count']}: "
            f"loss {loss:.4f} (l2r {loss_l2r.item():.4f} r2l {loss_r2l.item():.4f}), "
            f"iter time: {(t1 - iter_t0) * 1000:.2f}ms"
            f"{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f}h"
        )

        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss,
        )

        if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step):
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        if not is_accumulating and (state["step_count"] % args.val_freq == 0):
            validate(fabric, l2r_model, r2l_model, val_dataloader, state["step_count"])


# ---------------------------------------------------------------------------
# Validation (iterative-unmask dual-AR)
# ---------------------------------------------------------------------------

def validate(fabric, l2r_model, r2l_model, val_dataloader, step):
    correct_list = []
    for data in val_dataloader:
        input_ids = data['data']

        resp_length = 16
        prompt_ids = input_ids[:, :-resp_length]
        target_ids = input_ids[:, -resp_length:]

        output_ids = double_ar_sample_anchor_then_expand(
            l2r_model, r2l_model, prompt_ids,
            response_length=resp_length, device=l2r_model.device,
        )[:, -resp_length:]
        output_ids = output_ids.cpu()
        correct = (output_ids == target_ids).all(dim=-1).float().tolist()
        correct_list.extend(correct)

    acc = np.mean(correct_list)
    fabric.print(f"Step {step}, validation accuracy: {acc * 100:.2f}%")
    fabric.log("val/val_acc", acc, step=step)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup(devices=num_of_devices)
