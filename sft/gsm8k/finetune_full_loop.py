import glob
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional, Union
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from functools import partial

wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.diffmodel import TransEncoder, Block, Config
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import get_default_supported_precision, num_parameters, step_csv_logger
from pytorch_lightning.loggers import WandbLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from sft.gsm8k_data import preprocess_gsm8k
from transformers import AutoTokenizer
import argparse
from safetensors.torch import load_file
from datasets import load_dataset

from eval.gen_model_answer import diff_sample
from evaluate_diff import set_seed
from evaluate_gsm8k import get_diff_sample, get_acc


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, help='model parameters')
    parse.add_argument('--bs', type=int, default=256, help='batch size')
    parse.add_argument('--epoch', type=int, default=40, help='training epoch')
    parse.add_argument('--pretrain_path', type=str, help='pretrain ckpt path')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of devices')
    parse.add_argument('--n_gpu', type=int, default=4, help='number of gpu')
    # eval args
    parse.add_argument('--eval_step_interval', type=int, default=5000, help='evaluate every N optimizer steps')
    parse.add_argument('--eval_steps', type=int, default=256, help='diffusion steps for evaluation sampling')
    parse.add_argument('--eval_cfg1', type=float, default=0.1, help='cfg scale for first pass')
    parse.add_argument('--eval_cfg2', type=float, default=0.1, help='cfg scale for second pass')
    parse.add_argument('--eval_temperature', type=float, default=0.1, help='sampling temperature for evaluation')
    parse.add_argument('--eval_length', type=int, default=256, help='context length for evaluation sampling')
    parse.add_argument('--eval_batch_size', type=int, default=64, help='batch size for evaluation')
    args = parse.parse_args()
    return args


args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'
out_dir = Path('workdir')

# Hyperparameters
num_of_devices = args.n_gpu
global_batch_size = int(args.bs / args.nodes_num)
learning_rate = 2e-4
micro_batch_size = 16
max_step = int(769240 * args.epoch / args.bs)
warmup_steps = int(max_step * 0.01)
log_step_interval = 10
save_step_interval = 5000

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


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, p_mask


def extract_number(filename):
    match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
    return int(match.group(1)) if match else 0


def evaluate(fabric, model, tokenizer, eval_args):
    """Run GSM8K test set evaluation distributed across all ranks.

    Each rank processes a contiguous slice of the test set. All ranks call
    model() the same number of times (required by FSDP all-gather). Results
    are summed via all_reduce before computing accuracy.
    """
    model.eval()
    set_seed(1234)

    dataset = load_dataset('json', data_files=str(wd / 'data/gsm8k/test.jsonl'))
    data = dataset['train']
    length = len(data)
    rank = fabric.global_rank
    world_size = fabric.world_size
    batch_size = eval_args.eval_batch_size

    # Assign each rank a contiguous slice. per_rank is the same for every rank
    # so n_batches is identical across ranks — required for FSDP all-gather sync.
    per_rank = math.ceil(length / world_size)
    n_batches = math.ceil(per_rank / batch_size)
    start = rank * per_rank

    acc = 0
    num = 0
    with torch.no_grad():
        for i in range(n_batches):
            real_start = min(start + i * batch_size, length)
            real_end = min(start + (i + 1) * batch_size, min(start + per_rank, length))

            if real_start >= real_end:
                # Padding batch: this rank has no more real data but must still
                # call model() to participate in FSDP all-gather on other ranks.
                dummy_q = ['Question: ' + data[0]["question"]]
                get_diff_sample(eval_args, dummy_q, model, tokenizer)
                continue

            batch = data[list(range(real_start, real_end))]
            questions = ['Question: ' + q for q in batch["question"]]
            right_answers = batch["target"]

            preds = get_diff_sample(eval_args, questions, model, tokenizer)

            for pred, right_answer in zip(preds, right_answers):
                if get_acc(pred, right_answer):
                    acc += 1
            num += len(questions)

    # Sum acc and num across all ranks, then compute accuracy on the aggregated counts.
    stats = torch.tensor([acc, num], dtype=torch.float64, device=fabric.device)
    stats = fabric.all_reduce(stats, reduce_op="sum")
    total_acc, total_num = int(stats[0].item()), int(stats[1].item())

    accuracy = total_acc / total_num
    fabric.print(f"[Eval] GSM8K accuracy: {accuracy:.4f} ({total_acc}/{total_num})")
    model.train()
    return accuracy


def setup(
    devices: int = 8,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'mdm-gsm8k-{args.model}M'
    out_dir = Path('workdir/finetune') / hp_name
    pretrain_path = args.pretrain_path
    wandb_logger = WandbLogger(name=hp_name, save_dir=out_dir, project='smdm-gsm8k')

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
    fabric.launch(main, pretrain_path, resume)


def main(fabric, pretrain_path, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                              padding_side="right", use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token_id = 32000

    train_set = preprocess_gsm8k(tokenizer, max_length=256)

    fabric.seed_everything(3407)
    train_dataloader = DataLoader(train_set, batch_size=micro_batch_size, shuffle=True, drop_last=True,
                                  num_workers=8, pin_memory=True, persistent_workers=True)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))
        ckpt_dic = load_file(pretrain_path)
        model.load_state_dict(ckpt_dic)
        fabric.print(f"Loading model from {pretrain_path}")

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        try:
            resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
        except Exception:
            resume = False
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, monitor, resume, tokenizer)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, monitor, resume, tokenizer):
    model = state["model"]
    optimizer = state["optimizer"]

    with torch.device("meta"):
        meta_model = TransEncoder(model.config)
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    initial_iter = state["iter_num"]
    curr_iter = 0
    resume_t0 = time.perf_counter()
    fabric.print(f"Resuming from iter {initial_iter}, fast-forwarding dataloader...")

    # Build eval_args namespace for get_diff_sample
    import types
    eval_args = types.SimpleNamespace(
        steps=args.eval_steps,
        cfg1=args.eval_cfg1,
        cfg2=args.eval_cfg2,
        temperature=args.eval_temperature,
        length=args.eval_length,
        eval_batch_size=args.eval_batch_size,
    )

    def get_train_dataloader(dataset_loader):
        while True:
            for data in dataset_loader:
                yield data

    train_dataloader_ = get_train_dataloader(train_dataloader)
    loss_func = CrossEntropyLoss(reduction='none')

    for train_data in train_dataloader_:
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume fast-forward finished in {:.2f}s (total from start: {:.2f}s)".format(
                    time.perf_counter() - resume_t0, time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break

        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()
        input_ids = train_data['data']
        prompt_length = train_data['input_length']
        max_length = 256
        input_ids = input_ids[:, :max_length]

        total_dim = 32000
        noisy_input, p_mask = forward_process(input_ids, total_dim=total_dim)
        temp_tensor = torch.arange(noisy_input.size(1), device=noisy_input.device).expand(noisy_input.size(0), noisy_input.size(1))
        prompt_index = (temp_tensor < prompt_length.unsqueeze(1))
        noisy_input[prompt_index] = input_ids[prompt_index].clone()
        mask_indices = (noisy_input == total_dim)

        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(noisy_input)
            loss = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * max_length - prompt_length.sum())
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1
        elif fabric.device.type == "xla":
            xm.mark_step()

        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
            f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours."
            f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days."
        )

        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item()
        )

        if not is_accumulating:
            step = state["step_count"]

            # Periodic evaluation
            if step % args.eval_step_interval == 0 or step == max_step:
                fabric.print(f"Running GSM8K evaluation at step {step}...")
                accuracy = evaluate(fabric, model, tokenizer, eval_args)
                fabric.log_dict({"eval/accuracy": accuracy}, step=step)

            # Checkpoint saving
            if step % save_step_interval == 0 or step == max_step:
                checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
                fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                fabric.save(checkpoint_path, state)


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
