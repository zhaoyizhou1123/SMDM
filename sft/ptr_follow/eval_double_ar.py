# Split train set to train/val
# Add validation accuracy loop

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
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
# from sharegpt_data import preprocess_sharegpt
from transformers import AutoTokenizer
import random
import argparse
from safetensors.torch import load_file
from eval.gen_model_answer import double_ar_sample

import numpy as np


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, help='model parameters')
    parse.add_argument('--bs', type=int, default=256, help='batch size')
    parse.add_argument('--epoch', type=int, default=1, help='training epoch')
    # parse.add_argument('--pretrain_path', type=str, help='pretrain ckpt path')
    parse.add_argument('--task', type=str, default='ptr_follow')
    parse.add_argument('--order', type=str, default='reverse', help='reverse, middle')
    parse.add_argument('--n_gpu', type=int, default=4, help='number of gpu')
    # parse.add_argument('--save_freq', type=int, default=10)
    # parse.add_argument('--r2l', action='store_true', help='train r2l model')
    parse.add_argument('--l2r_path', type=str, required=True, help='l2r model ckpt path for r2l training')
    parse.add_argument('--r2l_path', type=str, required=True, help='r2l model ckpt path training')
    parse.add_argument('--postfix', type=str, default='')
    parse.add_argument('--num_val', type=int, default=100, help='number of validation samples')
    # parse.add_argument('--val_freq', type=int, default=10, help='validation frequency in steps')
    args = parse.parse_args()
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
max_step = int(10000 * args.epoch / global_batch_size) # 3 epochs
warmup_steps = 200
log_step_interval = 1
# save_step_interval = args.save_freq

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


def setup(
    devices: int = 8,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'double_arm-{args.model}M-masked-{args.order}'
    if args.postfix != '':
        hp_name += f'-{args.postfix}'
    out_dir = Path('workdir/finetune') / hp_name
    # pretrain_path = args.pretrain_path
    wandb_logger = WandbLogger(name=hp_name, save_dir=out_dir, project=f'18786_{args.task}_{args.order}')

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
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
    #fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, resume)


def main(fabric, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    # filename = 'data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
                                              padding_side="right", use_fast=True)

    # with open(filename) as f:
    #     data = json.load(f)
    if args.task == 'ptr_follow':
        from ptr_follow_data import preprocess_ptr_follow_ar_split
        train_set, val_set = preprocess_ptr_follow_ar_split(tokenizer, r2l=False, order=args.order, num_val=args.num_val)
    else:
        raise NotImplementedError(f"Task {args.task} not implemented")
    # train_set = preprocess_sharegpt(data, tokenizer)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)
    train_dataloader = DataLoader(train_set, batch_size=micro_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = DataLoader(val_set, batch_size=micro_batch_size, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True)
    # val_dataloader = fabric.setup_dataloaders(val_dataloader)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    def load_ckpt(path):
        model = GPT(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))

        if path.endswith('.safetensors'):
            ckpt_dic = load_file(path)
        else:
            ckpt_dic = torch.load(path, map_location='cpu')["model"]
        model.load_state_dict(ckpt_dic)
        fabric.print(f"Loading model from {path}")
        return model
    with fabric.init_module(empty_init=False):
        l2r_model = load_ckpt(args.l2r_path)
        r2l_model = load_ckpt(args.r2l_path)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(l2r_model):,}")

    l2r_model = fabric.setup(l2r_model)
    r2l_model = fabric.setup(r2l_model)

    validate(fabric, l2r_model, r2l_model, val_dataloader, step=0)

    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    # )
    # # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    # optimizer = fabric.setup_optimizers(optimizer)

    # state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    # if resume is True:
    #     try:
    #         resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
    #     except:
    #         resume = False
    # if resume :
    #     fabric.print(f"Resuming training from {resume}")
    #     fabric.load(resume, state)

    # train_time = time.perf_counter()
    # train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    # fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    # if fabric.device.type == "cuda":
    #     fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

def validate(fabric, l2r_model, r2l_model, val_dataloader, step):
    correct_list = []
    for data in val_dataloader:
        input_ids = data['data'] # [prompt + answer + padding]
        # prompt_length = data['input_length']  # prompt length (torch.Tensor)
        # length = data['length'] # [prompt + answer] length (torch.Tensor)

        # hard code response length, not good
        resp_length = 16
        prompt_ids = input_ids[:, :-resp_length]
        target_ids = input_ids[:, -resp_length:]
        
        output_ids = double_ar_sample(l2r_model, r2l_model, None, prompt_ids, temperature = 0., response_length=resp_length, device = l2r_model.device, n_iters=10)[:,-resp_length:]
        output_ids = output_ids.cpu()
        correct = (output_ids == target_ids).all(dim=-1).float().tolist()
        correct_list.extend(correct)
    acc = np.mean(correct_list)
    fabric.print(f"Step {step}, validation accuracy: {np.mean(correct_list)*100:.2f}%")
    fabric.log("val/val_acc", acc, step=step)


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup(devices=num_of_devices)
