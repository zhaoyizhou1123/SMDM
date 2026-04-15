# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SMDM (Scaling up Masked Diffusion Models on Text) is a research codebase implementing and evaluating masked diffusion models (MDMs) for language modeling. It establishes scaling laws for MDMs, trains models up to 1.1B parameters, and introduces unsupervised classifier-free guidance (CFG). The codebase is based on the [TinyLlama](https://github.com/jzhang38/TinyLlama) framework.

Paper: arXiv:2410.18514 | Pretrained models: [HuggingFace nieshen/SMDM](https://huggingface.co/nieshen/SMDM)

## Environment Setup

```sh
# Based on TinyLlama conda environment; then:
pip install lm-eval==0.4.4 numpy==1.25.0 bitsandbytes==0.43.1
pip install openai==0.28 fschat==0.2.34 anthropic
```

Key dependencies: PyTorch 2.4.1 + CUDA 12.1, Lightning 2.1.2, Flash-Attention 2.6.3, xformers 0.0.28, Transformers 4.31.0. See [CONDA.md](CONDA.md) for full installation commands.

## Commands

### Pretraining

```sh
# ARM (autoregressive), e.g. 1028M params, 100e18 FLOPs, 8 GPUs
lightning run model --node-rank=0 --accelerator=cuda --devices=8 --num-nodes=1 \
    pretrain/train_ar.py --model 1028 --flops 100.

# MDM (masked diffusion), e.g. 170M params, 10e18 FLOPs
lightning run model --node-rank=0 --accelerator=cuda --devices=8 --num-nodes=1 \
    pretrain/train_mdm.py --model 170 --flops 10.

# MDM with stochastic sequence length (1% of data uses random length)
lightning run model --node-rank=0 --accelerator=cuda --devices=8 --num-nodes=1 \
    pretrain/train_mdm_rl.py --model 170 --flops 60. --ssl_ratio 0.01

# Multi-node: set --node-rank=$RANK --main-address=$MASTER_ADDR --num-nodes=N --nodes_num=N
```

### Fine-tuning (SFT)

```sh
# Conditional generation — unsupervised CFG (--cfg 0.) or standard CFG (--cfg 0.1)
lightning run model --node-rank=0 --accelerator=cuda --devices=8 --num-nodes=1 \
    sft/finetune_mdm.py --model 1028 --pretrain_path models/mdm-1028M-1600e18.safetensors --cfg 0.

# Math reasoning (GSM8K)
lightning run model --node-rank=0 --accelerator=cuda --devices=8 --num-nodes=1 \
    sft/finetune_mdm_gsm8k.py --model 1028 --pretrain_path models/mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors

# Reverse curse
lightning run model --node-rank=0 --accelerator=cuda --devices=8 --num-nodes=1 \
    sft/finetune_mdm_reverse.py --model 1028 --pretrain_path models/mdm-1028M-1600e18.safetensors
```

### Evaluation

```sh
# Commonsense/reading comprehension benchmarks
python evaluate_ar.py --tasks hellaswag,openbookqa,arc_easy,boolq,piqa,social_iqa,race,lambada_standard \
    --model ar --model_args model_name=170,ckpt_path='models/ar-170M-100e18.safetensors'

# MDM benchmark evaluation (see eval_mdm.sh for full commands)
python evaluate_diff.py --model mdm --model_args model_name=170,ckpt_path='models/mdm-170M-100e18.safetensors'

# Math reasoning
python evaluate_gsm8k.py --ckpt_path "models/mdm-1028M-3300e18-rsl-gsm8k.safetensors"

# Reverse curse
python evaluate_reverse.py --qs_type ntd --model 1028 --ckpt-path "models/mdm-1028M-1600e18-reverse.safetensors"

# MT-Bench (conditional generation)
python eval/gen_model_answer.py --model-id 1028 --model-type 'mdm' \
    --model-path "models/mdm-1028M-1600e18-sharegpt.safetensors" \
    --steps 128 --cfg-scale 0.6 --answer-file "data/mt_bench/model_answer/mdm.jsonl"
export OPENAI_API_KEY=...
python eval/gen_judgment.py --parallel 10 --judge-model "gpt-4o-2024-05-13"
python eval/show_result.py --judge-model "gpt-4o-2024-05-13"

# Temporal degradation (FineWeb)
python evaluate_fineweb.py --type mdm --model 170 --ckpt-path 'models/mdm-170M-100e18.safetensors' \
    --fineweb "CC-MAIN-2024-18" --mc-samples 128
```

### Unit Tests

```sh
python test.py
python test_dat.py
python test_trigpt.py
```

## Architecture

### Core Model Implementations (`lit_gpt/`)

| File | Class | Purpose |
|------|-------|---------|
| `model.py` | `GPT` | Autoregressive LM with causal self-attention and KV-cache |
| `diffmodel.py` | `TransEncoder` | MDM — bidirectional (non-causal) transformer; core model |
| `datmodel.py` / `datmodel_v2.py` | `DualStreamGPT` | Dual-stream experimental architecture |
| `trimodel.py` | `TriGPT` | Three-stream experimental architecture |
| `config.py` | `Config` | Central model config; named configs like `Diff_LLaMA_170M` |
| `packed_dataset.py` | `PackedDataset` | Memory-mapped binary dataset for efficient distributed training |

### Training Pipeline

Scripts in `pretrain/` and `sft/` use **Lightning Fabric** for distributed training. The general pattern:
- Model/config loaded from `lit_gpt/config.py` via `--model` arg (e.g., `170` maps to `Diff_LLaMA_170M`)
- Checkpoints saved as `.safetensors` files
- SlimPajama data expected at `/dataset/slim_star_combined` (preprocessed binary chunks)
- Wandb integration for experiment tracking

You should mostly pay attention to `scripts/ptr_follow`. The scripts in it are most frequently used.

### Diffusion-Specific Concepts

- **Masking process**: tokens are randomly masked (replaced with `[MASK]` token) during training; the model predicts original tokens from partially-masked sequences
- **Stochastic sequence length (RSL/SSL)**: a fraction of training data uses random context lengths — improves short-context generation
- **Classifier-free guidance (CFG)**: during SFT, condition tokens are randomly dropped to train unconditional branch; at inference, guidance scale controls condition strength
  - Unsupervised CFG (`--cfg 0.`): uses unpaired data, drops condition entirely
  - Standard CFG (`--cfg 0.1`): drops condition 10% of time

### Evaluation Integration

`evaluate_ar.py` and `evaluate_diff.py` wrap models as custom `lm-eval` adapters (`ArEvalHarness`, `MDLMEvalHarness`). MDM evaluation uses Monte Carlo NLL estimation (configurable `--mc-samples`).

### Model Sizes

Named configs in `lit_gpt/config.py` follow `Diff_LLaMA_{N}M` pattern. The `--model` CLI arg accepts the number (e.g., `170`, `1028`). Supported sizes include: 6, 19, 34, 48, 66, 85, 113, 142, 170, 206, 231, 268, 302, 336, 472, 551, 629, 717, 831, 944, 1028, 1233.

## Data

- **Pretraining**: SlimPajama dataset preprocessed by TinyLlama scripts → `/dataset/slim_star_combined`
- **GSM8K**: augmented train data at `./data/gsm8k/train.txt`, test at `./data/gsm8k/test.jsonl`
- **ShareGPT**: JSON file at `./data/` for conditional generation SFT
- **Reverse curse**: `./data/reverse_experiments/` folder from lukasberglund/reversal_curse
- **FineWeb**: preprocessed via `scripts/prepare_fineweb.py` (requires separate `fineweb` conda env with `datatrove==0.2.0`)
