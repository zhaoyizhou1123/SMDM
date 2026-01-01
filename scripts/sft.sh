# export LIGHTNING_DISABLE_UPDATE_CHECK=1
# CUDA_VISIBLE_DEVICES=0,2,3,4 lightning run model \
#     --node-rank=0  \
#     --accelerator=cuda \
#     --devices=4 \
#     --num-nodes=1 \
# sft/finetune_mdm_gsm8k.py --model 1028 \
#     --pretrain_path models/mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors \
#     --n_gpu 4

# CUDA_VISIBLE_DEVICES=2 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29500 \
#     sft/finetune_mdm_gsm8k.py \
#     --model 1028 \
#     --pretrain_path models/mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors \
#     --n_gpu 1 \
#     --bs 256

CUDA_VISIBLE_DEVICES=2 torchrun \
    --nproc_per_node=1 \
    --node_rank=0 \
    --nnodes=1 \
    --master_port=29500 \
    sft/finetune_mdm_sudoku.py \
    --model 336 \
    --pretrain_path models/mdm-336M-100e18.safetensors \
    --n_gpu 1 \
    --bs 256