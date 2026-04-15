# CUDA_VISIBLE_DEVICES=0 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29500 \
#     -m sft.ptr_follow.finetune_ar_custom_full_loop \
#     --model 19 \
#     --pretrain_path models/ar-19M-10e18.safetensors \
#     --n_gpu 1 \
#     --bs 256 \
#     --save_freq 1000 \
#     --order middle \
#     --r2l \
#     --val_freq 1 \
#     --num_val 100

CUDA_VISIBLE_DEVICES=9 torchrun \
    --nproc_per_node=1 \
    --node_rank=0 \
    --nnodes=1 \
    --master_port=29500 \
    -m sft.ptr_follow.finetune_ar_custom_full_loop \
    --model 19 \
    --pretrain_path models/ar-19M-10e18.safetensors \
    --n_gpu 1 \
    --bs 256 \
    --save_freq 1000 \
    --order middle \
    --val_freq 1 \
    --num_val 100 \
    --r2l


# r2l training
# CUDA_VISIBLE_DEVICES=3 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29500 \
#     sft/finetune_ar_masked.py \
#     --model 336 \
#     --pretrain_path models/ar-336M-100e18.safetensors \
#     --n_gpu 1 \
#     --bs 256 \
#     --save_freq 5 \
#     --r2l