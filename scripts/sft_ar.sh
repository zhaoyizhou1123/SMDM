CUDA_VISIBLE_DEVICES=2 torchrun \
    --nproc_per_node=1 \
    --node_rank=0 \
    --nnodes=1 \
    --master_port=29500 \
    sft/finetune_ar_masked.py \
    --model 1028 \
    --pretrain_path models/ar-1028M-100e18.safetensors \
    --n_gpu 1 \
    --bs 256 \
    --save_freq 50 \
    --order middle \
    --r2l

# CUDA_VISIBLE_DEVICES=6 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29501 \
#     sft/finetune_ar_masked.py \
#     --model 336 \
#     --pretrain_path models/ar-336M-100e18.safetensors \
#     --n_gpu 1 \
#     --bs 256 \
#     --save_freq 100 \
#     --order middle \
#     --r2l


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