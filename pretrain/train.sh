# CUDA_VISIBLE_DEVICES=8 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29500 \
#     pretrain/train_ar_custom.py --model 336 --flops 100. --n_gpu 1

# CUDA_VISIBLE_DEVICES=3 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29501 \
#     pretrain/train_mdm_custom.py --model 336 --flops 100. --n_gpu 1

# CUDA_VISIBLE_DEVICES=9 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29502 \
#     pretrain/train_mad_custom.py --model 170 --flops 100. --n_gpu 1

CUDA_VISIBLE_DEVICES=9 torchrun \
    --nproc_per_node=1 \
    --node_rank=0 \
    --nnodes=1 \
    --master_port=29503 \
    pretrain/train_mad_custom.py --model 170 --flops 100. --n_gpu 1 --r2l