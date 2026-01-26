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

# CUDA_VISIBLE_DEVICES=9 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29503 \
#     pretrain/train_mad_custom.py --model 170 --flops 100. --n_gpu 1 --r2l

CUDA_VISIBLE_DEVICES=5,6,7,8 torchrun \
--nproc_per_node=4 \
--node_rank=0 \
--nnodes=1 \
--master_port=29504 \
pretrain/train_dat_custom.py --model 336 --flops 100. --n_gpu 4

# CUDA_VISIBLE_DEVICES=2 torchrun \
# --nproc_per_node=1 \
# --node_rank=0 \
# --nnodes=1 \
# --master_port=29505 \
# pretrain/train_dat_v2_custom.py --model 336 --flops 100. --n_gpu 1