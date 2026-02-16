CUDA_VISIBLE_DEVICES=5,6 torchrun \
--nproc_per_node=2 \
--node_rank=0 \
--nnodes=1 \
--master_port=29507 \
pretrain/train_ar_constrained.py --model 206 --flops 1. --n_gpu 2 --micro_batch_size 16