CUDA_VISIBLE_DEVICES=7,8 torchrun \
--nproc_per_node=2 \
--node_rank=0 \
--nnodes=1 \
--master_port=29506 \
pretrain/train_mdm_constrained.py --model 472 --flops 1. --n_gpu 2 --micro_batch_size 8