CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node=2 \
    --node_rank=0 \
    --nnodes=1 \
    --master_port=29502 \
    -m sft.finetune_mdm_gsm8k \
    --model 1028 \
    --pretrain_path models/mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors \
    --n_gpu 2 \
    --bs 256
