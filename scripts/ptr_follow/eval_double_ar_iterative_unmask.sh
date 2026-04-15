CUDA_VISIBLE_DEVICES=9 torchrun \
    --nproc_per_node=1 \
    --node_rank=0 \
    --nnodes=1 \
    --master_port=29501 \
    -m sft.ptr_follow.eval_double_ar_iterative_unmask \
    --model 336 \
    --n_gpu 1 \
    --bs 256 \
    --order middle \
    --num_val 100 \
    --num_data 100 \
    --debug_n 2 \
    --ckpt_path workdir/finetune/arm-19M-masked-middle-double/iter-007812-ckpt.pth

# CUDA_VISIBLE_DEVICES=9 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29501 \
#     -m sft.ptr_follow.eval_double_ar_iterative_unmask \
#     --model 19 \
#     --n_gpu 1 \
#     --bs 256 \
#     --order middle \
#     --num_val 100 \
#     --num_data 100 \
#     --debug_n 2 \
#     --l2r_path /home/zhaoyiz/projects/SMDM/workdir/finetune/arm-19M-masked-middle/iter-000624-ckpt.pth \
#     --r2l_path /home/zhaoyiz/projects/SMDM/workdir/finetune/arm-19M-masked-middle-r2l/iter-000624-ckpt.pth