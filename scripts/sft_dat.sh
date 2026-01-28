CKPT_PATH=workdir/pretrain/dat-336M-100.0/iter-008000-ckpt.pth

ensure_safetensors() {
    local input_path="$1"
    
    # Check if the file ends with .pth
    if [[ "$input_path" == *.pth ]]; then
        local safetensor_path="${input_path%.pth}.safetensors"
        
        # Run the python script. 
        # IMPORTANT: We redirect output to >&2 (stderr) so that logs 
        # don't get mixed into the variable we are "returning".
        python scripts/ckpt_convert.py --pth_path "$input_path" --safetensor_path "$safetensor_path" >&2
        
        # Update our local variable to the new path
        input_path="$safetensor_path"
    fi
    
    # "Return" the final path by printing it
    echo "$input_path"
}
CKPT_PATH=$(ensure_safetensors "$CKPT_PATH")

# CUDA_VISIBLE_DEVICES=0 torchrun \
#     --nproc_per_node=1 \
#     --node_rank=0 \
#     --nnodes=1 \
#     --master_port=29500 \
#     sft/finetune_dat_v2.py \
#     --model 336 \
#     --pretrain_path $CKPT_PATH \
#     --n_gpu 1 \
#     --bs 256 \
#     --save_freq 10

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --node_rank=0 \
    --nnodes=1 \
    --master_port=29500 \
    sft/finetune_dat.py \
    --model 336 \
    --pretrain_path $CKPT_PATH \
    --n_gpu 1 \
    --bs 256 \
    --save_freq 50