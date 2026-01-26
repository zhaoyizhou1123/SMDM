export CUDA_VISIBLE_DEVICES=2

SIZE=1028
# CKPT_PATH_L2R=/home/zhaoyiz/projects/SMDM/workdir/finetune/arm-336M-masked-middle/iter-000960-ckpt.pth
CKPT_PATH_L2R=/home/zhaoyiz/projects/SMDM/workdir/finetune/arm-1028M-masked-middle/iter-006400-ckpt.pth
CKPT_PATH_R2L=/home/zhaoyiz/projects/SMDM/workdir/finetune/arm-1028M-masked-middle-r2l/iter-006400-ckpt.pth
# if pth path, convert to safetensors

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

CKPT_PATH_L2R=$(ensure_safetensors "$CKPT_PATH_L2R")
CKPT_PATH_R2L=$(ensure_safetensors "$CKPT_PATH_R2L")
# CKPT_PATH=/home/zhaoyiz/projects/SMDM/workdir/finetune/mdm-sudoku-336M-v2/iter-005120-ckpt.safetensors
# python evaluate_gsm8k.py --ckpt_path $CKPT_PATH
# python evaluate_sudoku.py --model 336 --ckpt_path $CKPT_PATH --cfg1 0 --cfg2 0 --length 272
# python evaluate_ptr_follow.py --model 336 --ckpt_path $CKPT_PATH --cfg1 0 --cfg2 0 --length 272
python evaluate_masked_ar.py \
    --model $SIZE \
    --ckpt_path_l2r $CKPT_PATH_L2R \
    --ckpt_path_r2l $CKPT_PATH_R2L \
    --length 16 \
    --temperature 0.0 \
    --data_path zzy1123/ptr_follow_middle_order_rl
