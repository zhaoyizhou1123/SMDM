export CUDA_VISIBLE_DEVICES=6

SIZE=206
# CKPT_PATH_L2R=/home/zhaoyiz/projects/SMDM/workdir/finetune/arm-336M-masked-middle/iter-000960-ckpt.pth
CKPT_PATH=workdir/finetune/trigpt-206M-masked-reverse/iter-003200-ckpt.pth
# if pth path, convert to safetensors
# CKPT_PATH=/home/zhaoyiz/projects/SMDM/workdir/finetune/mdm-sudoku-336M-v2/iter-005120-ckpt.safetensors
# python evaluate_gsm8k.py --ckpt_path $CKPT_PATH
# python evaluate_sudoku.py --model 336 --ckpt_path $CKPT_PATH --cfg1 0 --cfg2 0 --length 272
# python evaluate_ptr_follow.py --model 336 --ckpt_path $CKPT_PATH --cfg1 0 --cfg2 0 --length 272
python evaluate_trigpt.py \
    --model $SIZE \
    --ckpt_path $CKPT_PATH \
    --length 16 \
    --steps 16 \
    --temperature 0.0 \
    --data_path zzy1123/ptr_follow_reverse_order_rl
