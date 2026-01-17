export CUDA_VISIBLE_DEVICES=3
CKPT_PATH_L2R=/home/zhaoyiz/projects/SMDM/workdir/finetune/arm-336M-masked/iter-005120-ckpt.safetensors
CKPT_PATH_R2L=/home/zhaoyiz/projects/SMDM/workdir/finetune/arm-336M-masked-r2l/iter-000960-ckpt.safetensors
# if pth path, convert to safetensors
# if [[ $CKPT_PATH == *.pth ]]; then
#     SAFETENSOR_PATH=${CKPT_PATH%.pth}.safetensors
#     python scripts/ckpt_convert.py --pth_path $CKPT_PATH --safetensor_path $SAFETENSOR_PATH
#     CKPT_PATH=$SAFETENSOR_PATH
# fi
# CKPT_PATH=/home/zhaoyiz/projects/SMDM/workdir/finetune/mdm-sudoku-336M-v2/iter-005120-ckpt.safetensors
# python evaluate_gsm8k.py --ckpt_path $CKPT_PATH
# python evaluate_sudoku.py --model 336 --ckpt_path $CKPT_PATH --cfg1 0 --cfg2 0 --length 272
# python evaluate_ptr_follow.py --model 336 --ckpt_path $CKPT_PATH --cfg1 0 --cfg2 0 --length 272
python evaluate_masked_ar.py --model 336 --ckpt_path_l2r $CKPT_PATH_L2R --ckpt_path_r2l $CKPT_PATH_R2L --length 16 --temperature 0.0