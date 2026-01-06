# export CUDA_VISIBLE_DEVICES=2
# CKPT_PATH=workdir/finetune/mdm-gsm8k-1028M/iter-020000-ckpt.pth
# CKPT_PATH="models/mdm-1028M-3300e18-rsl-gsm8k.safetensors"
# CKPT_PATH=/home/zhaoyiz/projects/SMDM/workdir/finetune/mdm-sudoku-336M/iter-160000-ckpt.pth
CKPT_PATH=/home/zhaoyiz/projects/SMDM/workdir/finetune/mdm-sudoku_simple-336M/iter-000320-ckpt.pth
# if pth path, convert to safetensors
if [[ $CKPT_PATH == *.pth ]]; then
    SAFETENSOR_PATH=${CKPT_PATH%.pth}.safetensors
    python scripts/ckpt_convert.py --pth_path $CKPT_PATH --safetensor_path $SAFETENSOR_PATH
    CKPT_PATH=$SAFETENSOR_PATH
fi
# CKPT_PATH=/home/zhaoyiz/projects/SMDM/workdir/finetune/mdm-sudoku-336M-v2/iter-005120-ckpt.safetensors
# python evaluate_gsm8k.py --ckpt_path $CKPT_PATH
python evaluate_sudoku.py --model 336 --ckpt_path $CKPT_PATH --cfg1 0 --cfg2 0 --length 272