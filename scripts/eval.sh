export CUDA_VISIBLE_DEVICES=0
CKPT_PATH=workdir/finetune/mdm-gsm8k-1028M/iter-020000-ckpt.pth
# CKPT_PATH="models/mdm-1028M-3300e18-rsl-gsm8k.safetensors"
# if pth path, convert to safetensors
if [[ $CKPT_PATH == *.pth ]]; then
    SAFETENSOR_PATH=${CKPT_PATH%.pth}.safetensors
    python scripts/ckpt_convert.py --pth_path $CKPT_PATH --safetensor_path $SAFETENSOR_PATH
    CKPT_PATH=$SAFETENSOR_PATH
fi
python evaluate_gsm8k.py --ckpt_path $CKPT_PATH