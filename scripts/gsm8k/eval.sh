CUDA_VISIBLE_DEVICES=4 python evaluate_gsm8k.py \
    --ckpt_path workdir/finetune/mdm-gsm8k-1028M/iter-080000-ckpt.safetensors \
    --steps 64 \
    --length 256 \
    --cfg1 0 \
    --cfg2 0 \
    --temperature 0.1
