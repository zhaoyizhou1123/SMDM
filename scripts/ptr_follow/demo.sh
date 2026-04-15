export CUDA_VISIBLE_DEVICES=1
# python sft/ptr_follow/demo.py --model 1028 \
#     --order middle \
#     --l2r_path workdir/finetune/arm-1028M-masked-middle/iter-006400-ckpt.safetensors \
#     --r2l_path workdir/finetune/arm-1028M-masked-middle-r2l/iter-006400-ckpt.safetensors

# This script is working correctly. Do not touch
# python sft/ptr_follow/demo.py --model 336 \
#     --order middle \
#     --ckpt_path /home/zhaoyiz/projects/SMDM/workdir/finetune/arm-336M-masked-middle-double-corrected/iter-032000-ckpt.pth

python sft/ptr_follow/demo_v3.py --model 336 \
    --order middle \
    --ckpt_path /home/zhaoyiz/projects/SMDM/workdir/finetune/arm-336M-masked-middle-double-corrected/iter-032000-ckpt.pth
