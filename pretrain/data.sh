# python pretrain/prepare_slimpajama.py --source_path /home/share/dataset/SlimPajama-627B/  --tokenizer_path /home/zhaoyiz/projects/rl_reasoning/models/Ar_LLaMA_R2L_336M_ptr_follow_reverse_order_v2_sft_384  --destination_path data/slim_star_combined --split train --percentage 0.001

# python pretrain/prepare_slimpajama.py --source_path /home/share/dataset/SlimPajama-627B/  --tokenizer_path /home/zhaoyiz/projects/rl_reasoning/models/Ar_LLaMA_R2L_336M_ptr_follow_reverse_order_v2_sft_384  --destination_path /home/zhaoyiz/projects/SMDM/data/slim_star_combined --split validation --percentage 1

# python pretrain/prepare_jsonl.py --input_file "/home/zhaoyiz/projects/diffusion-data-constraint/data/c4_10m.jsonl" \
#     --tokenizer_path /home/zhaoyiz/projects/rl_reasoning/models/Ar_LLaMA_R2L_336M_ptr_follow_reverse_order_v2_sft_384  \
#     --destination_path /home/zhaoyiz/projects/SMDM/data/c4_10m \
#     --split train --percentage 1

python pretrain/prepare_jsonl.py --input_file "/home/zhaoyiz/projects/diffusion-data-constraint/data/c4_validation.jsonl" \
    --tokenizer_path /home/zhaoyiz/projects/rl_reasoning/models/Ar_LLaMA_R2L_336M_ptr_follow_reverse_order_v2_sft_384  \
    --destination_path /home/zhaoyiz/projects/SMDM/data/c4_validation \
    --split validation --percentage 1