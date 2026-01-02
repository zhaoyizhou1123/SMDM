#!/bin/bash
#SBATCH --job-name=sudoku_sft    # 作业名称
#SBATCH --array=0
# SBATCH --nodelist=node-gpu02
#SBATCH --output=slurm/%x/job_%A_%a.out                   # 输出日志路径
#SBATCH --nodes=1                              # 节点数
#SBATCH --ntasks-per-node=1                    # 每个节点的任务数
#SBATCH --cpus-per-task=32                     # 每个任务的 CPU 核心数
# SBATCH --partition=HGPU                       # 指定分区
#SBATCH --gres=gpu:H100:2                        # 需要 1 个 GPU
#SBATCH --time=48:00:00                      # 最大运行时间
#SBATCH --chdir=/home/zhaoyiz/projects/SMDM

set -x

export n_gpu=1

dir=scripts
# main=test.sh
# main=sudoku.sh
main=sft_slurm.sh

# export MODEL=models/mdm-1028M-3300e18-rsl-0.01-bs-1024.safetensors
export MODEL=models/mdm-336M-100e18.safetensors
export PARAM=336
export POSTFIX=v2_5


export TRITON_LIBCUDA_PATH="/.singularity.d/libs/"
local_dir=.local_newverl
apptainer exec --nv -B $HOME/$local_dir:$HOME/.local,/mnt/cephfs/cluster/dgx/users/zhaoyiz:/workspace --env-file  $HOME/containers/env.txt $HOME/containers/pytorch_24.05-py3.sif \
bash $dir/$main
