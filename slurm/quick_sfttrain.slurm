#!/bin/bash
#SBATCH --job-name=open-r1-sft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition=capella  # Adjust this for your cluster
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL


module load release/24.04  GCCcore/13.2.0 Python/3.11.5 CUDA/12.4.0
source env/openr1/bin/activate

export NUM_GPUS=$((SLURM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
echo "Running on $SLURM_JOB_NUM_NODES nodes with $SLURM_GPUS_PER_NODE GPUs per node (total: $NUM_GPUS GPUs)"

# Get master address from Slurm
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=7010

# srun accelerate launch --config_file recipes/accelerate_configs/zero3_1node.yaml src/open_r1/sft.py \
#     --config recipes/Qwen2.5-1.5B-Instruct/sft/my_config.yaml


# srun accelerate launch \
#     --config_file recipes/accelerate_configs/zero3.yaml \
#     --rdzv_backend c10d \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     src/open_r1/sft.py \
#     --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml

srun accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    src/open_r1/sft.py \
    --config recipes/OpenR1-Qwen-7B/sft/config.yaml