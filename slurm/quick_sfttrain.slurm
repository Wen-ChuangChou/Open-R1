#!/bin/bash
#SBATCH --job-name=open-r1-sft
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --partition=capella  # Adjust this for your cluster
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL

# Adjust module load for your cluster
module load release/24.04  GCCcore/13.2.0 Python/3.11.5 CUDA/12.4.0
source env/openr1/bin/activate

export NUM_GPUS=$((SLURM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
echo "Running on $SLURM_JOB_NUM_NODES nodes with $SLURM_GPUS_PER_NODE GPUs per node (total: $NUM_GPUS GPUs)"

# Get master address from Slurm
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=7010

# Calculate and set optimal threading
CPUS_PER_GPU=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE))
export OMP_NUM_THREADS=$CPUS_PER_GPU

# Create cache directories on fast NVMe storage
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
mkdir -p "${TRITON_CACHE_DIR}"

# Other environment variables
export ACCELERATE_LOG_LEVEL=info
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings in multi-process setup

# Print configuration
echo "Triton Cache: ${TRITON_CACHE_DIR}"
echo "Available space in /tmp:"
df -h /tmp

srun accelerate launch \
    --config_file recipes/accelerate_configs/zero3.yaml \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml

# Cleanup function  
cleanup() {
    echo "Job ending, cleaning up cache..."
    echo "Final cache usage:"
    du -sh "$TRITON_CACHE_DIR"/* 2>/dev/null || true
    rm -rf "$TRITON_CACHE_DIR" 2>/dev/null || true
}
trap cleanup EXIT


# srun accelerate launch \
#     --config_file recipes/accelerate_configs/zero3.yaml \
#     --rdzv_backend c10d \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     src/open_r1/sft.py \
#     --config recipes/OpenR1-Qwen-7B/sft/config.yaml


# srun accelerate launch \
#     --config_file recipes/accelerate_configs/zero3.yaml \
#     --rdzv_backend c10d \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     src/open_r1/sft.py \
#     --config recipes/OpenR1-Qwen-7B/sft/config_distill2.yaml