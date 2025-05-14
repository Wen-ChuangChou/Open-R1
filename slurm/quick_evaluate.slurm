#!/bin/bash
#SBATCH --job-name=open-r1-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --partition=capella
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL

module load release/24.04  GCCcore/13.2.0 Python/3.11.5 CUDA/12.4.0
source env/openr1/bin/activate

# Set the total number of GPUs
export NUM_GPUS=$((SLURM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
echo "Running on $SLURM_JOB_NUM_NODES nodes with $SLURM_GPUS_PER_NODE GPUs per node (total: $NUM_GPUS GPUs)"

MODEL=data/Qwen2.5-1.5B-Open-R1-GRPO
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 

# export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# TASK=aime24
# OUTPUT_DIR=data/evals/$MODEL

# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR