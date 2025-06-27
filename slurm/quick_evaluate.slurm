#!/bin/bash
#SBATCH --job-name=open-r1-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=110G
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:30:00
#SBATCH --partition=capella
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL

# sbatch slurm/quick_evaluate.slurm math_500 data/OpenR1-Distill-7B_Math-220k
# Test tasks: aime24, math_500, gpqa:diamond, codegeneration
# Test models: data/OpenR1-Distill-7B_Math-220k, data/OpenR1-Distill-7B, Qwen/Qwen2.5-Math-7B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

module load release/24.04  GCCcore/13.2.0 Python/3.11.5 CUDA/12.4.0
source env/openr1/bin/activate

TASK=$1
MODEL_ID=$2

# Set the total number of GPUs
export NUM_GPUS=$((SLURM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
echo "Running on $SLURM_JOB_NUM_NODES nodes with $SLURM_GPUS_PER_NODE GPUs per node (total: $NUM_GPUS GPUs)"

# Get unique job-specific cache directory on local NVMe
export VLLM_CACHE_DIR="/tmp/vllm_cache_${SLURM_JOB_ID}"
export TORCH_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"
export VLLM_TORCH_COMPILE_CACHE_DIR="${VLLM_CACHE_DIR}/torch_compile_cache"
# export TORCH_COMPILE_DEBUG_DIR="${TORCH_CACHE_DIR}/compile_debug"


# Create cache directories
mkdir -p ${VLLM_CACHE_DIR}/{torch_compile_cache}
mkdir -p ${TORCH_CACHE_DIR}

# Set environment variables for cache
# export VLLM_TORCH_COMPILE_CACHE_DIR="${VLLM_CACHE_DIR}/torch_compile_cache"
# export TORCHDYNAMO_CACHE_DIR="${TORCH_CACHE_DIR}/dynamo_cache"

echo "VLLM Cache: ${VLLM_CACHE_DIR}"
echo "Torch Cache: ${TORCH_CACHE_DIR}"
echo "Available space in /tmp:"
df -h /tmp

export VLLM_WORKER_MULTIPROC_METHOD=spawn

export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_USAGE_STATS=1
export RAY_WORKER_TIMEOUT=300
export RAY_TASK_TIMEOUT=3600
# export CUDA_VISIBLE_DEVICES=0,1,2,3

export OMP_NUM_THREADS=4
export MALLOC_ARENA_MAX=4
ulimit -n 65536

# MODEL_ARGS="model_name=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,max_num_batched_tokens=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:16384,temperature:0.6,top_p:0.95}"
MODEL_ARGS="model_name=$MODEL_ID,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals

if [ "$TASK" = "codegeneration" ]; then
    echo "Running code generation evaluation..."
    lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR \
        # --max-samples 100
else
    echo "Running standard evaluation for task: $TASK"
    lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
        --use-chat-template \
        --output-dir $OUTPUT_DIR
fi

# lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR

# Cleanup cache after job completes
# echo "Cleaning up cache directories..."
# rm -rf ${VLLM_CACHE_DIR}
# rm -rf ${TORCH_CACHE_DIR}
# echo "Cache cleanup completed."

# Cleanup function  
cleanup() {
    echo "Cleaning up..."
    
    # Force cleanup of Ray processes
    pkill -f "ray" 2>/dev/null
    
    echo "Final cache usage:"
    du -sh "$VLLM_CACHE_DIR"/* 2>/dev/null || true
    du -sh "$TORCH_CACHE_DIR"/* 2>/dev/null || true

    rm -rf "$VLLM_CACHE_DIR" 2>/dev/null
    rm -rf "$TORCH_CACHE_DIR" 2>/dev/null
    
    echo "Cleanup completed"
}
trap cleanup EXIT

# MODEL=data/Qwen2.5-1.5B-Open-R1-Distill
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# TASK=aime24
# OUTPUT_DIR=data/evals/$MODEL

# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR 

# export VLLM_WORKER_MULTIPROC_METHOD=spawn # Required for vLLM

# MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
# TASK=aime24
# OUTPUT_DIR=data/evals/$MODEL

# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --use-chat-template \
#     --output-dir $OUTPUT_DIR