#!/bin/bash
#SBATCH --job-name=open-r1-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --partition=capella
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --mail-user=wen-chuang.chou@tu-dresden.de
#SBATCH --mail-type=ALL

## Example usage:
## sbatch slurm/my_evaluate.slurm GPQA gpqa:diamond data/OpenR1-Qwen-7B-SFT
##                                 |     |      |
##                                 |     |      └── Path to the model (MODEL_ID)
##                                 |     └──────── Benchmark/task name (TASKS)
##                                 └────────────── Job name (TASK_NAME)

module load release/24.04  GCCcore/13.2.0 Python/3.11.5 CUDA/12.4.0

# Needed for vLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn

set -x -e

source ~/.bashrc
source env/openr1/bin/activate

# Set the total number of GPUs
export NUM_GPUS=$((SLURM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES))
echo "Running on $SLURM_JOB_NUM_NODES nodes with $SLURM_GPUS_PER_NODE GPUs per node (total: $NUM_GPUS GPUs)"

TASK_NAME=$1
TASKS=$2
MODEL_ID=$3
# MODEL_REVISION=$4
# Optional args
[ -z "$5"] && TENSOR_PARALLEL=False || TENSOR_PARALLEL=$5
[ -z "$6"] && TRUST_REMOTE_CODE=False || TRUST_REMOTE_CODE=$6
# $7 is reserved for system_prompt, see line 51
# NUM_GPUS=$(nvidia-smi -L | wc -l)

# Use TP to shard model across GPUs
if [ "$TENSOR_PARALLEL" = "True" ]; then
    # MODEL_ARGS="model_name=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
    MODEL_ARGS="model_name=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,max_num_batched_tokens=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
else
    # MODEL_ARGS="model_name=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
    MODEL_ARGS="model_name=$MODEL_ID,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,max_num_batched_tokens=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
fi

LM_EVAL_REPO_ID="open-r1/open-r1-eval-leaderboard"
MODEL_NAME=$(echo $MODEL_ID | sed 's/\//_/g') # replaces / with _
DETAILS_REPO_ID="open-r1/details-$MODEL_NAME"
# OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME"
OUTPUT_DIR="eval_results/$MODEL_ID/$TASK_NAME"
# We need this flag since we run this script from training jobs that use DeepSpeed and the env vars get progated which causes errors during evaluation
ACCELERATE_USE_DEEPSPEED=false

echo "Running lighteval script ..."
echo "Eval results will be saved to $OUTPUT_DIR"
lighteval vllm "$MODEL_ARGS" "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details \
    ${7:+--system-prompt "$(echo "$7" | base64 --decode)"}

OUTPUT_FILEPATHS=$(find $OUTPUT_DIR/results/ -type f \( -name "*.json" \))
for filepath in $OUTPUT_FILEPATHS; do
    echo "Uploading $filepath to Hugging Face Hub..."
    filename=$(basename -- "$filepath")
    for attempt in {1..20}; do
        if huggingface-cli upload --repo-type space --private $LM_EVAL_REPO_ID $filepath $OUTPUT_DIR/$filename; then
            echo "Upload succeeded for $filepath"
            break
        else
            echo "Upload failed for $filepath. Attempt $attempt of 20. Retrying in 5 seconds..."
            sleep 5
        fi
    done
done

echo "Uploading details to Hugging Face Hub..."
DETAILS_FILEPATHS=$(find $OUTPUT_DIR/details/ -type f \( -name "*.parquet" \))
echo "DETAILS_FILEPATHS: $DETAILS_FILEPATHS"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
# python scripts/upload_details.py --data_files $DETAILS_FILEPATHS --hub_repo_id $DETAILS_REPO_ID --config_name $MODEL_REVISION.$TASK_NAME.$TIMESTAMP
python scripts/upload_details.py --data_files $DETAILS_FILEPATHS --hub_repo_id $DETAILS_REPO_ID --config_name $TASK_NAME.$TIMESTAMP
    
echo "Cleaning up ..."
rm -rf $OUTPUT_DIR

echo "Done!"
