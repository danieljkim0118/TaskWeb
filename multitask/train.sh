#!/bin/bash

#SBATCH --job-name=JOB_NAME
#SBATCH --partition=PARTITION_NAME
#SBATCH --account=ACCOUNT_NAME
#SBATCH --cpus-per-task=NUM_CPUS
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:30:00
#SBATCH --array=0-2%3
#SBATCH --requeue
#SBATCH --output=SLURM_DIR

# Modify the cache variables
export CACHE_DIR="PATH_TO_CACHE"
export HF_DATASETS_CACHE="PATH_TO_CACHE"
export HF_EVALUATE_CACHE="PATH_TO_CACHE"
export HF_METRICS_CACHE="PATH_TO_CACHE"
export HF_MODULES_CACHE="PATH_TO_CACHE"
export NLTK_DATA="PATH_TO_CACHE"

# Output parent directory.
OUTPUT_DIR_NAME="PATH_TO_OUTPUT_DIR"

# Added to handle exception cases where the SLURM system crashes or something like that
if [[ -z "$SLURM_RESTART_COUNT" || "$SLURM_RESTART_COUNT" -lt 3 ]]
then
    failresponse="scontrol requeue ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    failresponse="echo \"Job failed $((SLURM_RESTART_COUNT + 1)) times, exiting\""
fi

# Helper function to convert a list of task names into a formatted string joined by underscores.
list2str() {
    LIST=("$@")
    LIST_LEN=${#LIST[@]}
    OUTPUT=${LIST[0]}
    for (( i=1; i<${LIST_LEN}; i++ )); do
        ELEMENT=${LIST[$i]}
        OUTPUT="${OUTPUT}_${ELEMENT}"
    done
    echo $OUTPUT
}

# Helper function to convert learning rate in string format to a decimal format to be used as training input.
string2decimal() {
    echo "$1*10^-0$2" | bc -l
}

# Modify the variables below. Some variables, such as training epochs, learning rate or number of samples, need to be fixed to replicate the results in our paper.

# Example of formatting the input for multitask training. The full list of all available tasks is given below (commented)
# anli_r3 boolq cb copa cosmosqa hellaswag imdb mrpc piqa qnli qqp quartz rotten_tomatoes rte scitail snli socialiqa squad_v2 stsb wic winogrande wsc
# Refer to multitask_info.py to find the list of all multitask configurations.
MULTI_TASK_LIST=(anli_r3 cosmosqa snli socialiqa wsc)

MULTI_TASK_LIST_STR="$(list2str ${MULTI_TASK_LIST[@]})"
echo MULTI_TASK_LIST_STR

# List of random seeds for training. If you choose to increase or decrease the number of seeds, then the SLURM array command on line 10 needs to be modified.
# This is because we set our training across random seeds to occur in parallel.
SEED_LIST=(15 30 42)

NUM_SEEDS=${#SEED_LIST[@]}
SEED_IDX=$((SLURM_ARRAY_TASK_ID - TASK_IDX * NUM_SEEDS))
SEED=${SEED_LIST[$SEED_IDX]}

# Input training file.
TRAIN_FILE=data/multitask/${MULTI_TASK_LIST_STR}.json

# Sample size of our data. We use 2,000 samples from each dataset.
SAMPLE_SIZE=2000

# Set the maximum length to the longest one in the multi-task training set
MAX_LENGTH=128
for (( i=0; i<${MULTI_TASK_LEN}; i++ )); do
    TASK_NOW=${MULTI_TASK_LIST[$i]}
    if [[ $TASK_NOW == "boolq" ]] && (( MAX_LENGTH < 512 )); then
        MAX_LENGTH=512
    elif [[ $TASK_NOW == "squad_v2" ]] && (( MAX_LENGTH < 512)); then
        MAX_LENGTH=512
    elif [[ $TASK_NOW == "rte" ]] && (( MAX_LENGTH < 290 )); then
        MAX_LENGTH=290
    elif [[ $TASK_NOW == "imdb" ]] && (( MAX_LENGTH < 256 )); then
        MAX_LENGTH=256
    elif [[ $TASK_NOW == "cb" ]] && (( MAX_LENGTH < 256 )); then
        MAX_LENGTH=256
    elif [[ $TASK_NOW == "hellaswag" ]] && (( MAX_LENGTH < 384 )); then
        MAX_LENGTH=384
    elif [[ $TASK_NOW == "piqa" ]] && (( MAX_LENGTH < 384 )); then
        MAX_LENGTH=384
    elif [[ $TASK_NOW == "cosmosqa" ]] && (( MAX_LENGTH < 200 )); then
        MAX_LENGTH=200
    elif [[ $TASK_NOW == "socialiqa" ]] && (( MAX_LENGTH < 200 )); then
        MAX_LENGTH=200
    fi
done

# Path to pre-trained language model.
MODEL="t5xl"
if [[ $MODEL == "t5xl" ]]; then
    MODEL_PATH=google/t5-xl-lm-adapt
fi

# Number of training epochs.
EPOCHS=5

# Learning rate. Specify the coefficient and the exponent separately as integers.
LR_COEFF=1
LR_EXPNT=4

LEARNING_RATE="$(string2decimal ${LR_COEFF} ${LR_EXPNT})"

# Learning rate scheduler.
LR_SCHEDULER_TYPE=constant

# Gradient accumulation steps.
GRAD_ACCUMULATION_STEPS=4

# Training batch size.
TRAIN_BATCH_SIZE=16

# Set the batch size to the smallest one in the multi-task training set.
for (( i=0; i<${MULTI_TASK_LEN}; i++ )); do
    TASK_NOW=${MULTI_TASK_LIST[$i]}
    if [[ $TASK_NOW == "boolq" ]] && (( TRAIN_BATCH_SIZE > 4 )); then
        TRAIN_BATCH_SIZE=4
    elif [[ $TASK_NOW == "squad_v2" ]] && (( TRAIN_BATCH_SIZE > 4 )); then
        TRAIN_BATCH_SIZE=4
    elif [[ $TASK_NOW == "hellaswag" ]] && (( TRAIN_BATCH_SIZE > 4 )); then
        TRAIN_BATCH_SIZE=4
    elif [[ $TASK_NOW == "piqa" ]] && (( TRAIN_BATCH_SIZE > 4 )); then
        TRAIN_BATCH_SIZE=4
    elif [[ $TASK_NOW == "rte" ]] && (( TRAIN_BATCH_SIZE > 8 )); then
        TRAIN_BATCH_SIZE=8
    elif [[ $TASK_NOW == "imdb" ]] && (( TRAIN_BATCH_SIZE > 8 )); then
        TRAIN_BATCH_SIZE=8
    elif [[ $TASK_NOW == "qnli" ]] && (( TRAIN_BATCH_SIZE > 8 )); then
        TRAIN_BATCH_SIZE=8
    elif [[ $TASK_NOW == "qqp" ]] && (( TRAIN_BATCH_SIZE > 8 )); then
        TRAIN_BATCH_SIZE=8
    elif [[ $TASK_NOW == "snli" ]] && (( TRAIN_BATCH_SIZE > 8 )); then
        TRAIN_BATCH_SIZE=8
    elif [[ $TASK_NOW == "cb" ]] && (( TRAIN_BATCH_SIZE > 8 )); then
        TRAIN_BATCH_SIZE=8
    elif [[ $TASK_NOW == "cosmosqa" ]] && (( TRAIN_BATCH_SIZE > 8 )); then
        TRAIN_BATCH_SIZE=8
    elif [[ $TASK_NOW == "socialiqa" ]] && (( TRAIN_BATCH_SIZE > 8 )); then
        TRAIN_BATCH_SIZE=8
    fi
done

OUTPUT_DIR=${OUTPUT_DIR_NAME}/${MULTI_TASK_LIST_STR}/${MODEL}-e${EPOCHS}-lr${LR_COEFF}e${LR_EXPNT}-b${TRAIN_BATCH_SIZE}-s${SEED}

python run_t5.py \
    --model_name_or_path $MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --train_file $TRAIN_FILE \
    --do_train \
    --max_source_length $MAX_LENGTH \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --num_train_epochs $EPOCHS \
    --seed $SEED \
    --save_strategy no \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir || $failresponse
