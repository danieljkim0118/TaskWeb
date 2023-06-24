#!/bin/bash

#SBATCH --job-name=JOB_NAME
#SBATCH --partition=PARTITION_NAME
#SBATCH --account=ACCOUNT_NAME
#SBATCH --cpus-per-task=NUM_CPUS
#SBATCH --mem=24G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --array=0-14%15
#SBATCH --requeue
#SBATCH --output=SLURM_DIR

# Modify the cache variables
export CACHE_DIR="PATH_TO_CACHE"
export HF_DATASETS_CACHE="PATH_TO_CACHE"
export HF_EVALUATE_CACHE="PATH_TO_CACHE"
export HF_METRICS_CACHE="PATH_TO_CACHE"
export HF_MODULES_CACHE="PATH_TO_CACHE"
export NLTK_DATA="PATH_TO_CACHE"

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

# List of target tasks. The target task must be repeated along with the corresponding prompt.
TARGET_TASK_LIST=(cb cb cb cb cb cb)

# List of prompts to be evaluated for the target task. Refer to promptsource (https://github.com/bigscience-workshop/promptsource) for more info.
PROMPT_NAME_LIST=(does_it_follow_that based_on_the_previous_passage can_we_infer does_this_imply must_be_true should_assume)

# Example of formatting the input for multitask training. The full list of all available tasks is given below (commented)
# anli_r3 boolq cb copa cosmosqa hellaswag imdb mrpc piqa qnli qqp quartz rotten_tomatoes rte scitail snli socialiqa squad_v2 stsb wic winogrande wsc
# Refer to multitask_info.py to find the list of all multitask configurations.
MULTI_TASK_LIST=(anli_r3 cosmosqa snli socialiqa wsc)

# Input directory name
INPUT_DIR_NAME="MULTITASK"

# Output directory name
OUTPUT_DIR_NAME="ZEROSHOT"

# List of random seeds for training. If you choose to increase or decrease the number of seeds, then the SLURM array command on line 10 needs to be modified.
# This is because we set our training across random seeds to occur in parallel.
SEED_LIST=(15 30 42)

# Based on our target task, prompt and seed input, we automatically assign parallel evaluation runs to multiple GPUs. 
NUM_TARGET_TASKS=${#TARGET_TASK_LIST[@]}
NUM_SEEDS=${#SEED_LIST[@]}

TARGET_TASK_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
TARGET_TASK_NAME=${TARGET_TASK_LIST[$TARGET_TASK_IDX]}
PROMPT_NAME=${PROMPT_NAME_LIST[$TARGET_TASK_IDX]}

SEED_IDX=$((SLURM_ARRAY_TASK_ID - TARGET_TASK_IDX * NUM_SEEDS))
SOURCE_SEED=${SEED_LIST[$SEED_IDX]}
TARGET_SEED=$SOURCE_SEED

# Set task category for evaluation.
if [[ $TARGET_TASK_NAME == "squad_v2" ]]; then
    TARGET_TASK_CATEGORY=extractive_qa
elif [[ $TARGET_TASK_NAME == "stsb" ]]; then
    TARGET_TASK_CATEGORY=regression
else
    TARGET_TASK_CATEGORY=classification
fi

# Set metric for evaluation.
if [[ $TARGET_TASK_CATEGORY == "classification" ]]; then
    if [[ $TARGET_TASK_NAME == "boolq" ]] || [[ $TARGET_TASK_NAME == "cb" ]] || [[ $TARGET_TASK_NAME == "copa" ]] || [[ $TARGET_TASK_NAME == "wic" ]] || [[ $TARGET_TASK_NAME == "wsc" ]]; then
        TARGET_TASK_METRIC=super_glue
    else
        TARGET_TASK_METRIC=glue
    fi
elif [[ $TARGET_TASK_CATEGORY == "regression" ]]; then
    TARGET_TASK_METRIC=glue
fi

# Load validation file
VAL_FILE=data/eval/${TARGET_TASK_NAME}_${PROMPT_NAME}.json

if [[ $TARGET_TASK_NAME == "squad_v2" ]] || [[ $TARGET_TASK_NAME == "boolq" ]]; then
    MAX_LENGTH=512
elif [[ $TARGET_TASK_NAME == "hellaswag" ]] || [[ $TARGET_TASK_NAME == "piqa" ]]; then
    MAX_LENGTH=384
elif [[ $TARGET_TASK_NAME == "rte" ]]; then
    MAX_LENGTH=290
elif [[ $TARGET_TASK_NAME == "imdb" ]]; then
    MAX_LENGTH=256
elif [[ $TARGET_TASK_NAME == "cb" ]]; then
    MAX_LENGTH=256
elif [[ $TARGET_TASK_NAME == "cosmosqa" ]]; then
    MAX_LENGTH=200
elif [[ $TARGET_TASK_NAME == "socialiqa" ]]; then
    MAX_LENGTH=200
elif [[ $TARGET_TASK_NAME == "rotten_tomatoes" ]] || [[ $TARGET_TASK_NAME == "scitail" ]]; then
    MAX_LENGTH=80
elif [[ $TARGET_TASK_NAME == "copa" ]]; then
    MAX_LENGTH=60
elif [[ $TARGET_TASK_NAME == "wic" ]]; then
    MAX_LENGTH=50
else
    MAX_LENGTH=128
fi

# Training epochs of the multi-task model.
SOURCE_EPOCHS=5

# Learning rate of the multi-task model.
SOURCE_LR_COEFF=1
SOURCE_LR_EXPNT=4

# Training batch size of the multi-task model.
SOURCE_BATCH_SIZE=16

# Evaluation batch size.
TARGET_BATCH_SIZE=16
for (( i=0; i<${MULTI_TASK_LEN}; i++ )); do
    TASK_NOW=${MULTI_TASK_LIST[$i]}
    if [[ $TASK_NOW == "boolq" ]] && (( SOURCE_BATCH_SIZE > 4 )); then
        SOURCE_BATCH_SIZE=4
    elif [[ $TASK_NOW == "squad_v2" ]] && (( SOURCE_BATCH_SIZE > 4 )); then
        SOURCE_BATCH_SIZE=4
    elif [[ $TASK_NOW == "hellaswag" ]] && (( SOURCE_BATCH_SIZE > 4 )); then
        SOURCE_BATCH_SIZE=4
    elif [[ $TASK_NOW == "piqa" ]] && (( SOURCE_BATCH_SIZE > 4 )); then
        SOURCE_BATCH_SIZE=4
    elif [[ $TASK_NOW == "rte" ]] && (( SOURCE_BATCH_SIZE > 8 )); then
        SOURCE_BATCH_SIZE=8
    elif [[ $TASK_NOW == "imdb" ]] && (( SOURCE_BATCH_SIZE > 8 )); then
        SOURCE_BATCH_SIZE=8
    elif [[ $TASK_NOW == "qnli" ]] && (( SOURCE_BATCH_SIZE > 8 )); then
        SOURCE_BATCH_SIZE=8
    elif [[ $TASK_NOW == "qqp" ]] && (( SOURCE_BATCH_SIZE > 8 )); then
        SOURCE_BATCH_SIZE=8
    elif [[ $TASK_NOW == "snli" ]] && (( SOURCE_BATCH_SIZE > 8 )); then
        SOURCE_BATCH_SIZE=8
    elif [[ $TASK_NOW == "cb" ]] && (( SOURCE_BATCH_SIZE > 8 )); then
        SOURCE_BATCH_SIZE=8
    elif [[ $TASK_NOW == "cosmosqa" ]] && (( SOURCE_BATCH_SIZE > 8 )); then
        SOURCE_BATCH_SIZE=8
    elif [[ $TASK_NOW == "socialiqa" ]] && (( SOURCE_BATCH_SIZE > 8 )); then
        SOURCE_BATCH_SIZE=8
    fi
done

# The source model is from the input folder.
SOURCE_MODEL_PATH=${INPUT_DIR_NAME}/${MULTI_TASK_LIST_STR}/${MODEL}-e${SOURCE_EPOCHS}-lr${SOURCE_LR_COEFF}e${SOURCE_LR_EXPNT}-b${SOURCE_BATCH_SIZE}-s${SOURCE_SEED}

# The target model is contained in the current outputs folder, with its name indicating both source and target hyperparameters.
TARGET_MODEL_PATH=${OUTPUT_DIR_NAME}/${TARGET_TASK_NAME}/${MULTI_TASK_LIST_STR}/${MODEL}-${PROMPT_NAME}-e${SOURCE_EPOCHS}-lr${SOURCE_LR_COEFF}e${SOURCE_LR_EXPNT}-s${TARGET_SEED}

python run_t5.py \
    --model_name_or_path $SOURCE_MODEL_PATH \
    --cache_dir $CACHE_DIR \
    --dataset_category $TARGET_TASK_CATEGORY \
    --dataset_name_custom $TARGET_TASK_NAME \
    --validation_file $VAL_FILE \
    --metric $TARGET_TASK_METRIC \
    --do_eval \
    --max_source_length $MAX_LENGTH \
    --per_device_eval_batch_size $TARGET_BATCH_SIZE \
    --predict_with_generate \
    --seed $TARGET_SEED \
    --save_strategy no \
    --output_dir $TARGET_MODEL_PATH \
    --overwrite_output_dir || $failresponse
