#!/bin/bash

PYTHON_SCRIPT="./Experiments/ica_exp.py"
TARGET_MODEL="google/gemma-7b-it"
FEW_SHOT_NUM=1
EARLY_STOP=True
EOS_NUM=20
LOG_PATH="Logs/${TARGET_MODEL}/ICA-${FEW_SHOT_NUM}"

# Create the log directory if it does not exist
mkdir -p "$LOG_PATH"

# Conditional flag for EARLY_STOP
EARLY_STOP_FLAG=""
if [ "$EARLY_STOP" = "True" ]; then
    EARLY_STOP_FLAG="--early_stop"
fi

# Function to find the first available GPU
find_free_gpu() {
    for i in {0..7}; do
        free_mem=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}')
        if [ "$free_mem" -ge 40000 ]; then
            echo $i
            return
        fi
    done

    echo "-1" # Return -1 if no suitable GPU is found
}

# Start the jobs with GPU assignment
for index in {0..127}; do

    FREE_GPU=-1

    # Keep looping until a free GPU is found
    while [ $FREE_GPU -eq -1 ]; do
        FREE_GPU=$(find_free_gpu)
        if [ $FREE_GPU -eq -1 ]; then
            sleep 5 # Wait for 5 seconds before trying to find a free GPU again
        fi
    done

    # Run the Python script on the free GPU
    (
        echo "Task $index started on GPU $FREE_GPU."
        CUDA_VISIBLE_DEVICES=$FREE_GPU python -u "$PYTHON_SCRIPT" --index $index --target_model $TARGET_MODEL --few_shot_num $FEW_SHOT_NUM --max-new-tokens 512 $EARLY_STOP_FLAG --eos_num $EOS_NUM > "${LOG_PATH}/${index}.log" 2>&1
        echo "Task $index on GPU $FREE_GPU finished."
    ) &

    # Wait for 50 seconds to give the GPU some time to allocate memory
    sleep 50
done

# Wait for all background jobs to finish
wait
