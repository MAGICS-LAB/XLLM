#!/bin/bash

PYTHON_SCRIPT="./Experiments/gcg_exp.py"
MODEL_PATH="google/gemma-7b-it"
ADD_EOS=True
RUN_INDEX=2
# Set the log path based on ADD_EOS
if [ "$ADD_EOS" = "True" ]; then
    LOG_PATH="Logs/${MODEL_PATH}/GCG_eos-${RUN_INDEX}"
else
    LOG_PATH="Logs/${MODEL_PATH}/GCG-${RUN_INDEX}"
fi

# Create the log directory if it does not exist
mkdir -p "$LOG_PATH"

# Conditional flag for ADD_EOS
ADD_EOS_FLAG=""
if [ "$ADD_EOS" = "True" ]; then
    ADD_EOS_FLAG="--add_eos"
fi

# Function to find the first available GPU
find_free_gpu() {
    for i in {0..7}; do
        free_mem=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}')
        if [ "$free_mem" -ge 60000 ]; then
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
        CUDA_VISIBLE_DEVICES=$FREE_GPU python -u "$PYTHON_SCRIPT" --index $index --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX > "${LOG_PATH}/${index}.log" 2>&1
        echo "Task $index on GPU $FREE_GPU finished."
    ) &

    # Wait for 60 seconds to give the GPU some time to allocate memory
    sleep 60
done

# Wait for all background jobs to finish
wait