#!/bin/bash

PYTHON_SCRIPT="./Experiments/gcg_exp.py"
MODEL_PATH="meta-llama/Llama-2-13b-chat-hf"
ADD_EOS=True
RUN_INDEX=0
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

# Function to find the first available GPU(s)
find_free_gpus() {
    if [[ "$MODEL_PATH" == *"13B"* ]]; then
        local required_free_mem=120000 # Assuming 120GB for 2 GPUs
        local num_gpus_needed=2
    else
        local required_free_mem=60000 # Assuming 60GB for 1 GPU
        local num_gpus_needed=1
    fi

    local available_gpus=()

    for i in {0..7}; do
        local free_mem=$(nvidia-smi -i $i --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1}')
        if [ "$free_mem" -ge $(($required_free_mem / $num_gpus_needed)) ]; then
            available_gpus+=($i)
            if [ "${#available_gpus[@]}" -eq "$num_gpus_needed" ]; then
                echo "${available_gpus[@]}"
                return
            fi
        fi
    done

    echo "-1" # Return -1 if no suitable GPU(s) is/are found
}

# Start the jobs with GPU assignment
for index in {0..127}; do

    FREE_GPUS="-1"

    # Keep looping until free GPU(s) is/are found
    while [ "$FREE_GPUS" == "-1" ]; do
        FREE_GPUS=$(find_free_gpus)
        if [ "$FREE_GPUS" == "-1" ]; then
            sleep 5 # Wait for 5 seconds before trying to find free GPU(s) again
        fi
    done

    # Run the Python script on the free GPU(s)
    (
        CUDA_VISIBLE_DEVICES=$FREE_GPUS python -u "$PYTHON_SCRIPT" --index $index --model_path $MODEL_PATH $ADD_EOS_FLAG --run_index $RUN_INDEX > "${LOG_PATH}/${index}.log" 2>&1
        echo "Task $index on GPU(s) $FREE_GPUS finished."
    ) &

    # Wait for 30 seconds to give the GPU(s) some time to allocate memory
    sleep 30
done

# Wait for all background jobs to finish
wait
