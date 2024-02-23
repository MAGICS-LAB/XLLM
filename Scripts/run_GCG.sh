#!/bin/bash

PYTHON_SCRIPT="./Experiments/gcg_exp.py"
MODEL_PATH="mosaicml/mpt-7b-chat"
ADD_EOS=True
# Check if ADD_EOS is true
if [[ "${ADD_EOS,,}" == "true" ]]; then
    LOG_PATH="Logs/${MODEL_PATH}/GCG_eos"
else
    LOG_PATH="Logs/${MODEL_PATH}/GCG"
fi

# Create the log directory if it does not exist
mkdir -p "$LOG_PATH"

# Function to find the first available GPU
find_free_gpu() {
    for i in {4..7}; do
        if nvidia-smi -i $i | grep 'No running processes found' > /dev/null; then
            echo $i
            return
        fi
    done

    echo "-1" # Return -1 if no free GPU is found
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
        CUDA_VISIBLE_DEVICES=$FREE_GPU python -u "$PYTHON_SCRIPT" --index $index --model_path $MODEL_PATH --add_eos $ADD_EOS > "${LOG_PATH}/${index}.log" 2>&1
        echo "Task $index on GPU $FREE_GPU finished."
    ) &

    # Wait for 30 seconds to give the GPU some time to allocate memory
    sleep 30
done

# Wait for all background jobs to finish
wait
