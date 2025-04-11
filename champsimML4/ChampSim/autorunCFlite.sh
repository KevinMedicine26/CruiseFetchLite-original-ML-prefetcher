#!/bin/bash

# CruiseFetchPro Automated Training and Testing Script
# This script automates the process of training, generating prefetches, building, and testing for CruiseFetchPro
# 
# INSTRUCTIONS: Edit the variables below to customize your training and testing process

#------------------------------------
# CONFIGURABLE PATHS AND PARAMETERS
#------------------------------------

# Trace files
TRACE_TRAIN="./traces/605.mcf-s0.txt.xz"     # Trace file for training
TRACE_GENERATE="./traces/605.mcf-s1.txt.xz"  # Trace file for generating prefetches
TRACE_TEST="./traces/605.mcf-s1.trace.xz"    # Trace file for testing

# Model and prefetch files
MODEL="./prefetch_files/model605-cruisefetchpro-sen-train0warm-605s0"  # Path to save/load model
PREFETCH_FILE="./prefetch_files/prefetches_605s1-cruisefetchpro-sen-train0warm-605s1.txt"  # Path for generated prefetches

# Parameters
WARMUP_TRAIN=10     # Number of warmup instructions for training
WARMUP_GENERATE=10   # Number of warmup instructions for generating
WARMUP_TEST=0      # Number of warmup instructions for testing
USE_NO_BASE=true    # Whether to use --no-base option in testing (true/false)
SKIP_TRAINING=false # Whether to skip the training step (true/false)

#------------------------------------
# UTILITY FUNCTIONS - DO NOT MODIFY
#------------------------------------

# Function to clear memory caches
clear_memory() {
    echo "Clearing memory cache..."
    sync                       # Synchronize cached writes to persistent storage
    echo 3 > /proc/sys/vm/drop_caches   # Clear pagecache, dentries and inodes
    echo "Memory cache cleared."
}

# Function to wait with a countdown
countdown() {
    local seconds=$1
    echo "Waiting $seconds seconds before next step..."
    for (( i=$seconds; i>0; i-- )); do
        echo -ne "$i...\r"
        sleep 1
    done
    echo -e "\nContinuing to next step."
}

# Check if a command succeeded
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: Previous command failed with exit code $?. Stopping execution."
        exit 1
    else
        echo "Success: Command completed successfully."
    fi
}

# Prepare no-base option for testing
NO_BASE_OPTION=""
if [ "$USE_NO_BASE" = true ]; then
    NO_BASE_OPTION="--no-base"
    echo "Using --no-base option for testing"
else
    echo "Not using --no-base option for testing"
fi

#------------------------------------
# MAIN EXECUTION
#------------------------------------

if [ "$SKIP_TRAINING" = true ]; then
    echo "===== SKIPPING STEP 1: TRAINING MODEL ====="
    echo "Training has been skipped as per configuration. Make sure the model file exists at: $MODEL"
else
    echo "===== STEP 1: TRAINING MODEL ====="
    echo "Running: python3 ml_prefetch_sim.py train $TRACE_TRAIN --model $MODEL --num-prefetch-warmup-instructions $WARMUP_TRAIN"
    python3 ml_prefetch_sim.py train $TRACE_TRAIN --model $MODEL --num-prefetch-warmup-instructions $WARMUP_TRAIN
    check_success
    clear_memory
    countdown 10
fi

echo "===== STEP 2: GENERATING PREFETCHES ====="
echo "Running: python3 ml_prefetch_sim.py generate $TRACE_GENERATE $PREFETCH_FILE --model $MODEL --num-prefetch-warmup-instructions $WARMUP_GENERATE"
python3 ml_prefetch_sim.py generate $TRACE_GENERATE $PREFETCH_FILE --model $MODEL --num-prefetch-warmup-instructions $WARMUP_GENERATE
check_success
clear_memory
countdown 10

echo "===== STEP 3: BUILDING ====="
echo "Running: python3 ml_prefetch_sim.py build"
python3 ml_prefetch_sim.py build
check_success
clear_memory
countdown 10

echo "===== STEP 4: TESTING ====="
echo "Running: python3 ml_prefetch_sim.py run $TRACE_TEST --prefetch $PREFETCH_FILE --num-prefetch-warmup-instructions $WARMUP_TEST $NO_BASE_OPTION"
python3 ml_prefetch_sim.py run $TRACE_TEST --prefetch $PREFETCH_FILE --num-prefetch-warmup-instructions $WARMUP_TEST $NO_BASE_OPTION
check_success

#echo "===== STEP 5: MODEL EVALUATION ====="
#echo "Running: python3 ml_prefetch_sim.py eval --results-dir ./results"
#python3 ml_prefetch_sim.py eval --results-dir ./results
#check_success

echo "===== ALL STEPS COMPLETED SUCCESSFULLY ====="
echo "Training and evaluation results for CruiseFetchPro"
echo "Model saved to: $MODEL"
echo "Prefetches generated to: $PREFETCH_FILE"