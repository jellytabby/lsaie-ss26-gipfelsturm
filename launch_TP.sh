#!/bin/bash

GPUS_PER_NODE=4

for MODEL_SIZE in 125m 350m 760m 1.5b 3b 8b; do
    for TP in 1 2 4; do
        echo "Launching throughput test for model size $MODEL_SIZE with TP=$TP"
        sed "s/    --tensor-model-parallel-size .*/    --tensor-model-parallel-size $TP/" launch.sh > launch_tmp.sh
        sed -i "s/JOB_NAME=\"gipfel-\${MODE}-\${MODEL_SIZE}-\${TRAINING_STEPS}s-\${NODES}n\"/JOB_NAME=\"gipfel-\${MODE}-\${MODEL_SIZE}-tp${TP}-\${NODES}n\"/" launch_tmp.sh
        chmod +x launch_tmp.sh
        ./launch_tmp.sh throughput $MODEL_SIZE 50 1
    done
done