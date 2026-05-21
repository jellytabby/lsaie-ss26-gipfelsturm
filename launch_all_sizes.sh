#!/bin/bash
for MODEL_SIZE in 125m 350m 760m 1.5b 3b 8b; do
    ./launch.sh throughput $MODEL_SIZE 50 1
    echo "Launching throughput test for model size $MODEL_SIZE"
done