#!/bin/bash
#
# sweep_mbs.sh — queue 760m × {offload, nooffload} × {MBS 6..10} throughput jobs
#
# Submits 10 SLURM jobs (50 steps, 1 node each) to measure tokens/sec/GPU
# across micro-batch sizes with and without CPU activation offloading.
#
# Usage: ./sweep_mbs.sh

set -euo pipefail

MODEL=760m
STEPS=50
NODES=1

for OFFLOAD_FLAG in --offload --no-offload; do
    for MBS in 6 10 12 15; do
        echo "Queuing: model=$MODEL mbs=$MBS $OFFLOAD_FLAG"
        ./launch.sh throughput "$MODEL" "$STEPS" "$NODES" "--mbs=$MBS" "$OFFLOAD_FLAG"
    done
done
