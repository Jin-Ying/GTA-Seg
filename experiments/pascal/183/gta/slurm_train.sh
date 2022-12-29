#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
job='183_gta'
ROOT=../../../../
mkdir -p log

# use slurm
srun --mpi=pmi2 -p $3 -n$1 --gres=gpu:$1 --ntasks-per-node=$1 --job-name=$job \
    python -u $ROOT/train_gta.py --config=config.yaml --seed 2 --port $2 2>&1 | tee log/seg_183_gta.txt
