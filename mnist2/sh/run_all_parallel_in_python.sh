#!/bin/bash
## Give the job a descriptive name
#PBS -N mnist_run_all_parallel_in_python

## Output and error files
#PBS -o mnist_run_all_parallel_in_python.out
#PBS -e mnist_run_all_parallel_in_python.err

## Limit memory, runtime etc.
#PBS -l walltime=00:30:00

cd $HOME/mnist
OPENBLAS_NUM_THREADS=1 python3 train_all_parallel.py
