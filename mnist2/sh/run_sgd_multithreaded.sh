#!/bin/bash
## Give the job a descriptive name
#PBS -N mnist_run_sgd_multithreaded

## Output and error files
#PBS -o mnist_run_sgd_multithreaded.out
#PBS -e mnist_run_sgd_multithreaded.err

## Limit memory, runtime etc.
#PBS -l walltime=00:30:00

cd $HOME/mnist
OPENBLAS_NUM_THREADS=1 taskset 0x1 python3 train_sgd_multithreaded.py
OPENBLAS_NUM_THREADS=2 taskset 0x3 python3 train_sgd_multithreaded.py
OPENBLAS_NUM_THREADS=4 taskset 0xF python3 train_sgd_multithreaded.py
OPENBLAS_NUM_THREADS=8 taskset 0xFF python3 train_sgd_multithreaded.py
