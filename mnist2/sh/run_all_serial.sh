#!/bin/bash
## Give the job a descriptive name
#PBS -N mnist_run_all_serial

## Output and error files
#PBS -o mnist_run_all_serial.out
#PBS -e mnist_run_all_serial.err

## Limit memory, runtime etc.
#PBS -l walltime=00:60:00

cd $HOME/mnist
OPENBLAS_NUM_THREADS=1 taskset 0x1 python3 train_all_serial.py
