#!/bin/bash
## Give the job a descriptive name
#PBS -N mnist_run_all_parallel_torque

## Output and error files
#PBS -o mnist_run_all_parallel_torque.out
#PBS -e mnist_run_all_parallel_torque.err

## Limit memory, runtime etc.
#PBS -l walltime=00:60:00

cd $HOME/mnist
export OPENBLAS_NUM_THREADS=1 
taskset 0x1 python3 train_lbfgs.py &
taskset 0x2 python3 train_adam.py &
taskset 0x4 python3 train_sgd.py &
taskset 0x8 python3 train_sgd_nn_depth.py &
taskset 0x10 python3 train_sgd_nn_width.py &
taskset 0x20 python3 train_sgd_batch_64.py &
taskset 0x40 python3 train_sgd_batch_128.py &
taskset 0x80 python3 train_sgd_batch_256.py &
wait