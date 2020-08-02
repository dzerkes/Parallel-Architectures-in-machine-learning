#!/usr/bin/env python3

import numpy as np
import os
import time
import warnings
from multiprocessing import Manager
from multiprocessing import Process
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

manager = Manager()
exec_times = manager.dict()
jobs = []

X_train = np.random.uniform(low=0, high=255, size=(20000, 784))
y_train = np.random.randint(low=0, high=9, size=(20000,))

# Define a function that will be executed by every process
def train(exp_id, classifier, X_train, y_train, timings):
    tstart = time.time()
    classifier.fit(X_train, y_train)
    tend = time.time()
    timings[exp_id] = tend - tstart
    
# 1. Use different solvers
#
#    LBFGS solver
#
print("Exp. 1: using LBFGS solver...")
mlp1 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10, alpha=1e-4,
                    solver='lbfgs', tol=1e-4, random_state=1)
# Create process for this experiment
p1 = Process(target=train, args=(1, mlp1, X_train, y_train, exec_times))
jobs.append(p1)
#
#    ADAM solver
#
print("Exp. 2: using ADAM solver...")
mlp2 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10, alpha=1e-4,
                    solver='adam', tol=1e-4, random_state=1)
# Create process for this experiment
p2 = Process(target=train, args=(2, mlp2, X_train, y_train, exec_times))
jobs.append(p2)
#
#    SGD solver
#
print("Exp. 3: using SGD solver...")
mlp3 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1)
# Create process for this experiment
p3 = Process(target=train, args=(3, mlp3, X_train, y_train, exec_times))
jobs.append(p3)

# 2. Use different Neural Network architectures
#
#    More layers
#
print("Exp. 4: increasing depth...")
mlp4 = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=100, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1)
# Create process for this experiment
p4 = Process(target=train, args=(4, mlp4, X_train, y_train, exec_times))
jobs.append(p4)
#
#    Wider layers
#
print("Exp. 5: increasing width...")
mlp5 = MLPClassifier(hidden_layer_sizes=(150, 150), max_iter=100, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1)
# Create process for this experiment
p5 = Process(target=train, args=(5, mlp5, X_train, y_train, exec_times))
jobs.append(p5)

# 3. Use different mini-batch sizes for SGD solver
#
#    batch size = 64
#
print("Exp. 6: using SGD solver with a mini-batch of 64...")
mlp6 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1, batch_size=64)
# Create process for this experiment
p6 = Process(target=train, args=(6, mlp6, X_train, y_train, exec_times))
jobs.append(p6)
#
#    batch size = 128
#
print("Exp. 7: using SGD solver with a mini-batch of 128...")
mlp7 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1, batch_size=128)
# Create process for this experiment
p7 = Process(target=train, args=(7, mlp7, X_train, y_train, exec_times))
jobs.append(p7)
#
#    batch size = 256
#
print("Exp. 8: using SGD solver with a mini-batch of 256...")
mlp8 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=100, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1, batch_size=256)
# Create process for this experiment
p8 = Process(target=train, args=(8, mlp8, X_train, y_train, exec_times))
jobs.append(p8)

tstart = time.time()
# Launch processes and set their affinity
for i in range(len(jobs)):
    jobs[i].start()
    os.system("taskset -cp %d %d" % ((i % os.cpu_count()), jobs[i].pid))

# Wait until all processes are finished 
for proc in jobs:
    proc.join()
tend = time.time()
elapsed_time = tend - tstart

for exp, time in exec_times.items():
    print("Training time of Exp. %d (secs): %f" % (exp, time))

# Total training time of all experiments
print("Total training time (secs): %f" % elapsed_time)
