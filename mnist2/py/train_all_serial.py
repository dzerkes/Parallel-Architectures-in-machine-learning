#!/usr/bin/env python3

import numpy as np
import time
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

X_train = np.random.uniform(low=0, high=255, size=(20000, 784))
y_train = np.random.randint(low=0, high=9, size=(20000,))

# Define a function that will be used for every experiment
def train(classifier, X_train, y_train):
    tstart = time.time()
    classifier.fit(X_train, y_train)
    tend = time.time()
    elapsed_time = tend - tstart
    print("Training time (secs): %f" % elapsed_time)
    return elapsed_time

elapsed_time = 0.0

# 1. Use different solvers
#
#    LBFGS solver
#
print("Exp. 1: using LBFGS solver...")
mlp1 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10, alpha=1e-4,
                     solver='lbfgs', tol=1e-4, random_state=1)
elapsed_time += train(mlp1, X_train, y_train)
#
#    ADAM solver
#
print("Exp. 2: using ADAM solver...")
mlp2 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10, alpha=1e-4,
                     solver='adam', tol=1e-4, random_state=1)
elapsed_time += train(mlp2, X_train, y_train)
#
#    SGD solver
#
print("Exp. 3: using SGD solver...")
mlp3 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10, alpha=1e-4,
                     solver='sgd', tol=1e-4, random_state=1)
elapsed_time += train(mlp3, X_train, y_train)

# 2. Use different Neural Network architectures
#
#    More layers
#
print("Exp. 4: increasing depth...")
mlp4 = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=10, alpha=1e-4,
                     solver='sgd', tol=1e-4, random_state=1)
elapsed_time += train(mlp4, X_train, y_train)
#
#    Wider layers
#
print("Exp. 5: increasing width...")
mlp5 = MLPClassifier(hidden_layer_sizes=(150, 150), max_iter=10, alpha=1e-4,
                     solver='sgd', tol=1e-4, random_state=1)
elapsed_time += train(mlp5, X_train, y_train)

# 3. Use different mini-batch sizes for SGD solver
#
#    batch size = 64
#
print("Exp. 6: using SGD solver with a mini-batch of 64...")
mlp6 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10, alpha=1e-4,
                     solver='sgd', tol=1e-4, random_state=1, batch_size=64)
elapsed_time += train(mlp6, X_train, y_train)
#
#    batch size = 128
#
print("Exp. 7: using SGD solver with a mini-batch of 128...")
mlp7 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10, alpha=1e-4,
                     solver='sgd', tol=1e-4, random_state=1, batch_size=128)
elapsed_time += train(mlp7, X_train, y_train)
#
#    batch size = 256
#
print("Exp. 8: using SGD solver with a mini-batch of 256...")
mlp8 = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10, alpha=1e-4,
                     solver='sgd', tol=1e-4, random_state=1, batch_size=256)
elapsed_time += train(mlp8, X_train, y_train)

# Total training time of all experiments
print("Total training time (secs): %f" % elapsed_time)
