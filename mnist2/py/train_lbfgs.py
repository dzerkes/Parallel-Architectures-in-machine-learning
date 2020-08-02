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
    return elapsed_time

#
#    LBFGS solver
#
print("Exp. 1: using LBFGS solver...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10, alpha=1e-4,
                    solver='lbfgs', tol=1e-4, random_state=1)
elapsed_time = train(mlp, X_train, y_train)
print("Training time (secs): %f" % elapsed_time)
