#!/usr/bin/env python3

import time
import warnings
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Measure total execution time
total_tstart = time.time()

# Load MNIST dataset from https://www.openml.org/d/554
#
# The MNIST database of handwritten digits with 784 features,
# raw data available at: http://yann.lecun.com/exdb/mnist/.
# It can be split in a training set of the first 60,000 examples,
# and a test set of 10,000 examples
#
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Rescale the data, use the suggested train/test split
X = X / 255.
X_train = X[:60000]
y_train = y[:60000]

# Define a function that will be used for every experiment
def train(classifier, X_train, y_train):
    tstart = time.time()
    classifier.fit(X_train, y_train)
    tend = time.time()
    elapsed_time = tend - tstart
    return elapsed_time

#
#    epochs = 30, batch size = 256
#
print("Running SGD solver with a mini-batch of 256 for 30 epochs...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=30, alpha=1e-4,
                    solver='sgd', tol=1e-4, random_state=1, batch_size=256)
training_time = train(mlp, X_train, y_train)
total_tend = time.time()
print("Training time (secs): %f" % training_time)
print("Total time (secs): %f" % (total_tend - total_tstart))
