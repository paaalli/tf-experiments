import tensorflow as tf
import numpy as np
import math
import os
import matplotlib.pyplot as plt

def split_data(X, Y, dev_size, test_size):
    num_examples = X.shape[1]
    train_size = num_examples - test_size - dev_size
    permutation = list(np.random.permutation(num_examples))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    X_train = shuffled_X[:, :train_size]
    Y_train = shuffled_Y[:, :train_size]
    X_dev = shuffled_X[:, train_size:train_size + dev_size]
    Y_dev = shuffled_Y[:, train_size:train_size + dev_size]
    X_test = shuffled_X[:, train_size + dev_size:]
    Y_test = shuffled_Y[:, train_size + dev_size:]
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

def initialize_parameters(layers_dims):
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = tf.get_variable("W" + str(l), \
                                                   [layers_dims[l], layers_dims[l - 1]],
                                                   initializer=tf.contrib.layers.xavier_initializer())
        parameters["b" + str(l)] = tf.get_variable("b" + str(l), \
                                                   [layers_dims[l], 1],
                                                   initializer=tf.contrib.layers.xavier_initializer())

    return parameters

def forward_propagation(X, parameters):

    L = int(len(parameters) / 2)  # Number of layers in the Neural Network
    A = X
    for l in range(1, L):
        Z = tf.add(tf.matmul(parameters['W' + str(l)], A), parameters['b' + str(l)])
        A = tf.nn.relu(Z)
    Z = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])
    return Z

def random_mini_batches(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of training examples

    permutation = list(np.random.permutation(m))
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size - 1: -1]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size - 1: -1]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches