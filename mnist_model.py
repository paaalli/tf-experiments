import tensorflow as tf
# import pandas as pd
import csv
import numpy as np
import math

class MnistModel(object):
    def __init__(self):
        super(MnistModel, self).__init__()

    def load_data(self, train_size, dim):
        vals = np.zeros((train_size,dim))
        labels = np.zeros((train_size,1))
        with open("train.csv") as csvfile:
            reader = csv.DictReader(csvfile)
            for r, row in enumerate(reader):
                if (r >= train_size):
                    break
                v = np.zeros(dim)
                l = row['label']
                
                for i in range(0,dim):
                    v[i] = row["pixel" + str(i)]
                vals[r] = v
                labels[r] = l

        train_dataset = vals
        train_labels = self.one_hot_matrix(labels, 10)
        return train_dataset, train_labels

    def initialize_parameters(self, layers_dims):
        parameters = {}
        for l in range(1, len(layers_dims)):
            parameters["W" + str(l)] = tf.get_variable("W" + str(l), \
                [layers_dims[l], layers_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer())
            parameters["b" + str(l)] = tf.get_variable("b" + str(l), \
                [layers_dims[l], 1], initializer = tf.contrib.layers.xavier_initializer())

        return parameters

    def forward_propagation(self, X, parameters):

        L = int(len(parameters) / 2) # Number of layers in the Neural Network
        A = X
        for l in range(1, L):
            Z = tf.add(tf.matmul(parameters['W' + str(l)], A), parameters['b' + str(l)])
            A = tf.nn.relu(Z)
        Z = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])
        return Z

    def one_hot_matrix(self, labels, C):
        C = tf.constant(C, name='C')
        one_hot_matrix = tf.squeeze(tf.one_hot(labels, C, axis=1))
        sess = tf.Session()
        one_hot = sess.run(one_hot_matrix)
        sess.close()

        return one_hot

    def random_mini_batches(self, X, Y, mini_batch_size = 64):
        """
        Creates a list of random minibatches from (X, Y)
        
        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer
        
        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        m = X.shape[1]                  # number of training examples
        
        permutation = list(np.random.permutation(m))
        mini_batches = []
            
        # Step 1: Shuffle (X, Y)
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k*mini_batch_size: (k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size: (k+1)*mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size -1 : -1]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size - 1: -1]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches