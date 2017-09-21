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

    def initialize_parameters(self):
        W1 = tf.get_variable("W1", [25,784], initializer = tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
        W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
        W3 = tf.get_variable("W3", [10,12], initializer = tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3", [10,1], initializer = tf.zeros_initializer())

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        
        return parameters

    def forward_propagation(self, X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']

        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        return Z3

    def one_hot_matrix(self, labels, C):
        C = tf.constant(C, name='C')
        
        # Use tf.one_hot, be careful with the axis (approx. 1 line)
        one_hot_matrix = tf.squeeze(tf.one_hot(labels, C, axis=1))

        
        # Create the session (approx. 1 line)
        sess = tf.Session()
        
        # Run the session (approx. 1 line)
        one_hot = sess.run(one_hot_matrix)
        
        # Close the session (approx. 1 line). See method 1 above.
        sess.close()
        
        ### END CODE HERE ###
        
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
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:, k*mini_batch_size: (k+1)*mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size: (k+1)*mini_batch_size]
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size -1 : -1]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size - 1: -1]
            ### END CODE HERE ###
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches