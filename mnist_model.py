import tensorflow as tf
# import pandas as pd
import csv
import numpy as np

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

        tf_train_dataset = tf.cast(vals, tf.float32)
        tf_train_labels = tf.squeeze(tf.one_hot(labels, 10, axis=1))
        return tf_train_dataset, tf_train_labels

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