import tensorflow as tf
# import pandas as pd
import csv
import numpy as np
from mnist_model import MnistModel
from tensorflow.python.framework import ops

import matplotlib.pyplot as plt

data_size = 42000  # max 42k
test_size = 280  # max 28k
dim = 784
lambd = 0.01
# learning_rate = 0.001
mm = MnistModel()
train_dataset, train_labels = mm.load_csv_or_pickle("train.csv", "train.pickle", data_size, dim)


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


dev_size = 2000
test_size = 2000
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_data(train_dataset.T,
                                                            train_labels.T, dev_size, test_size)
train_size = data_size - dev_size - test_size
print(X_train.shape, X_dev.shape, X_test.shape)

layers_dims = [dim, 50, 30, 20, 10]
num_epochs = 20
minibatch_size = 128


# ----------






def model(X_train, Y_train, X_dev, Y_dev, learning_rate, lambd, layers_dims):
    ops.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(dim, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(10, None), name='Y')
    parameters = mm.initialize_parameters(layers_dims)
    ZL = mm.forward_propagation(X, parameters)
    train_prediction = tf.nn.softmax(ZL)
    loss = mm.loss(tf.transpose(Y), tf.transpose(ZL), lambd, parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Save the loss on training and dev set after each epoch, as well as the dev accuracy.
    loss_train = []
    loss_dev = []
    dev_accs = []

    with tf.Session() as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases. 
        tf.global_variables_initializer().run()
        print('Initialized')
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(train_size / minibatch_size)
            mini_batches = mm.random_mini_batches(X_train, Y_train, minibatch_size)
            for step, mini_batch in enumerate(mini_batches):
                # Run the computations. We tell .run() that we want to run the optimizer,
                # and get the loss value and the training predictions returned as numpy
                # arrays.

                _, l, predictions = session.run([optimizer, loss, train_prediction],
                                                feed_dict={X: mini_batch[0], Y: mini_batch[1]})
                # if (step % 100 == 0):
                #    print('Loss at step %d: %f' % (step, l))
                # Calculate the correct predictions
            loss_train.append(str(session.run(loss, {X: X_train, Y: Y_train})))
            loss_dev.append(str(session.run(loss, {X: X_dev, Y: Y_dev})))
            dev_accs.append(accuracy.eval({X: X_dev, Y: Y_dev}))
            print('----- epoch: {0} -----'.format(epoch+1))
            print('Loss train = ' + str(session.run(loss, {X: X_train, Y: Y_train})))
            print('Loss dev = ' + str(session.run(loss, {X: X_dev, Y: Y_dev})))
            print('Accuracy train ' + str(accuracy.eval({X: X_train, Y: Y_train})))
            print('Accuracy dev ' + str(accuracy.eval({X: X_dev, Y: Y_dev})))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        dev_accuracy = accuracy.eval({X: X_dev, Y: Y_dev})
        print(accuracy.eval({X: X_train, Y: Y_train}))
        print(accuracy.eval({X: X_dev, Y: Y_dev}))
        mm.plot_results(loss_train, loss_dev, dev_accs, learning_rate, train_accuracy, dev_accuracy)


for learning_rate in [0.001, 0.0015, 0.0021]:
    print("Learning rate is {}".format(learning_rate))
    model(X_train, Y_train, X_dev, Y_dev, learning_rate, lambd, layers_dims)
