import tensorflow as tf
# import pandas as pd
import csv
import numpy as np
from mnist_model import MnistModel
from tensorflow.python.framework import ops

data_size = 42000 #max 42k
test_size = 280 #max 28k
dim = 784
lambd = 0.01
#learning_rate = 0.001
mm = MnistModel()
data_X, data_Y = mm.load_csv_or_pickle("train.csv","train.pickle",data_size, dim)
print(type(data_X), type(data_Y))
print("data_set",data_X, data_X.shape)
print("data_set", data_Y, data_Y.shape)



dev_size = 2000
test_size = 2000
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = mm.split_data(data_X,
                                                            data_Y, dev_size, test_size)
train_size = data_size - dev_size - test_size
print(X_train.shape,X_dev.shape, X_test.shape)

layers_dims = [dim, 50, 30, 20, 10]
num_epochs = 20
minibatch_size = 128
#----------






def model(X_train, Y_train, X_dev, Y_dev, learning_rate, lambd, layers_dims):

    ops.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(dim, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(10, None), name='Y')
    parameters = mm.initialize_parameters(layers_dims)
    ZL = mm.forward_propagation(X, parameters)
    train_prediction = tf.nn.softmax(ZL)
    loss = mm.loss(tf.transpose(Y), tf.transpose(ZL), lambd, parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


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

                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict={X: mini_batch[0], Y: mini_batch[1]})
                #if (step % 100 == 0):
                #    print('Loss at step %d: %f' % (step, l))
                # Calculate the correct predictions

        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy.eval({X: X_train, Y: Y_train}))
        print(accuracy.eval({X: X_dev, Y: Y_dev}))


for learning_rate in [0.001, 0.0015, 0.0021]:
    print("Learning rate is {}".format(learning_rate))
    model(X_train, Y_train, X_dev, Y_dev, learning_rate, lambd, layers_dims)
