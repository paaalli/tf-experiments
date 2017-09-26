import tensorflow as tf
# import pandas as pd
import csv
import numpy as np
from mnist_model import MnistModel

data_size = 42000 #max 42k
test_size = 280 #max 28k
dim = 784

mm = MnistModel()
train_dataset, train_labels = mm.load_data(data_size, dim)

train_size= int(0.9*data_size)
def split_data(X, Y, train_size):
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    X_train = shuffled_X[:, :train_size]
    Y_train = shuffled_Y[:, :train_size]
    X_dev = shuffled_X[:, train_size+1:]
    Y_dev = shuffled_Y[:, train_size+1:]
    return X_train, Y_train, X_dev, Y_dev

X_train, Y_train, X_dev, Y_dev = split_data(train_dataset.T, train_labels.T, train_size)


parameters = mm.initialize_parameters([dim, 50, 20, 15, 10])
#X = tf.placeholder(tf.float32, shape=(784, None), name="X")
X = tf.placeholder(tf.float32, shape=(dim, None), name='X')
Y = tf.placeholder(tf.float32, shape=(10, None), name='Y')

ZL = mm.forward_propagation(X, parameters)
train_prediction = tf.nn.softmax(ZL)

#loss = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=tf.transpose(Y), logits=tf.transpose(ZL)))
loss = mm.loss(tf.transpose(Y), tf.transpose(ZL), 0.01, parameters)

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)



num_epochs = 30
minibatch_size = 128


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
            if (step % 100 == 0):
                print('Loss at step %d: %f' % (step, l))
            # Calculate the correct predictions

    correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy.eval({X: X_train, Y: Y_train}))
    print(accuracy.eval({X: X_dev, Y: Y_dev}))
