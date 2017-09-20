import tensorflow as tf
# import pandas as pd
import csv
import numpy as np
from mnist_model import MnistModel

train_size = 42000 #max 42k
test_size = 280 #max 28k
dim = 784

mm = MnistModel()
tf_train_dataset, tf_train_labels = mm.load_data(train_size, dim)

parameters = mm.initialize_parameters()
#X = tf.placeholder(tf.float32, shape=(784, None), name="X")
X = tf.placeholder(tf.float32, shape=(dim, None), name='X')
Y = tf.placeholder(tf.float32, shape=(10, None), name='Y')

Z3 = mm.forward_propagation(tf.transpose(tf_train_dataset), parameters)

#weights = tf.Variable(
#    tf.truncated_normal([28*28, 10]))
#biases = tf.Variable(tf.zeros([10]))

logits = tf.matmul(tf_train_dataset, weights) + biases

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=tf.transpose(Z3)))
#loss = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

train_prediction = tf.nn.softmax(logits)

num_epochs = 15
num_steps = 801
minibatch_size = 128

def accuracy(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(predictions),tf.argmax(labels))
    return tf.reduce_mean(tf.cast(correct_prediction,"float")).eval()

with tf.Session() as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases. 
    tf.global_variables_initializer().run()
    print('Initialized')

    #for epoch in range(num_epochs):
    #    epoch_cost = 0.
    #    num_minibatches = int(train_size / minibatch_size)
    #    minibatches = tf_train
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            #print(accuracy(predictions, tf_train_labels))
            print(accuracy(Z3, tf.transpose(tf_train_labels)))
            #print(accuracy(tf.transpose(train_prediction), tf.transpose(tf_train_labels)))
            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            # print('Validation accuracy: %.1f%%' % accuracy(
            #   valid_prediction.eval(), valid_labels))
            # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

