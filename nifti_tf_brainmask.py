import os
import numpy as np
import nibabel as nib
import csv
import pickle
from sklearn.feature_extraction import image
import tensorflow as tf
from tensorflow.python.framework import ops
from nifti_nn import *


def load_nifti(filename):
    folder = 'Atlases1.1/NAMIC_atlas/'
    filepath = os.path.join(folder, filename)
    img = nib.load(filepath)
    return np.asarray(img.get_data(), dtype=np.float32)

mask = load_nifti('atlas1_brainmask.nii')
t1 = load_nifti('atlas1_T1.nii')

t1_patches = image.extract_patches_2d(t1[:,:,100], (5, 5))
mask_patches = image.extract_patches_2d(mask[:,:,100], (5, 5))

head_mask = np.ones(len(t1_patches), dtype=bool)
head_indexes = []
for i, patch in enumerate(t1_patches):
    if not patch.any():
        head_indexes.append(i)

head_mask[head_indexes] = False
t1_patches = t1_patches[head_mask,...]
mask_patches = mask_patches[head_mask,...]

X = np.reshape(t1_patches, (t1_patches.shape[0], -1)).T
Y = np.reshape(mask_patches, (mask_patches.shape[0], -1)).T

X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_data(X, Y, 1000, 1000)

print('X_train: {}, Y_train: {}, X_dev: {}, Y_dev: {}, X_test: {}, Y_test: {}'.format(
    X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape, X_test.shape, Y_test.shape))

layers_dims = [25, 50, 40, 30, 25]
num_epochs = 20
minibatch_size = 128
learning_rate = 0.001
train_size = X_train.shape[1]

X = tf.placeholder(tf.float32, shape=(X_train.shape[0], None), name='X')
Y = tf.placeholder(tf.float32, shape=(Y_train.shape[0], None), name='Y')
parameters = initialize_parameters(layers_dims)
ZL = forward_propagation(X, parameters)
train_prediction = tf.nn.sigmoid(ZL)
#loss = mm.loss(tf.transpose(Y), tf.transpose(ZL), lambd, parameters)

loss = tf.reduce_mean(tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.transpose(Y), logits=tf.transpose(ZL)), axis=1))  # Multi task learning


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

np.set_printoptions(threshold=np.nan)
prediction = tf.cast(train_prediction > 0.5, tf.float32)
correct_prediction = tf.equal(tf.transpose(prediction), tf.transpose(Y))

true_positives = tf.equal(tf.cast(correct_prediction, tf.float32), tf.transpose(Y))
false_positives = tf.equal(tf.transpose(prediction), tf.cast(tf.transpose(Y) == 0, tf.float32))
precision = tf.reduce_mean(tf.cast(tf.reshape(true_positives, [-1]), tf.float32)) \
                / (tf.reduce_mean(tf.cast(tf.reshape(false_positives, [-1]), tf.float32))
                    + tf.reduce_mean(tf.cast(tf.reshape(true_positives, [-1]), tf.float32)))
                
accuracy = tf.reduce_mean(tf.cast(tf.reshape(correct_prediction, [-1]), tf.float32))#"float"))

# Save the loss on training and dev set after each epoch, as well as the dev accuracy.
loss_train = []
loss_dev = []
with tf.Session() as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases. 
    tf.global_variables_initializer().run()
    print('Initialized')
    for epoch in range(num_epochs):
        num_minibatches = int(train_size / minibatch_size)
        mini_batches = random_mini_batches(X_train, Y_train, minibatch_size)
        for step, mini_batch in enumerate(mini_batches):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.

            _, l, predictions = session.run([optimizer, loss, train_prediction],
                                            feed_dict={X: mini_batch[0], Y: mini_batch[1]})

        loss_train.append(str(session.run(loss, {X: X_train, Y: Y_train})))
        loss_dev.append(str(session.run(loss, {X: X_dev, Y: Y_dev})))
        print('----- epoch: {0} -----'.format(epoch + 1))
        print('Loss train = ' + str(session.run(loss, {X: X_train, Y: Y_train})))
        print('Loss dev = ' + str(session.run(loss, {X: X_dev, Y: Y_dev})))
        print('Accuracy train ' + str(accuracy.eval({X: X_train, Y: Y_train})))
        print('Accuracy dev ' + str(accuracy.eval({X: X_dev, Y: Y_dev})))
        print('False positives dev ' + str(false_positives.eval({X: X_dev, Y: Y_dev})))


