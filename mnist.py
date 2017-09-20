import tensorflow as tf
# import pandas as pd
import csv
import numpy as np

train_size = 42000 #42k
test_size = 28000 #28k
dim = 784

# Read train set
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

# Read test set
vals = np.zeros((train_size,dim))
with open("test.csv") as csvfile:
     reader = csv.DictReader(csvfile)
     for r, row in enumerate(reader):
		if (r >= train_size):
			break
		v = np.zeros(dim)
		for i in range(0,dim):
			v[i] = row["pixel" + str(i)]
		vals[r] = v


tf_test_dataset = tf.cast(vals, tf.float32)

print(tf.squeeze(tf_train_labels).shape)

print(tf_train_dataset.shape)
print(tf_train_labels.shape)

weights = tf.Variable(
	tf.truncated_normal([28*28, 10]))
biases = tf.Variable(tf.zeros([10]))

logits = tf.matmul(tf_train_dataset, weights) + biases

print(logits.shape)
print(tf_train_labels)
loss = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

train_prediction = tf.nn.softmax(logits)
test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 801

def accuracy(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(predictions),tf.argmax(labels))
    return tf.reduce_mean(tf.cast(correct_prediction,"float")).eval()

with tf.Session() as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print(accuracy(predictions, tf_train_labels))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      # print('Validation accuracy: %.1f%%' % accuracy(
      #   valid_prediction.eval(), valid_labels))
  # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
  print(test_prediction.eval())
