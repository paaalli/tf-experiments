from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from mnist_model import MnistModel
import numpy as np


data_size = 42000
dev_size = 1000
test_size = 1000
dim = 784
mm = MnistModel()
train_dataset, train_labels = mm.load_csv_or_pickle("train.csv", "train.pickle", data_size, dim)
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = mm.split_data(train_dataset.T,
                                                            train_labels.T, dev_size, test_size)

sequential = True
if sequential:
	model = Sequential()
	model.add(Dense(units=50, input_shape=(784,)))
	model.add(Activation('relu'))
	model.add(Dense(units=30))
	model.add(Activation('relu'))
	model.add(Dense(units=20))
	model.add(Activation('relu'))
	model.add(Dense(units=10))
	model.add(Activation('softmax'))

else: 
	inputs = Input(shape=(784,))

	# a layer instance is callable on a tensor, and returns a tensor
	x = Dense(50, activation='relu')(inputs)
	x = Dense(30, activation='relu')(x)
	x = Dense(20, activation='relu')(x)
	predictions = Dense(10, activation='softmax')(x)

	# This creates a model that includes
	# the Input layer and three Dense layers
	model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train.T, Y_train.T, epochs=30, batch_size=128)  # starts training

score = model.evaluate(X_dev.T, Y_dev.T, batch_size=128)





