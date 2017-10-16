from keras.models import Sequential
from keras.layers import Dense, Activation
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
model = Sequential()
model.add(Dense(units=50, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(units=30))
model.add(Activation('relu'))
model.add(Dense(units=20))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train.T, Y_train.T, epochs=30, batch_size=128)
score = model.evaluate(X_dev.T, Y_dev.T, batch_size=128)