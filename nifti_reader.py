import os
import numpy as np
import nibabel as nib
import csv
import pickle
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation
from nifti_nn import *
from pdb import *

def load_nifti(filename):
    folder = 'Atlases1.1/NAMIC_atlas/'
    filepath = os.path.join(folder, filename)
    img = nib.load(filepath)
    return np.asarray(img.get_data(), dtype=np.float32)

mask = load_nifti('atlas1_brainmask.nii')
t1 = load_nifti('atlas1_T1.nii')
t2 = load_nifti('atlas1_T2.nii')

t1_patches = image.extract_patches_2d(t1[:,:,100], (5, 5))
t2_patches = image.extract_patches_2d(t2[:,:,100], (5, 5))
mask_patches = image.extract_patches_2d(mask[:,:,100], (5, 5))

head_mask = np.ones(len(t1_patches), dtype=bool)
head_indexes = []
for i, patch in enumerate(t1_patches):
    if not patch.any():
        head_indexes.append(i)

head_mask[head_indexes] = False
t1_patches_train = t1_patches[head_mask,...]
t2_patches_train = t2_patches[head_mask,...]
mask_patches_train = mask_patches[head_mask,...]

t1_features = np.reshape(t1_patches_train, (t1_patches_train.shape[0], -1)).T
t2_features = np.reshape(t2_patches_train, (t2_patches_train.shape[0], -1)).T
X = np.concatenate((t1_features, t2_features), 0)
#X = np.reshape(t1_patches_train, (t1_patches_train.shape[0], -1)).T
Y = np.reshape(mask_patches_train, (mask_patches_train.shape[0], -1)).T


#model = load_model('25patchmodel')
#model = load_model('25patchmodel_T1T2')

X_train, Y_train, X_dev, Y_dev, X_test, Y_test = split_data(X, Y, 1000, 1000)
print('X_train: {}, Y_train: {}, X_dev: {}, Y_dev: {}, X_test: {}, Y_test: {}'.format(
    X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape, X_test.shape, Y_test.shape))

inputs = Input(shape=(X_train.shape[0],))
x = Dense(50, activation='relu')(inputs)
x = Dense(30, activation='relu')(x)
outputs = {}
for i in range(Y_train.shape[0]):
    outputs["y"+str(i)] = Dense(1, activation='sigmoid', name="y"+str(i))(x)
model = Model(inputs=inputs, outputs=list(outputs.values()))
model.compile(optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy'])

labels = [Y_train[i, :] for i in range(Y_train.shape[0])]
model.fit(X_train.T, labels, epochs=50, batch_size=128)  # starts training
model.save('25patchmodel_T1T2')
dev_labels = [Y_dev[i, :] for i in range(Y_train.shape[0])]
score = model.evaluate(X_dev.T, dev_labels, batch_size=128)
print(score)

t1_features = np.reshape(t1_patches, (t1_patches.shape[0], -1)).T
t2_features = np.reshape(t2_patches, (t2_patches.shape[0], -1)).T
input_image = np.concatenate((t1_features, t2_features), 0).T
#input_image = np.reshape(t1_patches, (t1_patches.shape[0], -1))
new_mask_patches = np.zeros(t1_patches.shape)
for i, patch in enumerate(input_image):
    output = model.predict(patch.reshape((1, 50)))
    output_restructured = np.asarray(np.squeeze(output))
    new_patch = np.round(np.reshape(output_restructured, (5,5)))
    new_mask_patches[i, :, :] += new_patch

new_mask = image.reconstruct_from_patches_2d(new_mask_patches, t1[:, :, 100].shape)
plt.figure(1)
plt.imshow(new_mask)
plt.figure(2)
plt.imshow(t1[:, :, 100])
plt.show()
set_trace()