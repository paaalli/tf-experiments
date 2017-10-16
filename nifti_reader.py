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

