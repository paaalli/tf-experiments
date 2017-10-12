import os
import numpy as np
import nibabel as nib
import csv
import pickle
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import pdb
def load_nifti(filename):
    folder = 'Atlases1.1/NAMIC_atlas/'
    filepath = os.path.join(folder, filename)
    img = nib.load(filepath)
    return np.asarray(img.get_data(), dtype=np.float32)

mask = load_nifti('atlas1_brainmask.nii')
t1 = load_nifti('atlas1_T1.nii')

#plt.figure(1)
#plt.imshow(t1[:,:,100])
#plt.figure(2)
#plt.imshow(mask[:,:,100])
#plt.show()
t1_patches = image.extract_patches_2d(t1[:,:,100], (15, 15))
mask_patches = image.extract_patches_2d(mask[:,:,100], (15, 15))

print(mask_patches[150])

for patch in t1_patches:
    if patch.any():
        print(patch)
#Er málið að vera með multi class labeling þannig að patch í mynd
#samsvari patch í mask? Svo væri hægt að hafa vote til þess
#að bæta accuracy enn frekar.

