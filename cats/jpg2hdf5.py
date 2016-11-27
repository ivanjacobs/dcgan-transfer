from scipy import misc
import os
import numpy as np
import h5py

if not os.path.exists("hdf5"):
	os.makedirs("hdf5")

directory = './rex_cat/'
images = [misc.imread(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and (f.endswith('.jpg') or f.endswith('.jpeg'))]

with h5py.File('cats_11230_128px.hdf5', 'w') as hf:
	hf.create_dataset('dataset_1', data=np.asarray(images))