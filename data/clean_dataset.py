'''
	clean_dataset.py
	Justin Chen
	11.2.16

	Use Google's InceptionNet to determine if images are cats to clean out dataset.
	#selfloops #GEB

	Boston University Computer Science
	Transfer Learning with DCGANS
'''

from PIL import Image
import classify_image as inception
from os import remove
from os import listdir
from os.path import isfile, join
import shutil
import sys
import augment as aug
import types
import random

image_dir = 'data/cat'
images    = [f for f in listdir(image_dir) if isfile(join(image_dir, f)) and not f.startswith('.') and ((join(image_dir, f).endswith('.jpg') or join(image_dir, f).endswith('.jpeg')))] 

for i, img in enumerate(images):
	file_path = join(image_dir, img)
	description = ''

	with open(file_path, 'rb') as f: 
		description, score = inception.run_inference_on_image(f.read())

	if 'cat' not in description:
		remove(file_path)
	else:
		shutil.move(file_path, join(image_dir, 'checked'))

print 'done cleaning dataset'