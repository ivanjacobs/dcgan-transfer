'''
	augment.py
	Justin Chen
	11.3.16

	Various image transformations for augmenting image datasets

	To view the transformation:
		1. Pass the function the path to the image
		   e.g. 'data/anime.jpg'
		2. Use show() to display image
		   e.g. horizontally_mirror('data/anime.jpg').show()

	Reference if you want to add more transformations or read up on the ones below:
	http://pillow.readthedocs.io/en/3.1.x/reference/ImageEnhance.html
	http://pillow.readthedocs.io/en/3.4.x/handbook/tutorial.html

	Boston University Computer Science
	Transfer Learning with DCGANS
'''

# flip the images horizontally
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import random

def horizontally_mirror(image_path):
	return Image.open(image_path).transpose(Image.FLIP_LEFT_RIGHT)

def vertically_mirror(image_path):
	return Image.open(image_path).transpose(Image.FLIP_TOP_BOTTOM)

def contrast(image_path):
	return ImageEnhance.Contrast(Image.open(image_path)).enhance(random.uniform(1, 10))

def brightness(image_path):
	return ImageEnhance.Brightness(Image.open(image_path)).enhance(random.uniform(.1, 1.5))

def gaussian_blur(image_path, radius=random.randint(0, 10)):
	return Image.open(image_path).filter(ImageFilter.GaussianBlur(radius))

def unsharp_mask(image_path, radius=random.randint(0, 10), percent=random.randint(0, 150), threshold=random.randint(0, 10)):
	return Image.open(image_path).filter(ImageFilter.UnsharpMask(radius, percent, threshold))

def rank_filter(image_path, size=5, rank=0):
	return Image.open(image_path).filter(ImageFilter.RankFilter(size, rank))

def median_filter(image_path, size=3):
	return Image.open(image_path).filter(ImageFilter.MedianFilter(size))

def min_filter(image_path, size=3):
	return Image.open(image_path).filter(ImageFilter.MinFilter(size))

def max_filter(image_path, size=3):
	return Image.open(image_path).filter(ImageFilter.MaxFilter(size))

def mode_filter(image_path, size=3):
	return Image.open(image_path).filter(ImageFilter.ModeFilter(size))
	
# Commented out b/c not sure how useful it'll be. This filter is overly sensitive when drawing edges.
#def kernel_filter(image_path, size=(3,3), kernel=[0,1,0, 1,-4,1, 0,1,0]):
#	return Image.open(image_path).filter(ImageFilter.Kernel(size, kernel))