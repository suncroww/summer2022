from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import numpy as np
 
# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]
 
# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Output', 'Input']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
		# pyplot.subplot(111)
		# pyplot.axis('off')
		# pyplot.imshow(images[i])
		# pyplot.savefig(titles[i] + "_e10.png",bbox_inches='tight',pad_inches=0)
		# np.save(titles[i] + "_e1.npy", images[i])
	pyplot.show()

def plot_some_images(src_img, gen_img):
	images = vstack((src_img, gen_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Output']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 2, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
		# pyplot.subplot(111)
		# pyplot.axis('off')
		# pyplot.imshow(images[i])
		# pyplot.savefig(titles[i] + "_e10.png",bbox_inches='tight',pad_inches=0)
		# np.save(titles[i] + "_e1.npy", images[i])
		# pyplot.savefig("testing_drnz.png",bbox_inches='tight',pad_inches=.3)
	pyplot.show()
 
# load dataset
[X1, X2] = load_real_samples('drnz.npz')
print('Loaded', X1.shape, X2.shape)
# load model
model = load_model('testing_001000.h5')

# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

# import numpy as np
# from PIL import Image, ImageOps
# image = (Image.open('scanme.png')).transpose(Image.FLIP_TOP_BOTTOM)
# src_image = np.array(image).astype(float)
# src_image = np.load("noisyderenzo25.png")
# src_image = np.load("McDisplayDerenzo.npy")

# generate image from source
gen_image = model.predict(src_image)
# plot all three images
# plot_images(src_image, gen_image, tar_image)
# plot two!
plot_some_images(src_image, gen_image)