import numpy as numpy
from skimage import img_as_float32, img_as_ubyte, io
import time

import sef

alpha = 4
beta = 0.5

img = img_as_float32(io.imread('data/input_cathedral.png'))
# img = img_as_float32(io.imread('data/input_horse.png'))

start = time.time()
img_pred = sef.simulated_exposure_fusion(img)
print('Done in %2.2f seconds' % (time.time() - start))

io.imsave('res.png', img_as_ubyte(img_pred))
