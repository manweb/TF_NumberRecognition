import numpy as np
import matplotlib.pyplot as plt

class display(object):
	def __init__(self):
		self._image = 0

	def displayNumber(self,data):
		# data should be an array of size 400 which is resized into a 20x20 matrix
		if (data.shape[0] != 400):
			return 0

		img = np.reshape(data, (20,20))

		plt.close('all')

		fig = plt.figure(figsize=(6, 3.2))
		plt.imshow(img, cmap='gray')

		plt.show(block=False)

