import numpy as np
import tensorflow as tf

class ImportData(object):
	def __init__(self, data, labels):
		print 'data file: '+data
		print 'label file: '+labels

		self._X = np.loadtxt(open(data), delimiter='\t', dtype=None)
		self._y = np.loadtxt(open(labels), delimiter='\t', dtype=None)
		self._yConv = self.convertLabels()

		if (self._X.shape[0] != self._y.shape[0]):
			print 'Length of data and labels do not match!'
		else:
			print 'Loading of data and labels successful! data: %ix%i labels: %ix1' % (self._X.shape[0],self._X.shape[1],self._y.shape[0])

	def convertLabels(self):
		convLabels = np.zeros((self._X.shape[0],10))

		for i in range(0, self._y.shape[0]):
			convLabels[i] = self.convertLabel(self._y[i], 10)

		return convLabels

	def convertLabel(self, label, m):
		if label == 10:
			label = 0
		lb = np.zeros(m)
		lb[label] = 1

		return lb

	def getData(self):
		return self._X

	def getLabels(self):
		return self._y

	def getConvLabels(self):
		return self._yConv

	def shuffleData(self):
		perm = np.arange(self._X.shape[0])
		np.random.shuffle(perm)

		self._X = self._X[perm]
		self._y = self._y[perm]
		self._yConv = self._yConv[perm]

