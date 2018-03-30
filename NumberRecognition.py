import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from utilities import ImportData
from displayData import display

dataFile = 'numberRecTrainData.txt'
labelFile = 'numberRecTrainLabel.txt'

data = ImportData(dataFile, labelFile)

data.shuffleData()
X = data.getData()
yConv = data.getConvLabels()

# split the data into traing, cross validation and test set
X_train = X[0:4000]
X_cv = X[4000:4500]
X_test = X[4500:5000]

y_train = yConv[0:4000]
y_cv = yConv[4000:4500]
y_test = yConv[4500:5000]

# plot some random rumbers
"""i = 0
while i < 10:
	dsp = display()
	dsp.displayNumber(random.choice(X))

	raw_input('Press enter to continue')

	i+=1
"""
# build the neural network
# the NN has 3 layers, 1 input, 1 hidden and 1 output layer
# layer sizes: 400 x 25 x 10

# input
x = tf.placeholder(tf.float32, shape=[None, 400])

# output
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# parameters and offsets
W0 = tf.get_variable("W0",shape=[400,25],initializer=tf.contrib.layers.xavier_initializer())
b0 = tf.Variable(tf.zeros([25]))
H0 = tf.nn.relu(tf.matmul(x,W0)+b0)

W1 = tf.get_variable("W1",shape=[25,10],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.zeros([10]))
y = tf.nn.relu(tf.matmul(H0,W1)+b1)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

fig = plt.figure(figsize=(10,5))

plot_x = []
plot_y = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	print 'Learning...'
	for i in range(10000):
		sess.run(train_step, feed_dict={x: X_train, y_: y_train})
		loss = sess.run(cross_entropy, feed_dict={x: X_train, y_: y_train})
		if i%100==0:
			print loss

		plot_x.append(i)
		plot_y.append(loss)
	print 'done'

	plt.plot(plot_x,plot_y)
	plt.show()

	i = 0
	dsp = display()
	while i < 10:
		index = random.randint(0,499)

		sample_x = X_test[index:index+1]
		sample_y = y_test[index:index+1]

		predicted = sess.run(y, feed_dict={x: sample_x, y_: sample_y})

		print("predicted: %i, true: %i"%(predicted.argmax(),sample_y.argmax()))

		dsp.displayNumber(X_test[index])

		raw_input('Press enter to continue')
		i += 1

