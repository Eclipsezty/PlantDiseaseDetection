import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

tf.nn.relu(2)

def myRELU(x):
	return max(0.0, x)

def testMyrelu():
	x = 1.0
	print('Applying Relu on (%.1f) gives %.1f' % (x, myRELU(x)))
	print('Applying Relu on (%.1f) gives %.1f' % (x, tf.nn.relu(x)))

	x = -10.5
	print('Applying Relu on (%.1f) gives %.1f' % (x, myRELU(x)))
	print('Applying Relu on (%.1f) gives %.1f' % (x, tf.nn.relu(x)))

	x = 0.0
	print('Applying Relu on (%.1f) gives %.1f' % (x, myRELU(x)))
	print('Applying Relu on (%.1f) gives %.1f' % (x, tf.nn.relu(x)))

	x = 15.0
	print('Applying Relu on (%.1f) gives %.1f' % (x, myRELU(x)))
	print('Applying Relu on (%.1f) gives %.1f' % (x, tf.nn.relu(x)))

	x = -20.0
	print('Applying Relu on (%.1f) gives %.1f' % (x, myRELU(x)))
	print('Applying Relu on (%.1f) gives %.1f' % (x, tf.nn.relu(x)))


def myActivation(x1,a,b):

	# a should > 0
	if(x1>a/2+b):
		return 1
	if(x1>=b and x1<=(2/a + b)):
		return (a*(x1-b) - ((a*(x1-b))**2)/4) * 0.5 + 0.5
	if(x1>=((-2/a) + b) and x1<b):
		return (a*(x1-b) + ((a*(x1-b))**2)/4) * 0.5 + 0.5
	else:
	#if(x<(-a/2)+b):
		return 0

def draw():
	x = np.arange(-10, 15, 1)
	print(12**2)

	# plt.plot(x, [myActivation(i,2,4) for i in x], "b-",label="a=2,b=4")
	# plt.plot(x, [myActivation(i,2,5) for i in x], "r-",label="a=2,b=5")
	# plt.plot(x, [myActivation(i,2,6) for i in x], "y-",label="a=2,b=6")
	plt.plot(x, [myActivation(i,2,4) for i in x], "b-",label="a=2,b=4")
	plt.plot(x, [myActivation(i,4,4) for i in x], "r-",label="a=4,b=4")
	#plt.plot(x, [myActivation(i) for i in x], "y-",label="a=8,b=4")


	plt.title('myActivation')
	plt.xlabel("x")
	plt.ylabel("y")

	plt.xlim(-10, 15)  # limit x axis

	plt.legend()  # display label

	#plt.grid(1)  # display gridlines

	plt.show()
draw()

#
# class MYrelu(Activation):
#     def __init__(self, activation, **kwargs):
#         super(MYrelu, self).__init__(activation, **kwargs)
#         self.__name__ = 'myrelu'
#
#
# def myrelu(x=0.0):
#     if (x > 0):
#         return x
#     else:
#         return 0
#
#
# def m_relu(x: float):
#     x=float(x)
#     return tf.math.maximum(0, x)
#
#
# myrelu_np = np.vectorize(myrelu)
# myrelu_32 = lambda x: myrelu_np(x).astype(np.float32)
#
# #get_custom_objects().update({'myrelu': MYrelu(myrelu)})
#



def m_tanh(x):
    return 1 - 2 / (tf.math.exp(2 * x) + 1)


def m_sigmoid(x):
    return 1 / (1 + tf.math.exp(-x))
