import math
import numpy as np


class LogisticRegression(dim):

	def __init__(self):
		W = np.zeros((dim, 1))
		b = 0
		assert (W.shape == (dim, 1))
		assert (isinstance(b, float) or isinstance(b, int))

	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))

	# def initialize(self, dim):
		

	def propagate(self, W, b, x, y):
		# num of samples
		m = x.shape[1]

		# Forward prop
		A = self.sigmoid(np.dot(W.T, x) + b)
		Loss = -1 / m * np.sum(y*np.log(A) + (1-y)*np.log(1-A))

		# Backward prop
		dA = A - y
		dW = (1/m) * np.dot(x, dA.T)
		db = (1/m) * np.sum(dA)

		loss = np.squeeze(loss)	#从数组的形状中删除单维度条目，即把shape中为1的维度去掉
		grads = {'dw': dW, 'db': db}

		return grads, loss

	def optimize(self, W, b, x, y, num_iterations, learning_rate, print_loss):
		losses = []

		for i in range(num_iterations):
			# Generate grads
			grads, loss = self.propagate(W, b, x, y)
			dW = grads['dW']
			db = grads['db']

			# Update params
			W = W - dW * learning_rate
			b = b - db * learning_rate

			# Record losses
			losses.append(loss)

			if print_loss and i % 10 == 0:
				print('Loss after iteration %i: %f' % (i, loss))
		print('Training finished!!!')

		params = {'W': w, 'b': b}
		grads = {'dw': dW, 'db': db}

		return params, grads, losses

	def predict(self, W, b, x):
		# num of samples
		m = x.shape[1]
		pred = np.zeros((1, m))

		A = self.sigmoid(np.dot(W.T, x)+b)
		for i in range(A.shape[1]):
			# Convert probabilities A[0,i] to actual predictions p[0,i]
			if A[i, 1] >= 0.5:
				pred[0, i] = 1
			else:
				pred[0, i] = 0
				
		return pred


if __name__ == '__main__':
	train_x = 
	LogisticRegression()











		