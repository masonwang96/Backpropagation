import math
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():

	def __init__(self, x, y, learning_rate, num_iterations):
		self.x = x
		self.y = y
		self.learning_rate = learning_rate
		self.num_iterations = num_iterations
		self.W = np.zeros(shape=(1, 1))	# m x 1 
		self.b = np.zeros(shape=(1, 1)) 			# 1 x 1 

	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))

	def sigmoid_derivative(self, x):
		s = self.sigmoid(x) 

		return s * (1-s)

	def propagate(self):
		# num of samples
		m = self.x.shape[0]

		for xi in self.x[1]:

		# Forward prop
		Z = np.dot(self.W.T, self.x) + self.b
		A = self.sigmoid(Z)
		Loss = 1/m * np.dot((A-self.y).T, A-self.y)

		# Backward prop
		dL_dA = 2 * (A-self.y)
		dA_dZ = self.sigmoid_derivative(self.x)
		dZ_dW = self.x
		dZ_db = 1
		dW = (1/m) * dL_dA * dA_dZ * dZ_dW
		db = (1/m) * dL_dA * dA_dZ * dZ_db

		loss = np.squeeze(loss)	#从数组的形状中删除单维度条目，即把shape中为1的维度去掉
		grads = {'dw': dW, 'db': db}

		return grads, loss

	def optimize(self):
		losses = []

		for i in range(self.num_iterations):
			# Generate grads
			grads, loss = self.propagate()
			dW = grads['dW']
			db = grads['db']

			# Update params
			W = W - dW * self.learning_rate
			b = b - db * self.learning_rate

			# Record losses
			losses.append(loss)

			if i % 10 == 0:
				print('Loss after iteration %i: %f' % (i, loss))
		print('Training finished!!!')

		params = {'W': W, 'b': b}
		grads = {'dw': dW, 'db': db}

		return params, grads, losses

	def predict(self, x):
		# num of samples
		m = x.shape[0]
		pred = np.zeros((m, 1))

		for i in range(A.shape[0]):
			# Convert probabilities A[0,i] to actual predictions p[0,i]
			A = self.sigmoid(np.dot(self.W.T, self.x)+self.b)
			pred[i, 0] = A	

		return pred


if __name__ == '__main__':
	x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
	noise = np.random.normal(0, 0.05, x_data.shape)
	y_data = 2*x_data - 2 + noise
	# y_data = np.square(x_data)- 0.5 + noise
	regressor = LinearRegression(x_data, y_data, 0.01, 100)
	regressor.optimize()

	# Plot data
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(x_data, y_data)
	prediction_value = regressor.predict(x_data)
	lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
	plt.show()
	
	








		