import numpy as np
import matplotlib.pyplot as plt

#learningRate学习率，Loopnum迭代次数
learning_rate = 0.1
num_iterations = 120 
data_x = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, data_x.shape)
data_y = -8*data_x + 5 + noise
# y_data = np.square(x_data)- 0.5 + noise

Weight=np.ones(shape=(1,data_x.shape[1]))
baise=np.array([[1]])

# Plot data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data_x, data_y)
plt.ion()
plt.show() 

 
for i in range(num_iterations):
    WX_Plus_B = np.dot(data_x, Weight.T) + baise 

    loss = np.dot((WX_Plus_B-data_y).T, WX_Plus_B-data_y)/data_y.shape[0]
    w_gradient = (2/data_x.shape[0]) * np.dot((WX_Plus_B-data_y).T,data_x)
    baise_gradient = 2*np.dot((WX_Plus_B-data_y).T, np.ones(shape=[data_x.shape[0],1]))/data_x.shape[0]

    Weight = Weight - learning_rate * w_gradient
    baise = baise - learning_rate * baise_gradient

    try:
        ax.lines.remove(lines[0])
    except Exception as e:
        pass
    else:
        pass
    finally:
        pass
    prediction_value = np.dot(data_x, Weight.T) + baise 
    lines = ax.plot(data_x, prediction_value, 'r-', lw = 2)
    plt.pause(0.1)

    if i%20 == 0:
        print('Iter:', i, ', Loss', loss)       #每迭代50次输出一次loss

      




