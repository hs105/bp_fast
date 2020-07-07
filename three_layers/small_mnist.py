import numpy as np
from numpy.linalg import norm as norm
from sklearn import datasets
import sklearn.metrics
from mse import MSE, Relu, Sigmoid
from network import Network

data = datasets.load_digits()

x = data["data"]
y = data["target"]
y = np.eye(10)[y]
print('#data:', len(x))
print('#data[0].shape', x[0].shape)

# input normalization (L2)
x = x / norm(x, axis=1)[:, None]
for i in range(len(x)):
    #     print('after normalization, norm(x[i])=', norm(x[i]))
    assert (abs(norm(x[i]) - 1.0) < 1e-8)

num_epochs = 1000
b_size = 32


# delta2 also decreases, but variance is a bit high
# the variance of the grad2 is really high.

# lr_array = [0.01, 0.015625, 0.0625, 0.1, 0.2, 0.25]# add power(2, -6.0), power(2, -4.0), (2, -2.0)
# lr_array = [0.5, 0.625]#error too big
lr_array = [0.3, 0.4]
for lr in lr_array:
    nn = Network((64, 100, 10), (Relu, Sigmoid))

    losses = nn.fit(x, y, loss=MSE, epochs=num_epochs, batch_size=b_size, learning_rate=lr)

    prediction = nn.predict(x)
    y_true = []
    y_pred = []
    for i in range(len(y)):
        y_pred.append(np.argmax(prediction[i]))
        y_true.append(np.argmax(y[i]))
    print(sklearn.metrics.classification_report(y_true, y_pred))

    # np.save('loss_const_'+str(lr)+'_updatelayer1.npy', losses)
    np.save('loss_const_' + str(lr) + '.npy', losses)

