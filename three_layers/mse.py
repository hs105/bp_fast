import numpy as np


# implement step-size idea
# minimize the mse, E|\delta|^2,

# np.random.seed(1)
np.random.seed()
print('random number:', np.random.randn())
print('random number:', np.random.randn())



# based on bp
# let's examine the training error in each layer during layer-by-layer training.
# my hypothesis: training on layer L-1 can increase the training error for L.


class MSE:
    def __init__(self, activation_fn):
        """
        :param activation_fn: Class object of the activation function.
        """
        self.activation_fn = activation_fn

    def activation(self, z):
        return self.activation_fn.activation(z)

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def prime(y_true, y_pred):
        return y_pred - y_true

    def delta(self, y_true, y_pred):
        """
        Back propagation error delta
        :return: (array)
        """
        return self.prime(y_true, y_pred) * self.activation_fn.prime(y_pred)


class Relu:
    @staticmethod
    def activation(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def prime(z):
        '''
        the derivative of z
        '''
        g = np.zeros_like(z)
        g[np.nonzero(z)] = 1.0
        return g


# x_test = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
# print(x_test)
# print(Relu.prime(x_test))


class Sigmoid:
    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        return Sigmoid.activation(z) * (1 - Sigmoid.activation(z))
