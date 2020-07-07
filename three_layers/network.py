import numpy as np
from numpy.linalg import norm as norm
from numpy.linalg import eigh as eig
from copy import deepcopy as deepcopy

# todo: indexing has to be made sure consistent, always starting from 1 not 0?

class Network:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.

        """

        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        self.dimensions = dimensions

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}

        self.w_err = {}
        self.b_err = {}

        # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
        self.activations = {}

        for i in range(len(dimensions) - 1):
            self.w[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
            self.b[i + 1] = np.random.randn(dimensions[i + 1])  # np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]

        self.total_updates = 0  # total updates to layer 1
        self.norm_params = None

    def _feed_forward(self, x):
        """
        Execute a forward feed through the network.

        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer.
                 The numbering of the output is equivalent to the layer numbers.
        """

        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.

        for i in range(1, self.n_layers):
            # current layer = i
            # activation layer = i + 1
            z[i + 1] = np.dot(a[i], self.w[i]) + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])

        return z, a

    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        _, a = self._feed_forward(x)
        return a[self.n_layers]

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.

        :param index: (int) index of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * np.mean(delta, 0)

    def _compute_grad_delta(self, z, a, y_true):
        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = np.dot(a[self.n_layers - 1].T, delta)

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        for i in reversed(range(2, self.n_layers)):
            delta = np.dot(delta, self.w[i].T) * self.activations[i].prime(z[i])
            dw = np.dot(a[i - 1].T, delta)
            update_params[i - 1] = (dw, delta)

        return update_params

    def _compute_norm(self, z, a, y_true):
        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])
        dw = np.dot(a[self.n_layers - 1].T, delta)

        norm_params = {
            self.n_layers - 1: (norm(dw, 'fro'), norm(delta, 'fro'))
        }

        for i in reversed(range(2, self.n_layers)):
            delta = np.dot(delta, self.w[i].T) * self.activations[i].prime(z[i])
            dw = np.dot(a[i - 1].T, delta)
            norm_params[i - 1] = (norm(dw), norm(delta))
        return norm_params

    def get_w2(self):
        '''shape of w2 is n2 by n1'''
        return self.w[2].T

    def _back_prop(self, z, a, y_true, x):
        """
        The input dicts keys represent the layers of the net.

        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              }

        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        """

        update_params = self._compute_grad_delta(z, a, y_true)

        # Update the weights and biases
        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])


    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        # Initiate the loss object with the final activation function
        self.loss = loss(self.activations[self.n_layers])
        self.learning_rate = learning_rate

        delta1 = []
        delta2 = []
        grad1 = []
        grad2 = []
        losses = []
        for i in range(epochs):
            # Shuffle the data
            index = np.arange(x.shape[0])
            np.random.shuffle(index)
            x_ = x[index]
            y_ = y_true[index]

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                z, a = self._feed_forward(x_[k:l])
                self._back_prop(z, a, y_[k:l], x_[k:l])

            if (i + 1) % 10 == 0:
                _, a = self._feed_forward(x)
                loss = self.loss.loss(y_true, a[self.n_layers])
                losses.append(loss)
                print("Loss:", loss)

        return losses


# nn = Network((2, 3, 1), (Relu, Sigmoid))

# print(nn.w)
# print(nn.b)
# print(nn.activations)