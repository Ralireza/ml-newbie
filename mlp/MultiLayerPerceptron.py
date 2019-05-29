import numpy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
# The following code is used for hiding the warnings and make this notebook clearer.
import warnings

warnings.filterwarnings('ignore')


class NueralNetwork:
    #########
    # parameters
    # ----------
    # self:      the class object itself
    # net_arch:  consists of a list of integers, indicating
    #            the number of neurons in each layer, i.e. the network architecture
    #########
    def __init__(self, input, net_arch):
        numpy.random.seed(0)

        # Initialized the weights, making sure we also
        # initialize the weights for the biases that we will add later

        self.activity = self.tanh
        self.activity_derivative = self.tanh_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 1
        self.arch = net_arch
        self.weights = []
        self.X = input

        # Random initialization with range of weight values (-1,1)
        for layer in range(self.layers - 1):
            w = 2 * numpy.random.rand(net_arch[layer] + 1, net_arch[layer + 1]) - 1
            self.weights.append(w)

    def tanh(self, x):
        return (1.0 - numpy.exp(-2 * x)) / (1.0 + numpy.exp(-2 * x))

    def tanh_derivative(self, x):
        return (1 + self.tanh(x)) * (1 - self.tanh(x))

    def _forward_prop(self, x):
        y = x

        for i in range(len(self.weights) - 1):
            activation = numpy.dot(y[i], self.weights[i])
            activity = self.activity(activation)

            # add the bias for the next layer
            activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
            y.append(activity)

        # last layer
        activation = numpy.dot(y[-1], self.weights[-1])
        activity = self.activity(activation)
        y.append(activity)

        return y

    def _back_prop(self, y, target, learning_rate):
        error = target - y[-1]
        delta_vec = [error * self.activity_derivative(y[-1])]

        # we need to begin from the back, from the next to last layer
        for i in range(self.layers - 2, 0, -1):
            error = delta_vec[-1].dot(self.weights[i][1:].T)
            error = error * self.activity_derivative(y[i][1:])
            delta_vec.append(error)

        # Now we need to set the values from back to front
        delta_vec.reverse()

        # Finally, we adjust the weights, using the backpropagation rules
        for i in range(len(self.weights)):
            layer = y[i].reshape(1, self.arch[i] + 1)
            delta = delta_vec[i].reshape(1, self.arch[i + 1])
            self.weights[i] += learning_rate * layer.T.dot(delta)

    #########
    # parameters
    # ----------
    # self:    the class object itself
    # data:    the set of all possible pairs of booleans True or False indicated by the integers 1 or 0
    # labels:  the result of the logical operation 'xor' on each of those input pairs
    #########
    def fit(self, data, labels, learning_rate=0.1, epochs=100):

        # Add bias units to the input layer -
        # add a "1" to the input data (the always-on bias neuron)
        ones = numpy.ones((1, data.shape[0]))
        Z = numpy.concatenate((ones.T, data), axis=1)

        for k in range(epochs):
            if (k + 1) % 10000 == 0:
                print('epochs: {}'.format(k + 1))

            sample = numpy.random.randint(self.X.shape[0])

            # We will now go ahead and set up our feed-forward propagation:
            x = [Z[sample]]
            y = self._forward_prop(x)

            # Now we do our back-propagation of the error to adjust the weights:
            target = labels[sample]
            self._back_prop(y, target, learning_rate)

    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    #
    # parameters
    # ----------
    # self:   the class object itself
    # x:      single input data
    #########
    def predict_single_data(self, x):
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activity(numpy.dot(val, self.weights[i]))
            val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))
        return val[1]

    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    #
    # parameters
    # ----------
    # self:   the class object itself
    # X:      the input data array
    #########
    def predict(self, X):
        Y = numpy.array([]).reshape(0, self.arch[-1])
        for x in X:
            y = numpy.array([[self.predict_single_data(x)]])
            Y = numpy.vstack((Y, y))
        return Y

    def plot_decision_regions(self, X, y, classifier, test_idx=None, resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)

        # highlight test samples
        if test_idx:
            # plot all samples
            X_test, y_test = X[test_idx, :], y[test_idx]

            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='',
                        alpha=1.0,
                        linewidths=1,
                        marker='o',
                        s=55, label='test set')
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
