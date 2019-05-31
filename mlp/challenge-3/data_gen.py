import numpy as np
import random
import csv
import random
import math
from mlp.MultiLayerPerceptron import NueralNetwork as mlp


def data_generator(mean, cov, count):
    np.random.seed(1)
    x, y = np.random.multivariate_normal(mean, cov, count).T
    # plt.plot(x, y, 'x')
    # plt.axis('equal')
    # plt.show()
    return [x, y]


def generate_dataset(count):
    # class 1
    mean1 = [0, 0]
    cov1 = [[0.5, 0.3], [0.3, 1]]

    # class 2
    mean2 = [1, 2]
    cov2 = [[0.25, 0.3], [0.3, 1]]

    # class 2
    mean3 = [2, 0]
    cov3 = [[0.5, 0.3], [0.3, 1]]

    x1, y1 = data_generator(mean1, cov1, count)
    x2, y2 = data_generator(mean2, cov2, count)
    x3, y3 = data_generator(mean3, cov3, count)
    my_classes = []
    for i in range(len(x1) - 1):
        my_classes.append([x1[i], y1[i], 1])
        my_classes.append([x2[i], y2[i], 2])
        my_classes.append([x3[i], y3[i], 3])

    random.shuffle(my_classes)
    return my_classes


def translate(data):
    labels = np.ndarray(shape=(len(data),), dtype="object")
    for i in range(len(data)):
        labels[i] = data[i][(len(data[0]) - 1)]
    last = (len(data[0]) - 1)
    for row in data:
        del row[last]
    data = np.array(data)
    return data, labels


def split_data(data, label, weight):
    n_training_samples = int(len(data) * weight)

    np.random.seed(1)  # lock the random numbers
    indices = np.random.permutation(len(data))
    learnset_data = data[indices[:-n_training_samples]]
    learnset_labels = label[indices[:-n_training_samples]]
    testset_data = data[indices[-n_training_samples:]]
    testset_labels = label[indices[-n_training_samples:]]

    return learnset_data, learnset_labels, testset_data, testset_labels


import math
import numpy as np


class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0


class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForword(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output)
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (
                    dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight
            dendron.weight = dendron.weight + dendron.dWeight
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)
        self.error = 0


class Network:
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword()

    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()
        return output

    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            if (o > 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()
        return output


if __name__ == '__main__':
    """ define the model structure """
    layers = []
    layers.append(2)
    layers.append(4)
    layers.append(3)
    model = Network(layers)

    data, label = translate(generate_dataset(100))

    epochs = 1000
    error_threshold = 0.08
    Neuron.eta = 0.2
    Neuron.alpha = 0.1
    inputs = data
    outputs = label
    oo = list()
    for o in outputs:
        if o is 1:
            oo.append([1, 0, 0])
        elif o is 2:
            oo.append([0, 1, 0])
        elif o is 3:
            oo.append([0, 0, 1])
    outputs = oo
    for _ in range(epochs):
        err = 0
        for i in range(len(inputs)):
            model.setInput(inputs[i])
            model.feedForword()
            model.backPropagate(outputs[i])
            err = err + model.getError(outputs[i])
        print("error : ", err)
        if err < error_threshold:
            break

    print(model.getThResults())
    print(model.getResults())
