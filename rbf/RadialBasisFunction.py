import numpy as np
from numpy.linalg.linalg import pinv
from sklearn import datasets


def scale(data):
    mat = np.asmatrix(data)
    height, width = mat.shape
    for i in range(0, width):
        minimum = np.min(mat[:, i])
        maximum = np.max(mat[:, i])
        for k in range(0, height):
            mat[k, i] = (mat[k, i] - minimum) / (maximum - minimum)
    return mat


class RBFNetwork:
    def __init__(self, pTypes, scaledData, labels):
        self.pTypes = pTypes
        self.protos = np.zeros(shape=(0, 4))
        self.scaledData = scaledData
        self.spread = 0
        self.labels = labels
        self.weights = 0

    def generatePrototypes(self):
        group1 = np.random.randint(0, 49, size=self.pTypes)
        group2 = np.random.randint(50, 100, size=self.pTypes)
        group3 = np.random.randint(101, 150, size=self.pTypes)
        self.protos = np.vstack(
            [self.protos, self.scaledData[group1, :], self.scaledData[group2, :], self.scaledData[group3, :]])
        return self.protos

    def sigma(self):
        dTemp = 0
        for i in range(0, self.pTypes * 3):
            for k in range(0, self.pTypes * 3):
                dist = np.square(np.linalg.norm(self.protos[i] - self.protos[k]))
                if dist > dTemp:
                    dTemp = dist
        self.spread = dTemp / np.sqrt(self.pTypes * 3)

    def train(self):
        self.generatePrototypes()
        self.sigma()
        hiddenOut = np.zeros(shape=(0, self.pTypes * 3))
        for item in self.scaledData:
            out = []
            for proto in self.protos:
                distance = np.square(np.linalg.norm(item - proto))
                neuronOut = np.exp(-(distance) / (np.square(self.spread)))
                out.append(neuronOut)
            hiddenOut = np.vstack([hiddenOut, np.array(out)])
        # print(hiddenOut)
        self.weights = np.dot(pinv(hiddenOut), self.labels)
        # print(self.weights)

    def test(self):
        items = [3, 4, 72, 82, 91, 120, 134, 98, 67, 145, 131]
        for item in items:
            data = self.scaledData[item]
            out = []
            for proto in self.protos:
                distance = np.square(np.linalg.norm(data - proto))
                neuronOut = np.exp(-(distance) / np.square(self.spread))
                out.append(neuronOut)

            netOut = np.dot(np.array(out), self.weights)
            print('---------------------------------')
            print(netOut)
            print('Class is ', netOut.argmax(axis=0) + 1)
            print('Given Class ', self.labels[item])


iris = datasets.load_iris()
X = iris.data[:, :]
y = iris.target
np.random.seed(1)
weight = 0.01
indices = np.random.permutation(len(X))
data = X[indices[:]]
labels = y[indices[:]]
oo = []
print(oo)
for o in labels:
    if o == 0:
        oo.append([1, 0, 0])
    elif o == 1:
        oo.append([0, 1, 0])
    elif o == 2:
        oo.append([0, 0, 1])
labels = oo
scaledData = scale(data)
network = RBFNetwork(4, scaledData, labels)
network.train()
network.test()
