import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from som.SelfOrganizedMap import SOM


# https://tcosmo.github.io/2017/07/27/fun-with-som.html

iris = datasets.load_iris()
X = iris.data[:, :]
y = iris.target
np.random.seed(1)
weight = 0.01
indices = np.random.permutation(len(X))
data = X[indices[:]]
labels = y[indices[:]]

som_iris = SOM(20, 20, 4)
frames_iris = []
som_iris.train(data, L0=0.8, lam=1e2, sigma0=10, frames=frames_iris)
print("quantization error:", som_iris.quant_err())