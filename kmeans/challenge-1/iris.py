from kmeans.Kmeans import Kmeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib import style
style.use('ggplot')


# https://pythonprogramming.net/k-means-from-scratch-2-machine-learning-tutorial/?completed=/k-means-from-scratch-machine-learning-tutorial/

iris = datasets.load_iris()
X = iris.data[:, :]
y = iris.target
np.random.seed(1)
weight = 0.01
indices = np.random.permutation(len(X))
data = X[indices[:]]
labels = X[indices[:]]
print(X.shape[0])
# to find optimum number of clusters use elbow method
WCSS_array = np.array([])
n_iter = 100
clf = Kmeans()
clf.fit(data)



colors = 10*["g","r","c","b","k"]

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()