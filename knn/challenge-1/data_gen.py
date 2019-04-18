import bayes.NB as nb
import numpy as np
import knn.KNN as KNN
import random


def data_generator(mean, cov, count):
    np.random.seed(1)
    x, y = np.random.multivariate_normal(mean, cov, count).T
    # plt.plot(x, y, 'x')
    # plt.axis('equal')
    # plt.show()
    return [x, y]


def generate_dataset_1(count):
    # class 1
    mean1 = [0, 0]
    cov1 = [[0.5, 0.3], [0.3, 1]]

    # class 2
    mean2 = [1, 2]
    cov2 = [[0.25, 0.3], [0.3, 1]]

    x1, y1 = data_generator(mean1, cov1, count)
    x2, y2 = data_generator(mean2, cov2, count)
    my_classes = []
    for i in range(len(x1) - 1):
        my_classes.append([x1[i], y1[i], 'class1'])
        my_classes.append([x2[i], y2[i], 'class2'])

    random.shuffle(my_classes)
    return my_classes


myknn = KNN.knn(data=generate_dataset_1(100), k=3, weight=0.67)

test, predicted = myknn.test_data()
ac, cm, re = myknn.report(test, predicted)

print(re, "\n")
print(cm, "\n")
print(ac, "\n")
