import numpy as np
import parzen.PARZEN as PARZEN
import random


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
        my_classes.append([x1[i], y1[i], 'class1'])
        my_classes.append([x2[i], y2[i], 'class2'])
        my_classes.append([x3[i], y3[i], 'class3'])

    random.shuffle(my_classes)
    return my_classes


radius = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 2, 3, 4]
myknn = PARZEN.knn(data=generate_dataset(100), r=radius, weight=0.5)

asd = myknn.kfold_validation(10)
print(asd)
# ac, cm, re = myknn.report(test, predicted)
#
# print(re, "\n")
# print(cm, "\n")
# print(ac, "\n")
#
# from sklearn.model_selection import KFold, RepeatedKFold
#
# kf = KFold(n_splits=5, shuffle=False)
# X = np.array([[1, 2], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4]])
# y = np.array([[1, 2], [3, 4], [3, 4], [3, 4], [3, 4], [3, 4]])
#
# for train_index, test_index in kf.split(X):
#     print("Train:", train_index, "Validation:", test_index, X[test_index])
#     X_train, X_test = X[train_index], X[test_index]
#     # y_train, y_test = y[train_index], y[test_index]
