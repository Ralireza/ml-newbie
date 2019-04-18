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


I_WANT_USE_BAYES = False

if I_WANT_USE_BAYES:
    nb = nb.GaussNB()
    data = generate_dataset_1(1000)
    train_list, test_list = nb.split_data(data, weight=0.8)
    # print("Using %s rows for training and %s rows for testing" % (len(train_list), len(test_list)))
    group = nb.group_by_class(data, -1)  # designating the last column as the class column
    # print("Grouped into %s classes: %s" % (len(group.keys()), group.keys()))
    nb.train(train_list, -1)
    predicted = nb.predict(test_list)
    ac, cm, re = nb.report(test_list, predicted)
    print(re, "\n")
    print(cm, "\n")
    print(ac, "\n")

else:

    myknn = KNN.knn(data=generate_dataset_1(100), k=3, weight=l)

    test, predicted = myknn.test_data()
    ac, cm, re = myknn.report(test, predicted)

    print(re, "\n")
    print(cm, "\n")
    print(ac, "\n")
