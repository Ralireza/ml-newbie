import data_generator as dg
import NB as nb
import numpy as np

import random


def generate_dataset_1():
    # class 1
    mean1 = [0, 0]
    cov1 = [[0.5, 0.3], [0.3, 1]]

    # class 2
    mean2 = [1, 2]
    cov2 = [[0.25, 0.3], [0.3, 1]]

    x1, y1 = dg.data_generator(mean1, cov1)
    x2, y2 = dg.data_generator(mean2, cov2)
    my_classes = []
    for i in range(len(x1) - 1):
        my_classes.append([x1[i], y1[i], 'class1'])
        my_classes.append([x2[i], y2[i], 'class2'])

    random.shuffle(my_classes)
    return my_classes


# nb = nb.GaussNB()
# data = generate_dataset_1()
# train_list, test_list = nb.split_data(data, weight=.67)
# print("Using %s rows for training and %s rows for testing" % (len(train_list), len(test_list)))
# group = nb.group_by_class(data, -1)  # designating the last column as the class column
# print("Grouped into %s classes: %s" % (len(group.keys()), group.keys()))
# nb.train(train_list, -1)
# predicted = nb.predict(test_list)
# accuracy = nb.accuracy(test_list, predicted)
# print('Accuracy: %.3f' % accuracy)

mydata = generate_dataset_1()



