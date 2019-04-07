import data_generator as dg
import NB as nb
import random
import KNN

def generate_dataset_2(count):
    # class 1
    mean1 = [-1.5, 1]
    cov1 = [[0.5, 0.3], [0.3, 1]]

    # class 2
    mean2 = [0.5, 0]
    cov2 = [[0.25, 0.3], [0.3, 1]]

    # class 3
    mean3 = [-1.5, -2.25]
    cov3 = [[0.25, 0.3], [0.3, 0.7]]

    x1, y1 = dg.data_generator(mean1, cov1, count)
    x2, y2 = dg.data_generator(mean2, cov2, count)
    x3, y3 = dg.data_generator(mean3, cov3, count)
    my_classes = []
    for i in range(len(x1) - 1):
        my_classes.append([x1[i], y1[i], 'class1'])
        my_classes.append([x2[i], y2[i], 'class2'])
        my_classes.append([x3[i], y3[i], 'class3'])

    random.shuffle(my_classes)
    return my_classes


# nb = nb.GaussNB()
# data = generate_dataset_2()
# train_list, test_list = nb.split_data(data, weight=.67)
# print("Using %s rows for training and %s rows for testing" % (len(train_list), len(test_list)))
# group = nb.group_by_class(data, -1)  # designating the last column as the class column
# print("Grouped into %s classes: %s" % (len(group.keys()), group.keys()))
# nb.train(train_list, -1)
# predicted = nb.predict(test_list)
# accuracy = nb.accuracy(test_list, predicted)
# print('Accuracy: %.3f' % accuracy)

myknn = KNN.knn(data=generate_dataset_2(200), k=3, weight=.67)

test, predicted = myknn.test_data()
print((test))
print((predicted))
print(myknn.accuracy(test, predicted))
