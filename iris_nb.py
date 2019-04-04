import NB as nb

nb = nb.GaussNB()
data = nb.load_csv('iris.csv', header=True)
train_list, test_list = nb.split_data(data, weight=.67)
print("Using %s rows for training and %s rows for testing" % (len(train_list), len(test_list)))
group = nb.group_by_class(data, -1)  # designating the last column as the class column
print("Grouped into %s classes: %s" % (len(group.keys()), group.keys()))
nb.train(train_list, -1)
predicted = nb.predict(test_list)
accuracy = nb.accuracy(test_list, predicted)
print('Accuracy: %.3f' % accuracy)
