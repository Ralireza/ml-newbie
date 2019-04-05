import KNN

myknn = KNN.knn(csv_file='iris.csv', data=None, k=1, weight=.67)

test, predicted = myknn.test_data()
print((test))
print((predicted))
print(myknn.accuracy(test, predicted))
