import KNN

myknn = KNN.knn(csv_file='iris.csv', data=None, k=1, weight=.90)

test, predicted = myknn.test_data()

ac, cm, re = myknn.report(test, predicted)


print(re,"\n")
print(cm,"\n")
print(ac,"\n")



