import NB as nb

nb = nb.GaussNB()
data = nb.load_csv('iris.csv', header=True)
train_list, test_list = nb.split_data(data, weight=.90)
group = nb.group_by_class(data, -1)  # designating the last column as the class column
nb.train(train_list, -1)
predicted = nb.predict(test_list)

ac, cm, re = nb.report(test_list, predicted)
print(re, "\n")
print(cm, "\n")
print(ac, "\n")
