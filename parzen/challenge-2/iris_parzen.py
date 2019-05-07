import parzen.PARZEN as parzen

radius = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 2, 3, 4]

my_parzen = parzen.ParzenClassifier(csv_file='iris.csv', data=None, r=radius, weight=.90)
radius_accuracy_dict, best_radius = my_parzen.kfold_validation(10)

test, predicted = my_parzen.test(best_radius)

ac, cm, re = my_parzen.report(test, predicted)


print(re,"\n")
print(cm,"\n")
print(ac,"\n")



