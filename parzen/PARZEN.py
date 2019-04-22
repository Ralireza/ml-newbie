import numpy as np
from collections import Counter
import csv
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
import sys


class knn:

    def __init__(self, csv_file=None, data=None, r=None, weight=.67):
        if r is None:
            r = [1, 2]
        if not csv_file is None:
            data = self.load_csv(csv_file, header=False)
        mydata, labels = self.translate(data)
        self.train_data, self.train_label, self.test_data, self.test_label = self.split_data(mydata,
                                                                                             labels,
                                                                                             weight)
        self.r = r

    def load_csv(self, data, header=False):

        ifile = open(data, "rt", )
        lines = csv.reader(ifile)
        dataset = list(lines)
        if header:
            # remove header
            dataset = dataset[1:]
        f = len(dataset)
        for i in range(len(dataset)):
            dataset[i] = [float(x) if re.search('\d', x) else x for x in dataset[i]]
        # print(dataset)
        return dataset

    def distance(self, instance1, instance2):
        # just in case, if the instances are lists or tuples:
        instance1 = np.array(instance1)
        instance2 = np.array(instance2)
        return np.linalg.norm(instance1 - instance2)

    def vote(self, neighbors):
        class_counter = Counter()
        for neighbor in neighbors:
            class_counter[neighbor[2]] += 1
            # print(class_counter)
        return class_counter.most_common(1)[0][0]

    def get_neighbors(self, training_set,
                      labels,
                      test_instance,
                      r,
                      distance=distance):

        distances = []
        for index in range(len(training_set)):
            dist = distance(test_instance, training_set[index])
            if dist < r:
                distances.append((training_set[index], dist, labels[index]))
        distances.sort(key=lambda x: x[1])
        neighbors = distances
        # print(test_instance)
        # print(neighbors)
        return neighbors

    def translate(self, data):
        labels = np.ndarray(shape=(len(data),), dtype="object")
        for i in range(len(data)):
            labels[i] = data[i][(len(data[0]) - 1)]
        last = (len(data[0]) - 1)
        for row in data:
            del row[last]
        data = np.array(data)
        return data, labels

    def split_data(self, data, label, weight):

        n_training_samples = int(len(data) * weight)
        np.random.seed(1)  # lock the random numbers
        indices = np.random.permutation(len(data))
        train_data = data[indices[:-n_training_samples]]
        train_label = label[indices[:-n_training_samples]]
        test_data = data[indices[-n_training_samples:]]
        test_label = label[indices[-n_training_samples:]]
        return train_data, train_label, test_data, test_label

    def kfold_validation(self, fold_number):
        kf = KFold(n_splits=fold_number, shuffle=False)
        accuracy_cvs = dict()
        bad_radius = set()
        all_redius = self.r
        for train_index, validation_index in kf.split(self.train_data):
            train_data, validation_data = self.train_data[train_index], self.train_data[validation_index]
            train_label, validation_label = self.train_label[train_index], self.train_label[validation_index]
            test = list(validation_label)
            good_radius = True

            for radius in all_redius:
                predict = list()
                for i in range(len(validation_label)):
                    neighbors = self.get_neighbors(train_data,
                                                   train_label,
                                                   validation_data[i],
                                                   radius,
                                                   distance=self.distance)
                    if len(neighbors) == 0:
                        print("Radius ", radius, "is too short !!")
                        bad_radius.add(radius)
                        good_radius = False
                        break
                    else:
                        predict.append(self.vote(neighbors))
                if good_radius and radius not in accuracy_cvs:
                    accuracy_cvs[radius] = accuracy_score(test, predict)
                elif good_radius and accuracy_cvs[radius] < accuracy_score(test, predict):
                    accuracy_cvs[radius] = accuracy_score(test, predict)
       # remove bad radius in all fold iteration
        for items in bad_radius:
            accuracy_cvs.pop(items)
        return accuracy_cvs

    def test_data(self):

        # print("index: ", i,
        #       ", result of vote: ", self.vote(neighbors),
        #       ", label: ", test_label[i],
        #       ", data: ", test_data[i])
        predicte = list()
        for i in range(len(self.test_data)):
            neighbors = self.get_neighbors(self.train_data,
                                           self.train_label,
                                           self.test_data[i],
                                           self.k,
                                           distance=self.distance)
            predicte.append(self.vote(neighbors))

            test = list(self.test_label)
        return test, predicte

    def report(self, test, predict):
        accuracy = accuracy_score(test, predict)
        cm = confusion_matrix(test, predict)
        report = classification_report(test, predict)
        return accuracy, cm, report
