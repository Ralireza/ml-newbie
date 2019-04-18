import numpy as np
from collections import Counter
import csv
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class knn:

    def __init__(self, csv_file=None, data=None, k=1, weight=.67):
        if not csv_file is None:
            data = self.load_csv(csv_file, header=False)
        mydata, labels = self.translate(data)
        self.learnset_data, self.learnset_labels, self.testset_data, self.testset_labels = self.split_data(mydata,
                                                                                                           labels,
                                                                                                           weight)
        self.k = k

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
        return class_counter.most_common(1)[0][0]

    def get_neighbors(self, training_set,
                      labels,
                      test_instance,
                      k,
                      distance=distance):

        distances = []
        for index in range(len(training_set)):
            dist = distance(test_instance, training_set[index])
            distances.append((training_set[index], dist, labels[index]))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:k]
        return neighbors

    def test_data(self):

        # print("index: ", i,
        #       ", result of vote: ", self.vote(neighbors),
        #       ", label: ", testset_labels[i],
        #       ", data: ", testset_data[i])
        predicte = list()
        for i in range(len(self.testset_data)):
            neighbors = self.get_neighbors(self.learnset_data,
                                           self.learnset_labels,
                                           self.testset_data[i],
                                           self.k,
                                           distance=self.distance)
            predicte.append(self.vote(neighbors))

            test = list(self.testset_labels)
        return test, predicte

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
        learnset_data = data[indices[:-n_training_samples]]
        learnset_labels = label[indices[:-n_training_samples]]
        testset_data = data[indices[-n_training_samples:]]
        testset_labels = label[indices[-n_training_samples:]]

        return learnset_data, learnset_labels, testset_data, testset_labels

    def report(self, test, predict):
        accuracy = accuracy_score(test, predict)
        cm = confusion_matrix(test, predict)
        report = classification_report(test, predict)
        return accuracy, cm, report
