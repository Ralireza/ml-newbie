from collections import defaultdict
from math import pi
from math import e
import random
import csv
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class GaussNB:
    def __init__(self):
        pass

    def load_csv(self, data, header=False):

        ifile = open(data, "rt", )
        lines = csv.reader(ifile)
        dataset = list(lines)
        if header:
            # remove header
            dataset = dataset[1:]
        for i in range(len(dataset)):
            dataset[i] = [float(x) if re.search('\d', x) else x for x in dataset[i]]
        return dataset

    def split_data(self, data, weight):

        train_size = int(len(data) * weight)
        train_set = []
        for i in range(train_size):
            random.seed(1)
            index = random.randrange(len(data))
            train_set.append(data[index])
            data.pop(index)
        return [train_set, data]

    def group_by_class(self, data, target):

        target_map = defaultdict(list)
        for index in range(len(data)):
            features = data[index]
            if not features:
                continue
            x = features[target]
            target_map[x].append(features[:-1])  # designating the last column as the class column
        return dict(target_map)

    def mean(self, numbers):

        result = sum(numbers) / float(len(numbers))
        return result

    def stdev(self, numbers):

        avg = self.mean(numbers)
        squared_diff_list = []
        for num in numbers:
            squared_diff = (num - avg) ** 2
            squared_diff_list.append(squared_diff)
        squared_diff_sum = sum(squared_diff_list)
        sample_n = float(len(numbers) - 1)

        if sample_n == 0:
            sample_n = 1
        var = squared_diff_sum / sample_n
        return var ** .5

    def summarize(self, test_set):
        """
        :param test_set: lists of features
        :return:
        Use zip to line up each feature into a single column across multiple lists.
        yield the mean and the stdev for each feature.
        """
        print(test_set)
        for feature in zip(*test_set):
            yield {
                'stdev': self.stdev(feature),
                'mean': self.mean(feature)
            }

    def prior_prob(self, group, target, data):
        """
        :return:
        The probability of each target class
        """
        total = float(len(data))
        result = len(group[target]) / total
        return result

    def train(self, train_list, target):
        """
        :param data:
        :param target: target class
        :return:
        For each target:
            1. yield prior_prob: the probability of each class. P(class) eg P(Iris-virginica)
            2. yield summary: list of {'mean': 0.0, 'stdev': 0.0}
        """
        group = self.group_by_class(train_list, target)
        self.summaries = {}
        for target, features in group.items():
            print(target)
            self.summaries[target] = {
                'prior_prob': self.prior_prob(group, target, train_list),
                'summary': [i for i in self.summarize(features)],
            }
        return self.summaries

    def normal_pdf(self, x, mean, stdev):
        """
        :param x: a variable
        :param mean: µ - the expected value or average from M samples
        :param stdev: σ - standard deviation
        :return: Gaussian (Normal) Density function.
        N(x; µ, σ) = (1 / 2πσ) * (e ^ (x–µ)^2/-2σ^2
        """
        variance = stdev ** 2
        exp_squared_diff = (x - mean) ** 2
        exp_power = -exp_squared_diff / (2 * variance)
        exponent = e ** exp_power
        denominator = ((2 * pi) ** .5) * stdev
        normal_prob = exponent / denominator
        return normal_prob

    def marginal_pdf(self, joint_probabilities):
        """
        :param joint_probabilities: list of joint probabilities for each feature
        :return:
        Marginal Probability Density Function (Predictor Prior Probability)
        Joint Probability = prior * likelihood
        Marginal Probability is the sum of all joint probabilities for all classes.
        marginal_pdf =
          [P(setosa) * P(sepal length | setosa) * P(sepal width | setosa) * P(petal length | setosa) * P(petal width | setosa)]
        + [P(versicolour) * P(sepal length | versicolour) * P(sepal width | versicolour) * P(petal length | versicolour) * P(petal width | versicolour)]
        + [P(virginica) * P(sepal length | verginica) * P(sepal width | verginica) * P(petal length | verginica) * P(petal width | verginica)]
        """
        marginal_prob = sum(joint_probabilities.values())
        return marginal_prob

    def joint_probabilities(self, test_row):
        """
        :param test_row: single list of features to test; new data
        :return:
        Use the normal_pdf(self, x, mean, stdev) to calculate the Normal Probability for each feature
        Take the product of all Normal Probabilities and the Prior Probability.
        """
        joint_probs = {}
        for target, features in self.summaries.items():
            total_features = len(features['summary'])
            likelihood = 1
            for index in range(total_features):
                feature = test_row[index]
                mean = features['summary'][index]['mean']
                stdev = features['summary'][index]['stdev']
                normal_prob = self.normal_pdf(feature, mean, stdev)
                likelihood *= normal_prob
            prior_prob = features['prior_prob']
            joint_probs[target] = prior_prob * likelihood
        return joint_probs

    def posterior_probabilities(self, test_row):
        """
        :param test_row: single list of features to test; new data
        :return:
        For each feature (x) in the test_row:
            1. Calculate Predictor Prior Probability using the Normal PDF N(x; µ, σ). eg = P(feature | class)
            2. Calculate Likelihood by getting the product of the prior and the Normal PDFs
            3. Multiply Likelihood by the prior to calculate the Joint Probability.
        E.g.
        prior_prob: P(setosa)
        likelihood: P(sepal length | setosa) * P(sepal width | setosa) * P(petal length | setosa) * P(petal width | setosa)
        joint_prob: prior_prob * likelihood
        marginal_prob: predictor prior probability
        posterior_prob = joint_prob/ marginal_prob
        returning a dictionary mapping of class to it's posterior probability
        """
        posterior_probs = {}
        joint_probabilities = self.joint_probabilities(test_row)
        marginal_prob = self.marginal_pdf(joint_probabilities)
        for target, joint_prob in joint_probabilities.items():
            posterior_probs[target] = joint_prob / marginal_prob
        return posterior_probs

    def get_map(self, test_row):
        """
        :param test_row: single list of features to test; new data
        :return:
        Return the target class with the largest/best posterior probability
        """
        posterior_probs = self.posterior_probabilities(test_row)
        map_prob = max(posterior_probs, key=posterior_probs.get)
        return map_prob

    def predict(self, test_set):
        """
        :param test_set: list of features to test on
        :return:
        Predict the likeliest target for each row of the test_set.
        Return a list of predicted targets.
        """
        map_probs = []
        for row in test_set:
            map_prob = self.get_map(row)
            map_probs.append(map_prob)
        return map_probs

    def report(self, test, predict):
        actual = [item[-1] for item in test]
        accuracy = accuracy_score(actual, predict)
        cm = confusion_matrix(actual, predict)
        report = classification_report(actual, predict)
        return accuracy, cm, report
