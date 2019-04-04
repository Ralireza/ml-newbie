import numpy as np
from sklearn import datasets
from collections import Counter


iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
# print(iris_data)
# print(iris_labels)

# create learn set and test set
np.random.seed(1)  # lock the random numbers
indices = np.random.permutation(len(iris_data))
n_training_samples = 15
learnset_data = iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]


def getExampleProb(self, test_example):
    '''
        Parameters:
        -----------
        1. a single test example
        What the function does?
        -----------------------
        Function that estimates posterior probability of the given test example
        Returns:
        ---------
        probability of test example in ALL CLASSES
    '''

    likelihood_prob = np.zeros(self.classes.shape[0])  # to store probability w.r.t each class

    # finding probability w.r.t each class of the given test example
    for cat_index, cat in enumerate(self.classes):

        for test_token in test_example.split():  # split the test example and get p of each test word

            ####################################################################################

            # This loop computes : for each word w [ count(w|c)+1 ] / [ count(c) + |V| + 1 ]

            ####################################################################################

            # get total count of this test token from it's respective training dict to get numerator value
            test_token_counts = self.cats_info[cat_index][0].get(test_token, 0) + 1

            # now get likelihood of this test_token word
            test_token_prob = test_token_counts / float(self.cats_info[cat_index][2])

            # remember why taking log? To prevent underflow!
            likelihood_prob[cat_index] += np.log(test_token_prob)

    # we have likelihood estimate of the given example against every class but we need posterior probility
    post_prob = np.empty(self.classes.shape[0])
    for cat_index, cat in enumerate(self.classes):
        post_prob[cat_index] = likelihood_prob[cat_index] + np.log(self.cats_info[cat_index][1])

    return post_prob



getExampleProb(learnset_data)


