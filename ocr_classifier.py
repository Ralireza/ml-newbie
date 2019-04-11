import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import pickle
import os
import csv
import NB as nb

import KNN


def extract_features(image_path, vector_size, label):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (vector_size, vector_size))

    small = small.flatten()
    features = []
    features = (small).tolist()
    features[-1] = str(label)
    # imgplot = plt.imshow(small)
    # plt.show()
    return features


# print(feature_extractor('/classes/0/0.jpg'))

def write_csv(images_path='classes/', pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    final_path = {}
    database = []
    for f in files:
        tmp_list = [os.path.join(f, p) for p in sorted(os.listdir(f))]
        # tmp_list[-1] = f[8:]
        final_path[f[8:]] = (tmp_list)
        # print(final_path)
        with open('file.csv', "w") as csv_file:
            for key, value in final_path.items():
                writer = csv.writer(csv_file, delimiter=',')
                for path in value:
                    writer.writerow(extract_features(path, 3, key))


write_csv()


I_WANT_USE_BAYES = True


if I_WANT_USE_BAYES:
    nb = nb.GaussNB()
    data = nb.load_csv('file.csv', header=False)
    group = nb.group_by_class(data, -1)  # designating the last column as the class column
    train_list, test_list = nb.split_data(data, weight=.90)
    nb.train(train_list, -1)
    predicted = nb.predict(test_list)

    ac, cm, re = nb.report(test_list, predicted)
    print(re, "\n")
    print(cm, "\n")
    print(ac, "\n")


else:

    myknn = KNN.knn(csv_file='file.csv', data=None, k=2, weight=.50)
    test, predicted = myknn.test_data()
    ac, cm, re = myknn.report(test, predicted)
    print(ac)
    print(cm)
    print(re)
