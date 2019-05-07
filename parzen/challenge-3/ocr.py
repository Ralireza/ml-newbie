import cv2
import os
import csv
import parzen.PARZEN as parzen


def extract_features(image_path, vector_size, label):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (vector_size, vector_size))

    small = small.flatten()
    features = (small).tolist()
    features[-1] = str(label)

    return features


def write_csv(images_path='persian_number/'):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    final_path = {}
    database = []
    for f in files:
        tmp_list = [os.path.join(f, p) for p in sorted(os.listdir(f))]
        # tmp_list[-1] = f[8:]
        final_path[f[15:]] = (tmp_list)
        # print(f[15:])
        with open('file.csv', "w") as csv_file:
            for key, value in final_path.items():
                writer = csv.writer(csv_file, delimiter=',')
                for path in value:
                    writer.writerow(extract_features(path, 30, key))


write_csv()

# dosent work for these feature
# radius = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 2, 3, 4]
#
# my_parzen = parzen.ParzenClassifier(csv_file='file.csv', data=None, r=radius, weight=.90)
# radius_accuracy_dict, best_radius = my_parzen.kfold_validation(10)
#
# test, predicted = my_parzen.test(best_radius)
#
# ac, cm, re = my_parzen.report(test, predicted)
#
# print(re, "\n")
# print(cm, "\n")
# print(ac, "\n")
