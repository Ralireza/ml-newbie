import cv2
import os
import csv
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

def write_csv(images_path='persian_number/', pickled_db_path="features.pck"):
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
                    writer.writerow(extract_features(path, 30, key))


write_csv()


#
myknn = KNN.knn(csv_file='file.csv', data=None, k=2, weight=.50)
test, predicted = myknn.test_data()
ac, cm, re = myknn.report(test, predicted)
print(ac)
print(cm)
print(re)
