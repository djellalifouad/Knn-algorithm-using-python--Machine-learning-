# import numpy library to calculate the distance

import numpy as np
import pandas as pd
import sklearn.model_selection as sklearn
from scipy.spatial import distance


class Point:
    distance = 0
    type = ""
    index = 0


    def __str__(self):
     return " distance: % s, " \
           "type: % s index %s" % (self.distance, self.type,self.index)

    def __repr__(self):
        return str(self)


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)

def biggest(a, y, z):
    Max = a
    if y > Max:
        Max = y
    if z > Max:
        Max = z
        if y > z:
            Max = y
    return Max
# sepal length
# sepal width
# petal length
# petal width
Object = lambda **kwargs: type("Object", (), kwargs)
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
iris = pd.read_csv('./iris.csv', names=names, header=None)
X = iris.iloc[:, 0:4]
Y = iris.iloc[:, -1]
# now we need to get our training set and testing set
X_train, X_test, Y_train, Y_test = sklearn.train_test_split(X, Y, test_size=5)


for i in range(len(X_test)):
    print(X_test.iloc[i].name)
    dist = []
    for j in range(len(X_train)):

       # print(X_train.iloc[j:j + 1, 0:4].values)
        dst = distance.euclidean(X_test.iloc[i:i + 1, 0:4].values, X_train.iloc[j:j + 1, 0:4].values)

        point = Point()
        point.distance = dst
        point.type = Y_train.iloc[j]
        point.index  = X_train.iloc[j].name
        dist.append(point)
    dist.sort(key=lambda x: x.distance, reverse=False)
    dist = dist[:10]

    print(dist)
    Iris_versicolor = 0
    Iris_setosa = 0
    Iris_virginica = 0

    for i in dist:
        if i.type == 'Iris-setosa':
            Iris_setosa += 1
        elif i.type == 'Iris-virginica':
            Iris_virginica += 1
        else:
            Iris_versicolor += 1

    a = biggest(Iris_versicolor, Iris_virginica, Iris_setosa)
    if (a == Iris_virginica):
        print("Iris_virginica")
    if (a == Iris_versicolor):
        print("Iris_versicolor")
    if (a == Iris_setosa):
        print("Iris_setosa")






