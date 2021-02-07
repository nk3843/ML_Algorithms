import pandas as pd
import numpy as np
from collections import Counter


class my_KNN:

    def __init__(self, n_neighbors=5, metric="euclidean", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # print(self.classes_)
        self.X = X
        # print(self.X)
        self.y = y
        # print(self.y)
        return

    def dist(self, x):
        # Calculate distances of training data to a single input data point (distances from self.X to x)
        # Output np.array([distances to x])
        dist = []

        if self.metric == "minkowski":
            #print(self.X.columns)
            for i in range(len(self.X)):
                distances = 0
                for j in range(len(self.X.columns)):
                    distances += (self.X.loc[i][j] - x[j]) ** self.p
                distances = distances ** (1 / self.p)
                dist.append(distances)

        elif self.metric == "euclidean":
            for i in range(len(self.X)):
                distances = 0
                for j in range(len(self.X.columns)):
                    distances += (self.X.loc[i][j] - x[j]) ** 2
                distances = distances ** 0.5

                dist.append(distances)

        elif self.metric == "manhattan":
            for i in range(len(self.X)):
                distances = 0
                for j in range(len(self.X.columns)):
                    distances += (self.X.loc[i][j] - x[j])
                dist.append(distances)


        elif self.metric == "cosine":
            testsum = 0

            for j in range(len(self.X.columns)):
                testsum += x[j] ** 2
            testsum = testsum ** 0.5
            trainsum = []
            for i in range(len(self.X)):
                train = 0
                for j in range(len(self.X.columns)):
                    train += self.X.loc[i][j] ** 2
                train = train ** 0.5
                trainsum.append(train)

            for i in range(len(self.X)):
                distances = 0
                for j in range(len(self.X.columns)):
                    distances += (self.X.loc[i][j] * x[j])

                distances = distances / (testsum * trainsum[i])
                distances = 1 - distances

                dist.append(distances)


        else:
            raise Exception("Unknown criterion.")

        return dist

    def k_neighbors(self, x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        output = []
        distances = self.dist(x)

        sorted_dist = np.array(distances).argsort()[:self.n_neighbors]

        res = []

        for i in sorted_dist:
            res.append(self.y[i])

        label = Counter(res)
        return label

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            probs.append({key: neighbors[key] / float(self.n_neighbors) for key in self.classes_})
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs
