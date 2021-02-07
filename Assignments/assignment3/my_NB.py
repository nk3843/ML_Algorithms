import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        # list of classes for this model
        self.classes_ = list(set(list(y)))
        classes = list(y)
        #print(self.classes_)
        # for calculation of P(y)
        self.P_y = Counter(y)
        #print(self.P_y)
        # self.P[yj][Xi][xi] = P(xi|yj) where Xi is the feature name and xi is the feature value, yj is a specific class label
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        self.P = {}
        #print(len(X))
        label_idx = {}
        all_values = {}
        for col in X.keys():
            all_values[col] = set(X[col])
        for label in self.classes_:
            self.P[label] = {}
            idx = []
            # idx has list of indices with given class name
            for i in range(len(classes)):
                if classes[i] == label:
                    idx.append(i)
            #print(X.keys())
            for x in X.keys():
                self.P[label][x] = {}
                res = []
                for index in idx:
                    #print(X.iloc[index])
                    res.append(X.iloc[index][x])
                cnt = Counter(res)
                # for key in cnt:
                for value in all_values[x]:
                    if cnt[value] is None:
                        cnt[value] = 0
                    self.P[label][x][value] = (cnt[value]+self.alpha) / (len(res)+self.alpha*len(all_values[x]))
                    #print(cnt[key]/len(res))
        return None

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions





