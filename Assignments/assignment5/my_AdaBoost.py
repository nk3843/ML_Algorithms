import pandas as pd
import numpy as np
from copy import deepcopy
from pdb import set_trace


class my_AdaBoost:

    def __init__(self, base_estimator=None, n_estimators=50):
        # Multi-class Adaboost algorithm (SAMME)
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        # distinct class labels
        self.classes_ = list(set(list(y)))
        # number of distinct class labels
        k = len(self.classes_)
        # Number of rows/ dependent variables in training data
        n = len(y)
        # weight initialization for all the rows in training data
        w = np.array([1.0 / n] * n)
        # Total labels in training data
        labels = np.array(y)
        # alpha is importance of a classifier
        self.alpha = []
        for i in range(self.n_estimators):
            # Sample with replacement from X, with probability w
            sample = np.random.choice(n, n, p=w)
            # Train base classifier with sampled training data
            sampled = X.iloc[sample]
            sampled.index = range(len(sample))
            self.estimators[i].fit(sampled, labels[sample])
            predictions = self.estimators[i].predict(X)
            diffs = np.array(predictions) != y
            # Compute error rate and alpha for estimator i
            error = np.sum(diffs * w)
            while error >= (1 - 1.0 / k):
                w = np.array([1.0 / n] * n)
                sample = np.random.choice(n, n, p=w)
                # Train base classifier with sampled training data
                sampled = X.iloc[sample]
                sampled.index = range(len(sample))
                self.estimators[i].fit(sampled, labels[sample])
                predictions = self.estimators[i].predict(X)
                diffs = np.array(predictions) != y
                # Compute error rate and alpha for estimator i
                error = np.sum(diffs * w)
            # Compute alpha for estimator i (don't forget to use k for multi-class)
            alpha_var= (np.log((1 - error) / error)) + np.log(k - 1)

            self.alpha.append(alpha_var)

            # Update wi
            for i in range(len(diffs)):
                if  diffs[i]== True:
                    w[i] =  w[i] * np.exp(alpha_var)
            w = w / np.sum(w)

        # Normalize alpha
        self.alpha = self.alpha / np.sum(self.alpha)
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]

        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = []
        #res = []
        for i in range(len(X)):
            class_pred= {}
            for label in self.classes_:
                prob_sum = 0
                for j in range(self.n_estimators):
                    predictions= self.estimators[j].predict(X)
                    pred= 0
                    if predictions[i]==label:
                        pred=1
                    prob_sum += (self.alpha[j]* pred)
                class_pred[label] = prob_sum
            probs.append(class_pred)

        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs
