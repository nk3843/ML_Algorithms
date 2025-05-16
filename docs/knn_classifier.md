# K-Nearest Neighbors Classifier Implementation

A custom implementation of the K-Nearest Neighbors (KNN) classifier that supports multiple distance metrics. This implementation follows a similar API to scikit-learn's KNeighborsClassifier.

## Features

- Multiple distance metrics supported:
  - Euclidean distance
  - Manhattan distance
  - Minkowski distance
  - Cosine similarity
- Probability predictions
- Comprehensive error handling and logging
- Type hints and documentation
- Vectorized operations for improved performance

## Installation

Ensure you have the required dependencies:
```bash
pip install numpy pandas
```

## Usage

```python
from knn_classifier import KNNClassifier
import pandas as pd

# Load your data
train_features = pd.read_csv("data/Iris_train.csv")
X = train_features[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = train_features["Species"]

# Create and train the classifier
clf = KNNClassifier(
    num_neighbors=5,
    distance_metric="euclidean",
    minkowski_power=2  # Only used for minkowski distance
)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Calculate accuracy
accuracy = clf.calculate_accuracy(X_test, y_test)
```

## Parameters

### KNNClassifier

- `num_neighbors` (int, default=5): 
  - Number of neighbors to use for classification
  - Must be greater than 0

- `distance_metric` (str, default="euclidean"): 
  - Distance metric to use for neighbor calculation
  - Options: "euclidean", "manhattan", "minkowski", "cosine"

- `minkowski_power` (int, default=2): 
  - Power parameter for Minkowski metric
  - Only used when distance_metric="minkowski"
  - Must be greater than 0

## Methods

### fit(feature_matrix, target_labels)
Fits the KNN classifier with training data.
- `feature_matrix`: pandas DataFrame of training features
- `target_labels`: array-like of target values
- Returns: self

### predict(feature_matrix)
Predicts class labels for samples in feature matrix.
- `feature_matrix`: pandas DataFrame of features to predict
- Returns: list of predicted labels

### predict_proba(feature_matrix)
Predicts class probabilities for samples.
- `feature_matrix`: pandas DataFrame of features
- Returns: DataFrame with probability for each class

### calculate_accuracy(test_features, true_labels)
Calculates accuracy score on test data.
- `test_features`: pandas DataFrame of test features
- `true_labels`: array-like of true labels
- Returns: accuracy score (float between 0 and 1)

## Distance Metrics

### Euclidean Distance
- Standard straight-line distance between two points
- Formula: sqrt(sum((x - y)^2))

### Manhattan Distance
- Sum of absolute differences of coordinates
- Formula: sum(|x - y|)

### Minkowski Distance
- Generalization of Euclidean and Manhattan distance
- Formula: (sum(|x - y|^p))^(1/p)
- p=2 gives Euclidean, p=1 gives Manhattan

### Cosine Distance
- 1 minus the cosine similarity between points
- Formula: 1 - (dot(x,y) / (||x|| * ||y||))

## Example Output
