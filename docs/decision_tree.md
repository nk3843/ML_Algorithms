# Decision Tree Classifier Implementation

A custom implementation of a Decision Tree classifier that supports both Gini and Entropy criteria for splitting nodes. This implementation follows a similar API to scikit-learn's DecisionTreeClassifier.

## Features

- Supports both Gini and Entropy impurity measures
- Configurable maximum tree depth
- Minimum samples split threshold
- Minimum impurity decrease threshold
- Probability predictions
- Binary tree structure using dictionary representation
- Comprehensive error handling and logging

## Installation

Clone the repository and ensure you have the required dependencies:

```bash
pip install numpy pandas
```

## Usage

```python
from decision_tree import DecisionTree
import pandas as pd

# Load your data
data_train = pd.read_csv("data/Iris_train.csv")
X = data_train[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = data_train["Species"]

# Create and train the model
clf = DecisionTree(
    criterion="gini",
    max_depth=8,
    min_samples_split=2,
    min_impurity_decrease=0
)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

## Parameters

- `criterion` (str, default="gini"): 
  - The function to measure the quality of a split
  - Supported criteria: "gini" or "entropy"

- `max_depth` (int, default=8): 
  - Maximum depth of the tree
  - Limits the size of the tree to prevent overfitting

- `min_samples_split` (int, default=2): 
  - Minimum number of samples required to split a node
  - Controls the minimum size of nodes

- `min_impurity_decrease` (float, default=0): 
  - Minimum required decrease in impurity for splitting
  - Controls whether a split should be made based on improvement

## Methods

### fit(X, y)
Builds the decision tree from training data.
- `X`: pandas DataFrame of features
- `y`: array-like of target values
- Returns: self

### predict(X)
Predicts class labels for samples in X.
- `X`: pandas DataFrame of features
- Returns: list of predicted labels

### predict_proba(X)
Predicts class probabilities for samples in X.
- `X`: pandas DataFrame of features
- Returns: pandas DataFrame with probability for each class

## Example Output

```python
# Sample predictions with probabilities
Iris-setosa     1.000000
Iris-versicolor 1.000000
Iris-virginica  1.000000
```

## Implementation Details

The tree is implemented using a dictionary structure where:
- Node 0 is the root
- For any node i:
  - Left child = 2i + 1
  - Right child = 2i + 2
- Nodes contain either:
  - Tuple of (feature, split_value) for internal nodes
  - Counter object with class distributions for leaf nodes

## Error Handling

The implementation includes comprehensive error handling and logging:
- Input validation
- Tree structure verification
- Informative error messages
- Logging of tree construction and prediction processes

## Author

[Nikhil Kumar]

## License

