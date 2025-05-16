# Naive Bayes Classifier Implementation

A custom implementation of a Naive Bayes classifier with Laplace smoothing, designed for categorical features. This implementation follows a similar API to scikit-learn's NaiveBayesClassifier.

## Features

- Laplace (additive) smoothing
- Handles categorical features
- Probability predictions
- Comprehensive error handling and logging
- Production-ready implementation

## Installation

Ensure you have the required dependencies:
```bash
pip install numpy pandas
```

## Usage

```python
from naive_bayes import NaiveBayes
import pandas as pd

# Load your data
train_features = pd.read_csv("data/audiology_train", header=None)
X = train_features[range(69)]  # Features are columns 0-68
y = train_features[70]         # Target is column 70

# Create and train the classifier
clf = NaiveBayes(alpha=1.0)  # alpha is the smoothing parameter
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

## Parameters

- `alpha` (float, default=1.0): 
  - Smoothing factor for Laplace smoothing
  - P(x_i = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
  - where n(i) is the number of unique values of feature i

## Methods

### fit(features, target_labels)
Trains the Naive Bayes classifier.
- `features`: pandas DataFrame of categorical features
- `target_labels`: array-like of target values
- Returns: self

### predict(features)
Predicts class labels for samples.
- `features`: pandas DataFrame of features
- Returns: list of predicted labels

### predict_proba(features)
Predicts class probabilities for samples.
- `features`: pandas DataFrame of features
- Returns: pandas DataFrame with probability for each class

## Example Output

```python
# Sample predictions with confidence scores
cochlear_age                   0.999408
normal_ear                     0.990685
mixed_cochlear_unk_fixation    0.832907

# Prediction Summary
Average confidence: 0.8616
Class distribution:
cochlear_age                    13 predictions
cochlear_poss_noise             2 predictions
cochlear_unknown                1 predictions
mixed_cochlear_unk_fixation     5 predictions
normal_ear                      4 predictions
cochlear_age_and_noise          1 predictions
```

## Implementation Details

The classifier implements the Naive Bayes algorithm with:
- Laplace smoothing for handling zero probabilities
- Logarithmic probability calculations for numerical stability
- Efficient storage of conditional probabilities
- Comprehensive error handling and input validation

## Error Handling

The implementation includes:
- Input validation
- Proper error messages
- Logging for debugging and monitoring
- Exception handling for robustness

## Testing

Run the test script to verify the implementation:
```bash
python test_naive_bayes.py
```

The test script includes:
- Data loading and validation
- Model training and prediction
- Performance evaluation
- Detailed output formatting

## Author

[Nikhil Kumar]

## License

