# Classification Metrics Implementation

A comprehensive implementation of classification evaluation metrics for machine learning models. This implementation supports both binary and multi-class classification tasks.

## Features

- **Multiple Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC-ROC
  - Confusion Matrix

- **Support for Different Averaging Methods**:
  - Macro averaging
  - Micro averaging
  - Weighted averaging

- **Comprehensive Class-wise Metrics**:
  - Per-class precision, recall, and F1 scores
  - Per-class AUC-ROC scores
  - Detailed confusion matrix analysis

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ML_Algorithms
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from my_evaluation import ClassificationMetrics

# Initialize with predictions and actual labels
metrics = ClassificationMetrics(
    predictions=y_pred,
    actuals=y_true,
    pred_proba=probabilities_df  # Optional
)

# Get overall accuracy
accuracy = metrics.get_accuracy()

# Get precision for a specific class
precision = metrics.get_precision(target="class_name")

# Get recall with macro averaging
recall = metrics.get_recall(average="macro")

# Get F1 score
f1 = metrics.get_f1(target="class_name", average="weighted")

# Get AUC-ROC score for a specific class
auc = metrics.get_auc(target="class_name")

# Get comprehensive summary
summary = metrics.get_summary()
```

### Example with Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier
from my_evaluation import ClassificationMetrics

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Get predictions and probabilities
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Create probability DataFrame
prob_df = pd.DataFrame(
    probabilities,
    columns=clf.classes_,
    index=X_test.index
)

# Initialize metrics
metrics = ClassificationMetrics(predictions, y_test, prob_df)

# Get evaluation summary
results = metrics.get_summary()
```

## Output Format

The evaluation summary includes:

```python
{
    "accuracy": float,
    "per_class": {
        "class_name": {
            "precision": float,
            "recall": float,
            "f1": float,
            "auc": float
        }
    },
    "averages": {
        "macro": {
            "precision": float,
            "recall": float,
            "f1": float
        },
        "micro": {
            "precision": float,
            "recall": float,
            "f1": float
        },
        "weighted": {
            "precision": float,
            "recall": float,
            "f1": float
        }
    }
}
```

## Features

### 1. Accuracy
- Overall classification accuracy
- Handles both binary and multi-class cases

### 2. Precision
- Supports per-class precision
- Multiple averaging methods (macro, micro, weighted)
- Handles edge cases (zero division)

### 3. Recall
- Per-class recall calculation
- Multiple averaging methods
- Robust to class imbalance

### 4. F1 Score
- Harmonic mean of precision and recall
- Supports all averaging methods
- Handles edge cases

### 5. AUC-ROC
- Per-class AUC calculation
- Requires prediction probabilities
- Handles multi-class cases

## Error Handling

The implementation includes comprehensive error handling:
- Input validation
- Type checking
- Edge case handling
- Informative error messages

## Logging

Built-in logging support:
- Detailed progress tracking
- Error reporting
- Performance metrics logging

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn (for example usage)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License]

## Author

[Nikhil Kumar]

## Acknowledgments

- Scikit-learn for inspiration
- Iris dataset for testing
