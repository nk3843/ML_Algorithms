# Machine Learning Algorithms Implementation

A comprehensive collection of machine learning algorithms implemented from scratch using Python, Pandas, and NumPy. Each implementation is benchmarked against the Scikit-learn library for validation and performance comparison.

## Implemented Algorithms

### 1. Classification Algorithms
- **Naive Bayes**
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
  - [Documentation](naive_bayes_implementation/naive_bayes.md)

- **K-Nearest Neighbors (KNN)**
  - Custom distance metrics
  - Weighted voting
  - K-fold cross-validation
  - [Documentation](knn_implementation/knn.md)

- **Decision Tree**
  - Information gain
  - Gini impurity
  - Tree visualization
  - [Documentation](decision_tree_implementation/decision_tree.md)

- **AdaBoost**
  - Custom base estimators
  - Weighted voting
  - Error analysis
  - [Documentation](adaboost_implementation/adaboost.md)

### 2. Clustering Algorithms
- **K-Means**
  - Custom initialization
  - Elbow method
  - Cluster visualization
  - [Documentation](KMeans_implementation/KMeans.md)

### 3. Dimensionality Reduction
- **Principal Component Analysis (PCA)**
  - Variance analysis
  - Component selection
  - Feature importance
  - [Documentation](PCA_implementation/pca.md)

### 4. Evaluation Metrics
- **Classification Metrics**
  - Accuracy, Precision, Recall
  - F1 Score
  - AUC-ROC
  - Confusion Matrix
  - [Documentation](evaluation_implementation/evaluation.md)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ML_Algorithms
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn (for benchmarking)

## Usage

Each algorithm implementation includes:
- Implementation code
- Test scripts
- Documentation
- Example usage

### Example Usage

```python
# Example using Naive Bayes
from naive_bayes_implementation.naive_bayes import NaiveBayes

# Initialize and train
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Make predictions
predictions = nb.predict(X_test)
```

## Features

- **From Scratch Implementation**: All algorithms are implemented from first principles
- **Comprehensive Testing**: Each implementation includes test scripts
- **Scikit-learn Benchmarking**: Performance comparison with Scikit-learn
- **Detailed Documentation**: Each algorithm has its own documentation
- **Error Handling**: Robust error handling and input validation
- **Logging**: Comprehensive logging for debugging and monitoring

## Testing

Run tests for specific implementation:
```bash
python naive_bayes_implementation/test_naive_bayes.py
python knn_implementation/test_knn.py
python decision_tree_implementation/test_decision_tree.py
python adaboost_implementation/test_adaboost.py
python KMeans_implementation/test_KMeans.py
python PCA_implementation/test_preprocess.py
python evaluation_implementation/test_ClassificationMetrics.py
```

## License

[Your License]

## Author

[Nikhil Kumar]

## Acknowledgments

- Scikit-learn for inspiration and benchmarking
- Iris dataset for testing


