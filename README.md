# Machine Learning Algorithms Implementation

A comprehensive collection of machine learning algorithms implemented from scratch using Python, Pandas, and NumPy. The project includes both the core algorithm implementations and a web application for interactive model training and prediction.

## Project Structure

```
ML_Algorithms/
├── algorithms/           # Core algorithm implementations
│   ├── naive_bayes_implementation/
│   ├── knn_implementation/
│   ├── decision_tree_implementation/
│   ├── adaboost_implementation/
│   ├── KMeans_implementation/
│   ├── PCA_implementation/
│   └── evaluation_implementation/
├── web_app/             # Flask web application
│   ├── templates/
│   └── static/
├── data/                # Dataset storage
├── tests/              # Test files
├── docs/               # Documentation
└── requirements.txt    # Project dependencies
```

## Implemented Algorithms

### 1. Classification Algorithms
- **Naive Bayes**
  - Gaussian Naive Bayes
  - Multinomial Naive Bayes
  - Bernoulli Naive Bayes
  - [Documentation](docs/naive_bayes.md)

- **K-Nearest Neighbors (KNN)**
  - Custom distance metrics
  - Weighted voting
  - K-fold cross-validation
  - [Documentation](docs/knn.md)

- **Decision Tree**
  - Information gain
  - Gini impurity
  - Tree visualization
  - [Documentation](docs/decision_tree.md)

- **AdaBoost**
  - Custom base estimators
  - Weighted voting
  - Error analysis
  - [Documentation](docs/adaboost.md)

### 2. Clustering Algorithms
- **K-Means**
  - Custom initialization
  - Elbow method
  - Cluster visualization
  - [Documentation](docs/kmeans.md)

### 3. Dimensionality Reduction
- **Principal Component Analysis (PCA)**
  - Variance analysis
  - Component selection
  - Feature importance
  - [Documentation](docs/pca.md)

### 4. Evaluation Metrics
- **Classification Metrics**
  - Accuracy, Precision, Recall
  - F1 Score
  - AUC-ROC
  - Confusion Matrix
  - [Documentation](docs/evaluation.md)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nk3843/ML_Algorithms.git
cd ML_Algorithms
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application
Run the Flask web application:
```bash
cd web_app
python app.py
```
The web interface will be available at `http://localhost:5000`

### Command Line Usage
Each algorithm can be used programmatically:

```python
# Example using Naive Bayes
from algorithms.naive_bayes_implementation.naive_bayes import NaiveBayes

# Initialize and train
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Make predictions
predictions = nb.predict(X_test)
```

## Features

- **From Scratch Implementation**: All algorithms are implemented from first principles
- **Interactive Web Interface**: Flask-based web application for model training and prediction
- **Comprehensive Testing**: Each implementation includes test scripts
- **Scikit-learn Benchmarking**: Performance comparison with Scikit-learn
- **Detailed Documentation**: Each algorithm has its own documentation
- **Error Handling**: Robust error handling and input validation
- **Logging**: Comprehensive logging for debugging and monitoring

## Testing

Run tests for specific implementation:
```bash
python -m tests.test_naive_bayes
python -m tests.test_knn
python -m tests.test_decision_tree
python -m tests.test_adaboost
python -m tests.test_KMeans
python -m tests.test_preprocess
python -m tests.test_ClassificationMetrics
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Nikhil Kumar

## Acknowledgments

- Scikit-learn for inspiration and benchmarking
- Iris dataset for testing
- Flask for the web application framework


