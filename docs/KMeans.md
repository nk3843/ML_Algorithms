# KMeans Clustering Implementation

## Overview
This implementation provides a production-ready version of the KMeans clustering algorithm. KMeans is an unsupervised learning algorithm that partitions data into K clusters, where each data point belongs to the cluster with the nearest mean.

## Features
- Multiple initialization methods (k-means++ and random)
- Configurable number of clusters
- Multiple runs with best result selection
- Convergence criteria with tolerance
- Distance-based clustering
- Support for both pandas DataFrame and numpy array inputs
- Comprehensive error handling and input validation
- Progress logging and debugging support

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ML_Algorithms.git

# Navigate to the project directory
cd ML_Algorithms

# Install dependencies
pip install -r requirements.txt
```

## Usage
```python
from KMeans import KMeans
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Initialize the KMeans classifier
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    max_iter=300,
    tol=1e-4,
    random_state=42
)

# Fit the model
kmeans.fit(data)

# Get cluster assignments
predictions = kmeans.predict(data)

# Get distances to cluster centers
distances = kmeans.transform(data)
```

## Parameters

### KMeans Class
- `n_clusters` (int, default=8): Number of clusters to form
- `init` (str, default="k-means++"): Method for initialization
  - "k-means++": Uses k-means++ algorithm for initialization
  - "random": Randomly selects k points as initial centers
- `n_init` (int, default=10): Number of times the algorithm will run
- `max_iter` (int, default=300): Maximum number of iterations
- `tol` (float, default=1e-4): Convergence tolerance
- `random_state` (int, optional): Random state for reproducibility

## Methods

### fit(X)
Fits the KMeans clustering algorithm to the data.

**Parameters:**
- `X`: Training data (pandas DataFrame or numpy array)

**Returns:**
- self: The fitted KMeans instance

### predict(X)
Predicts the closest cluster each sample belongs to.

**Parameters:**
- `X`: Data to predict (pandas DataFrame or numpy array)

**Returns:**
- numpy.ndarray: Cluster assignments

### transform(X)
Transforms data to cluster-distance space.

**Parameters:**
- `X`: Data to transform (pandas DataFrame or numpy array)

**Returns:**
- numpy.ndarray: Distances to each cluster center

### fit_predict(X)
Computes cluster centers and predicts cluster index for each sample.

**Parameters:**
- `X`: Training data (pandas DataFrame or numpy array)

**Returns:**
- numpy.ndarray: Cluster assignments

### fit_transform(X)
Computes clustering and transforms X to cluster-distance space.

**Parameters:**
- `X`: Training data (pandas DataFrame or numpy array)

**Returns:**
- numpy.ndarray: Distances to each cluster center

## Attributes

### cluster_centers_
The coordinates of cluster centers.

### inertia_
Sum of squared distances of samples to their closest cluster center.

### n_iter_
Number of iterations run in the best run.

## Example
```python
import pandas as pd
from KMeans import KMeans

# Load Iris dataset
data = pd.read_csv("iris.csv")
features = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(features)

# Get predictions
predictions = kmeans.predict(features)

# Print results
print("Cluster centers:")
print(kmeans.cluster_centers_)
print("\nInertia:", kmeans.inertia_)
print("\nNumber of iterations:", kmeans.n_iter_)
```

## Error Handling
The implementation includes comprehensive error handling for:
- Invalid parameter values
- Missing or invalid input data
- Unfitted model usage
- Data type mismatches
- Empty clusters

## Performance Considerations
- The algorithm uses vectorized operations for better performance
- Multiple runs are parallelized when possible
- Early stopping is implemented for faster convergence
- Memory-efficient distance calculations

## Dependencies
- numpy
- pandas
- logging

## Author
[Nikhil Kumar]

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- scikit-learn for inspiration
- UCI Machine Learning Repository for the Iris dataset
