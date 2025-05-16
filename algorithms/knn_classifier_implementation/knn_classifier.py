"""
K-Nearest Neighbors Classifier Implementation

A custom implementation of KNN classifier that supports multiple distance metrics.
This implementation follows a similar API to scikit-learn's KNeighborsClassifier.

Features:
- Multiple distance metrics (minkowski, euclidean, manhattan, cosine)
- Probability predictions
- Comprehensive error handling

Author: [Nikhil Kumar]
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Union, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KNNClassifier:
    """
    K-Nearest Neighbors classifier supporting multiple distance metrics.
    
    Attributes:
        num_neighbors (int): Number of neighbors to use
        distance_metric (str): Distance metric to use
        minkowski_power (int): Power parameter for Minkowski metric
        unique_classes (list): List of unique class labels
        training_features (pd.DataFrame): Training features
        training_labels (array-like): Training labels
    """

    VALID_DISTANCE_METRICS = {"minkowski", "euclidean", "manhattan", "cosine"}

    def __init__(self, num_neighbors: int = 5, distance_metric: str = "euclidean", minkowski_power: int = 2):
        """
        Initialize the KNN classifier.

        Args:
            num_neighbors: Number of neighbors to use (default=5)
            distance_metric: Distance metric {"minkowski", "euclidean", "manhattan", "cosine"}
            minkowski_power: Power parameter for Minkowski metric (default=2)

        Raises:
            ValueError: If parameters are invalid
        """
        if num_neighbors < 1:
            raise ValueError("Number of neighbors must be greater than 0")
        if distance_metric not in self.VALID_DISTANCE_METRICS:
            raise ValueError(f"Distance metric must be one of {self.VALID_DISTANCE_METRICS}")
        if minkowski_power < 1 and distance_metric == "minkowski":
            raise ValueError("Minkowski power must be greater than 0")

        self.num_neighbors = int(num_neighbors)
        self.distance_metric = distance_metric
        self.minkowski_power = minkowski_power
        self.unique_classes: List = []
        self.training_features: Optional[pd.DataFrame] = None
        self.training_labels: Optional[np.ndarray] = None
        
        logger.info(f"Initialized KNN with {num_neighbors} neighbors, {distance_metric} metric")

    def fit(self, feature_matrix: pd.DataFrame, target_labels: Union[List, np.ndarray, pd.Series]) -> 'KNNClassifier':
        """
        Fit the KNN classifier.

        Args:
            feature_matrix: Training data features
            target_labels: Target values

        Returns:
            self: The fitted classifier

        Raises:
            ValueError: If input data is invalid
        """
        try:
            if len(feature_matrix) != len(target_labels):
                raise ValueError("Feature matrix and target labels must have the same length")
            if feature_matrix.empty:
                raise ValueError("Feature matrix cannot be empty")

            self.unique_classes = list(set(target_labels))
            self.training_features = feature_matrix
            self.training_labels = np.array(target_labels)
            
            logger.info(f"Fitted KNN with {len(feature_matrix)} samples, {len(self.unique_classes)} classes")
            return self

        except Exception as e:
            logger.error(f"Error during fitting: {str(e)}")
            raise

    def calculate_distances(self, query_point: np.ndarray) -> np.ndarray:
        """
        Calculate distances between a query point and all training points.

        Args:
            query_point: Single data point to find neighbors for

        Returns:
            np.ndarray: Array of distances to all training points

        Raises:
            ValueError: If distance metric calculation fails
        """
        try:
            if self.training_features is None:
                raise ValueError("Model not fitted. Call fit before calculating distances.")

            if self.distance_metric == "minkowski":
                point_differences = np.abs(self.training_features - query_point) ** self.minkowski_power
                distances = np.sum(point_differences, axis=1) ** (1 / self.minkowski_power)
            
            elif self.distance_metric == "euclidean":
                squared_differences = (self.training_features - query_point) ** 2
                distances = np.sqrt(np.sum(squared_differences, axis=1))
            
            elif self.distance_metric == "manhattan":
                absolute_differences = np.abs(self.training_features - query_point)
                distances = np.sum(absolute_differences, axis=1)
            
            elif self.distance_metric == "cosine":
                query_point_norm = np.sqrt(np.sum(query_point ** 2))
                training_points_norm = np.sqrt(np.sum(self.training_features ** 2, axis=1))
                dot_products = np.sum(self.training_features * query_point, axis=1)
                cosine_similarities = dot_products / (training_points_norm * query_point_norm)
                distances = 1 - cosine_similarities
            
            return distances

        except Exception as e:
            logger.error(f"Error calculating distances: {str(e)}")
            raise

    def find_nearest_neighbors(self, query_point: np.ndarray) -> Counter:
        """
        Find k nearest neighbors for a query point.

        Args:
            query_point: Single data point to find neighbors for

        Returns:
            Counter: Frequency distribution of neighbor classes
        """
        try:
            point_distances = self.calculate_distances(query_point)
            nearest_neighbor_indices = np.argpartition(point_distances, self.num_neighbors)[:self.num_neighbors]
            neighbor_labels = self.training_labels[nearest_neighbor_indices]
            return Counter(neighbor_labels)

        except Exception as e:
            logger.error(f"Error finding neighbors: {str(e)}")
            raise

    def predict(self, feature_matrix: pd.DataFrame) -> List:
        """
        Predict class labels for samples in feature matrix.

        Args:
            feature_matrix: Features to predict

        Returns:
            List of predicted labels
        """
        try:
            class_probabilities = self.predict_proba(feature_matrix)
            predicted_labels = [
                self.unique_classes[np.argmax(prob_distribution)] 
                for prob_distribution in class_probabilities.to_numpy()
            ]
            return predicted_labels

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities for samples in feature matrix.

        Args:
            feature_matrix: Features to predict

        Returns:
            DataFrame with probability distribution for each class

        Raises:
            ValueError: If input features don't match training features
        """
        try:
            if not set(feature_matrix.columns) == set(self.training_features.columns):
                raise ValueError("Input features don't match training features")

            probability_distributions = []
            for query_point in feature_matrix.to_numpy():
                neighbor_distribution = self.find_nearest_neighbors(query_point)
                class_probabilities = {
                    class_label: neighbor_distribution[class_label] / self.num_neighbors 
                    for class_label in self.unique_classes
                }
                probability_distributions.append(class_probabilities)

            return pd.DataFrame(probability_distributions, columns=self.unique_classes)

        except Exception as e:
            logger.error(f"Error during probability prediction: {str(e)}")
            raise

    def calculate_accuracy(self, test_features: pd.DataFrame, true_labels: Union[List, np.ndarray, pd.Series]) -> float:
        """
        Calculate the accuracy score on given test data and labels.

        Args:
            test_features: Test feature matrix
            true_labels: True target labels

        Returns:
            float: Accuracy score (0.0 to 1.0)
        """
        try:
            predicted_labels = self.predict(test_features)
            correct_predictions = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
            accuracy_score = correct_predictions / len(true_labels)
            logger.info(f"Model accuracy: {accuracy_score:.4f}")
            return accuracy_score

        except Exception as e:
            logger.error(f"Error during accuracy calculation: {str(e)}")
            raise
