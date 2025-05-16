"""
Naive Bayes Classifier Implementation

A custom implementation of a Naive Bayes classifier with Laplace smoothing.
This implementation is designed for categorical features and follows a
similar API to scikit-learn's NaiveBayesClassifier.

Features:
- Laplace (additive) smoothing
- Handles categorical features
- Probability predictions
- Comprehensive error handling

Author: [Nikhil Kumar]
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Union, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaiveBayes:
    """
    Naive Bayes classifier for categorical features.
    
    Implements the Naive Bayes algorithm with Laplace smoothing for
    handling categorical features. Calculates P(x_i|y) using:
    P(x_i = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
    where n(i) is the number of unique values of feature i.

    Attributes:
        alpha (float): Smoothing factor (default=1 for Laplace smoothing)
        classes_ (list): List of unique class labels
        prior_probabilities (Counter): Prior probabilities P(y) for each class
        conditional_probabilities (dict): P(x_i|y) for each feature-value pair
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize the Naive Bayes classifier.

        Args:
            alpha: Smoothing factor for Laplace smoothing (default=1.0)
        
        Raises:
            ValueError: If alpha is negative
        """
        if alpha < 0:
            raise ValueError("Alpha must be non-negative")
        
        self.alpha = alpha
        self.classes_: List = []
        self.prior_probabilities: Counter = Counter()
        self.conditional_probabilities: Dict = {}
        
        logger.info(f"Initialized NaiveBayes with alpha={alpha}")

    def fit(self, features: pd.DataFrame, target_labels: Union[List, np.ndarray, pd.Series]) -> 'NaiveBayes':
        """
        Fit the Naive Bayes classifier.

        Args:
            features: Training data features (categorical)
            target_labels: Target values

        Returns:
            self: Returns the instance itself

        Raises:
            ValueError: If input data is invalid
        """
        try:
            # Validate input
            if len(features) != len(target_labels):
                raise ValueError("Features and target_labels must have the same length")
            if features.empty:
                raise ValueError("Features DataFrame cannot be empty")

            self.classes_ = list(set(target_labels))
            target_list = list(target_labels)
            self.prior_probabilities = Counter(target_labels)
            self.conditional_probabilities = {}

            # Calculate unique values for each feature
            unique_feature_values = {
                feature_name: set(features[feature_name]) 
                for feature_name in features.columns
            }

            # Calculate conditional probabilities for each class
            for class_label in self.classes_:
                self.conditional_probabilities[class_label] = {}
                
                # Get indices for current class
                class_sample_indices = [
                    idx for idx, label in enumerate(target_list) 
                    if label == class_label
                ]
                
                # Calculate probabilities for each feature
                for feature_name in features.columns:
                    self.conditional_probabilities[class_label][feature_name] = {}
                    
                    # Get feature values for current class
                    feature_value_counts = Counter(
                        features.iloc[class_sample_indices][feature_name]
                    )
                    
                    # Calculate smoothed probabilities for all possible values
                    num_samples = len(class_sample_indices)
                    num_categories = len(unique_feature_values[feature_name])
                    
                    for feature_value in unique_feature_values[feature_name]:
                        count = feature_value_counts.get(feature_value, 0)
                        # Apply Laplace smoothing
                        smoothed_probability = (count + self.alpha) / (num_samples + self.alpha * num_categories)
                        self.conditional_probabilities[class_label][feature_name][feature_value] = smoothed_probability

            logger.info("Successfully fitted NaiveBayes classifier")
            return self

        except Exception as e:
            logger.error(f"Error during fitting: {str(e)}")
            raise

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities for features.

        Args:
            features: Features to predict

        Returns:
            DataFrame with probability for each class

        Raises:
            ValueError: If model hasn't been fitted or input is invalid
        """
        if not self.classes_:
            raise ValueError("Fit the model before making predictions")

        try:
            # Initialize probability dictionary with prior probabilities
            class_probabilities = {
                class_label: pd.Series(self.prior_probabilities[class_label], index=features.index) 
                for class_label in self.classes_
            }
            
            # Calculate probabilities
            for class_label in self.classes_:
                for feature_name in features.columns:
                    # Multiply by conditional probabilities
                    class_probabilities[class_label] *= features[feature_name].apply(
                        lambda value: self.conditional_probabilities[class_label][feature_name].get(value, 1.0)
                    )

            # Convert to DataFrame and normalize
            probability_dataframe = pd.DataFrame(class_probabilities, columns=self.classes_)
            row_sums = probability_dataframe.sum(axis=1)
            normalized_probabilities = probability_dataframe.div(row_sums, axis=0)

            return normalized_probabilities

        except Exception as e:
            logger.error(f"Error during probability prediction: {str(e)}")
            raise

    def predict(self, features: pd.DataFrame) -> List:
        """
        Predict class labels for features.

        Args:
            features: Features to predict

        Returns:
            List of predicted labels

        Raises:
            ValueError: If model hasn't been fitted or input is invalid
        """
        try:
            probabilities = self.predict_proba(features)
            predicted_labels = [
                self.classes_[np.argmax(prob)] 
                for prob in probabilities.to_numpy()
            ]
            return predicted_labels

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def score(self, features: pd.DataFrame, true_labels: Union[List, np.ndarray, pd.Series]) -> float:
        """
        Calculate the accuracy score on the given test data and labels.

        Args:
            features: Test features
            true_labels: True labels

        Returns:
            float: Accuracy score (0.0 to 1.0)
        """
        try:
            predicted_labels = self.predict(features)
            correct_predictions = sum(1 for pred, true in zip(predicted_labels, true_labels) if pred == true)
            accuracy_score = correct_predictions / len(true_labels)
            logger.info(f"Model accuracy: {accuracy_score:.4f}")
            return accuracy_score

        except Exception as e:
            logger.error(f"Error during scoring: {str(e)}")
            raise





