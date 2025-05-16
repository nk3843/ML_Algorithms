"""
AdaBoost Classifier Implementation (SAMME Algorithm)

A custom implementation of the multi-class AdaBoost classifier using the SAMME algorithm.
This implementation follows a similar API to scikit-learn's AdaBoostClassifier.

Features:
- Supports any base classifier
- Handles multi-class classification
- Probability predictions
- Comprehensive error handling

Author: [Nikhil Kumar]
"""

import pandas as pd
import numpy as np
from copy import deepcopy
from typing import List, Any, Union, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaBoostClassifier:
    """
    Multi-class AdaBoost classifier using SAMME algorithm.
    
    Attributes:
        base_estimator: Base classifier class
        num_estimators: Number of estimator rounds
        estimators: List of fitted base estimators
        estimator_weights: Importance weight of each estimator
        classes_: List of unique class labels
    """

    def __init__(self, base_estimator: Any = None, num_estimators: int = 50):
        """
        Initialize the AdaBoost classifier.

        Args:
            base_estimator: The base classifier class
            num_estimators: Number of estimator rounds

        Raises:
            ValueError: If parameters are invalid
        """
        if num_estimators < 1:
            raise ValueError("num_estimators must be greater than 0")
        if base_estimator is None:
            raise ValueError("base_estimator cannot be None")

        self.base_estimator = base_estimator
        self.num_estimators = num_estimators
        self.estimators = [deepcopy(self.base_estimator) for _ in range(self.num_estimators)]
        self.estimator_weights = []
        self.classes_: List = []
        
        logger.info(f"Initialized AdaBoost with {num_estimators} estimators")

    def fit(self, features: pd.DataFrame, target_labels: Union[List, np.ndarray, pd.Series]) -> 'AdaBoostClassifier':
        """
        Fit the AdaBoost classifier.

        Args:
            features: Training data features
            target_labels: Target values

        Returns:
            self: The fitted classifier

        Raises:
            ValueError: If input data is invalid
        """
        try:
            if len(features) != len(target_labels):
                raise ValueError("Features and target_labels must have the same length")
            if features.empty:
                raise ValueError("Features DataFrame cannot be empty")

            self.classes_ = list(set(target_labels))
            num_classes = len(self.classes_)
            num_samples = len(target_labels)
            
            # Initialize sample weights uniformly
            sample_weights = np.array([1.0 / num_samples] * num_samples)
            labels = np.array(target_labels)
            
            logger.info(f"Starting training with {num_samples} samples, {num_classes} classes")

            for estimator_idx in range(self.num_estimators):
                # Sample with replacement according to weights
                sampled_indices = np.random.choice(
                    num_samples, 
                    num_samples, 
                    p=sample_weights
                )
                
                # Prepare sampled dataset
                sampled_features = features.iloc[sampled_indices].reset_index(drop=True)
                sampled_labels = labels[sampled_indices]
                
                # Train current estimator
                current_estimator = self.estimators[estimator_idx]
                current_estimator.fit(sampled_features, sampled_labels)
                
                # Make predictions and calculate error
                predictions = current_estimator.predict(features)
                incorrect_predictions = np.array(predictions) != labels
                weighted_error = np.sum(incorrect_predictions * sample_weights)
                
                # Ensure error is not too large
                while weighted_error >= (1 - 1.0 / num_classes):
                    logger.warning(f"Estimator {estimator_idx}: High error, retraining with new sample")
                    sample_weights = np.array([1.0 / num_samples] * num_samples)
                    sampled_indices = np.random.choice(num_samples, num_samples, p=sample_weights)
                    sampled_features = features.iloc[sampled_indices].reset_index(drop=True)
                    sampled_labels = labels[sampled_indices]
                    current_estimator.fit(sampled_features, sampled_labels)
                    predictions = current_estimator.predict(features)
                    incorrect_predictions = np.array(predictions) != labels
                    weighted_error = np.sum(incorrect_predictions * sample_weights)
                
                # Calculate estimator weight
                estimator_weight = np.log((1 - weighted_error) / weighted_error) + np.log(num_classes - 1)
                self.estimator_weights.append(estimator_weight)
                
                # Update sample weights
                sample_weights *= np.exp(estimator_weight * incorrect_predictions)
                sample_weights /= np.sum(sample_weights)  # Normalize weights
                
                logger.info(f"Estimator {estimator_idx}: error={weighted_error:.4f}, weight={estimator_weight:.4f}")

            # Normalize estimator weights
            self.estimator_weights = np.array(self.estimator_weights)
            self.estimator_weights /= np.sum(self.estimator_weights)
            
            logger.info("Successfully completed training")
            return self

        except Exception as e:
            logger.error(f"Error during fitting: {str(e)}")
            raise

    def predict(self, features: pd.DataFrame) -> List:
        """
        Predict class labels for samples in features.

        Args:
            features: Features to predict

        Returns:
            List of predicted labels
        """
        try:
            probabilities = self.predict_proba(features)
            predictions = [self.classes_[np.argmax(prob)] for prob in probabilities.to_numpy()]
            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities for samples in features.

        Args:
            features: Features to predict

        Returns:
            DataFrame with probability for each class
        """
        try:
            num_samples = len(features)
            probability_distributions = []

            for sample_idx in range(num_samples):
                class_probabilities = {}
                for class_label in self.classes_:
                    weighted_votes = 0
                    for estimator_idx in range(self.num_estimators):
                        prediction = self.estimators[estimator_idx].predict(features.iloc[[sample_idx]])
                        weighted_votes += (self.estimator_weights[estimator_idx] * 
                                        (prediction[0] == class_label))
                    class_probabilities[class_label] = weighted_votes
                probability_distributions.append(class_probabilities)

            return pd.DataFrame(probability_distributions, columns=self.classes_)

        except Exception as e:
            logger.error(f"Error during probability prediction: {str(e)}")
            raise
