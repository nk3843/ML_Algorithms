"""
Test script for KNN Classifier implementation.
Tests the classifier on the Iris dataset.

Author: [Nikhil Kumar]
"""

import pandas as pd
from algorithms.knn_classifier_implementation.knn_classifier import KNNClassifier
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path: Path, is_training: bool = True) -> tuple:
    """
    Load and prepare the Iris dataset.

    Args:
        data_path: Path to the data file
        is_training: Whether loading training or test data

    Returns:
        tuple: (features DataFrame, target labels Series) for training data
               (features DataFrame) for test data
    """
    try:
        dataset_type = "training" if is_training else "test"
        logger.info(f"Loading {dataset_type} data from {data_path}")
        
        data = pd.read_csv(data_path)
        
        # Separate features and target
        feature_columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
        features = data[feature_columns]
        
        if is_training:
            target_labels = data["Species"]
            return features, target_labels
        return features

    except Exception as e:
        logger.error(f"Error loading {dataset_type} data: {str(e)}")
        raise

def main():
    """
    Main function to run the KNN classifier on Iris data.
    """
    try:
        # Set up paths
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "data"
        
        # Load training data
        train_features, train_labels = load_data(
            data_dir / "Iris_train.csv",
            is_training=True
        )
        
        # Initialize and train classifier
        classifier = KNNClassifier(num_neighbors=5, distance_metric="euclidean")
        classifier.fit(train_features, train_labels)
        
        # Load and predict on test data
        test_features = load_data(
            data_dir / "Iris_test.csv",
            is_training=False
        )
        
        # Make predictions
        predictions = classifier.predict(test_features)
        probabilities = classifier.predict_proba(test_features)
        
        # Print results
        logger.info("Predictions with confidence scores:")
        for idx, prediction in enumerate(predictions):
            confidence = probabilities[prediction][idx]
            print(f"{prediction:<15} {confidence:.6f}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()