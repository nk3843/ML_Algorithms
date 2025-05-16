"""
Test script for Naive Bayes Classifier implementation.
Tests the classifier on the audiology dataset.

Author: [Nikhil Kumar]
"""

import pandas as pd
import numpy as np
from algorithms.naive_bayes_implementation.naive_bayes import NaiveBayes
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
    Load and prepare the audiology dataset.

    Args:
        data_path: Path to the data file
        is_training: Whether loading training or test data (for logging)

    Returns:
        tuple: (features DataFrame, target labels Series) for training data
               (features DataFrame) for test data
    """
    try:
        dataset_type = "training" if is_training else "test"
        logger.info(f"Loading {dataset_type} data from {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find {dataset_type} data at {data_path}")
        
        data = pd.read_csv(data_path, header=None)
        
        # Separate features (0-68) and target (70)
        feature_columns = range(69)
        features = data[feature_columns]
        
        if is_training:
            if 70 not in data.columns:
                raise ValueError("Training data missing target column (70)")
            target = data[70]
            return features, target
        return features

    except Exception as e:
        logger.error(f"Error loading {dataset_type} data: {str(e)}")
        raise

def evaluate_predictions(predictions: list, probabilities: pd.DataFrame) -> None:
    """
    Evaluate and log prediction results.

    Args:
        predictions: List of predicted classes
        probabilities: DataFrame of prediction probabilities
    """
    logger.info("Predictions with confidence scores:")
    
    # Track prediction statistics
    confidence_scores = []
    prediction_counts = {}
    
    for idx, prediction in enumerate(predictions):
        confidence = probabilities[prediction][idx]
        confidence_scores.append(confidence)
        prediction_counts[prediction] = prediction_counts.get(prediction, 0) + 1
        print(f"{prediction:<30} {confidence:.6f}")
    
    # Log summary statistics
    avg_confidence = np.mean(confidence_scores)
    logger.info(f"\nPrediction Summary:")
    logger.info(f"Average confidence: {avg_confidence:.4f}")
    logger.info("Class distribution:")
    for pred_class, count in prediction_counts.items():
        logger.info(f"{pred_class:<30} {count:>3} predictions")

def main():
    """
    Main function to run the Naive Bayes classifier on audiology data.
    """
    try:
        # Set up paths
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "data"
        
        # Load training data
        train_features, train_labels = load_data(
            data_dir / "audiology_train",
            is_training=True
        )
        
        # Initialize and train classifier
        classifier = NaiveBayes()
        classifier.fit(train_features, train_labels)
        
        # Load and predict on test data
        test_features = load_data(
            data_dir / "audiology_test",
            is_training=False
        )
        
        # Make predictions
        predictions = classifier.predict(test_features)
        probabilities = classifier.predict_proba(test_features)
        
        # Evaluate and display results
        evaluate_predictions(predictions, probabilities)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()