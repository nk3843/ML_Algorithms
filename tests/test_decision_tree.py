"""
Test script for Decision Tree Classifier implementation.
Tests the classifier on the Iris dataset.

Author: [Nikhil Kumar]
"""

import pandas as pd
from algorithms.decision_tree_implementation.decision_tree import DecisionTree
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
        
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find {dataset_type} data at {data_path}")
        
        data = pd.read_csv(data_path)
        
        # Define feature columns
        feature_columns = [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
        features = data[feature_columns]
        
        if is_training:
            if "Species" not in data.columns:
                raise ValueError("Training data missing Species column")
            target_labels = data["Species"]
            return features, target_labels
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
        print(f"{prediction:<20} {confidence:.6f}")
    
    # Log summary statistics
    logger.info("\nPrediction Summary:")
    logger.info(f"Average confidence: {sum(confidence_scores)/len(confidence_scores):.4f}")
    logger.info("Class distribution:")
    for pred_class, count in prediction_counts.items():
        logger.info(f"{pred_class:<20} {count:>3} predictions")

def main():
    """
    Main function to run the Decision Tree classifier on Iris data.
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
        classifier = DecisionTree()
        classifier.fit(train_features, train_labels)
        
        # Load and predict on test data
        test_features = load_data(
            data_dir / "Iris_test.csv",
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