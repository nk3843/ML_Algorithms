"""
Test script for model evaluation metrics.
Tests various evaluation metrics on the Iris dataset using a Decision Tree classifier.

Author: [Nikhil Kumar]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from algorithms.evaluation_implementation.ClassificationMetrics import ClassificationMetrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare the Iris dataset.

    Args:
        data_path: Path to the data file

    Returns:
        Tuple containing features DataFrame and target labels Series

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        logger.info(f"Loading data from {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find data at {data_path}")
        
        data = pd.read_csv(data_path)
        
        # Define feature columns
        feature_columns = [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
        
        # Validate required columns exist
        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        features = data[feature_columns]
        
        if "Species" not in data.columns:
            raise ValueError("Data missing Species column")
        
        target_labels = data["Species"]
        
        # Log data statistics
        logger.info(f"Data shape: {data.shape}")
        logger.info("\nClass distribution:")
        for label, count in target_labels.value_counts().items():
            logger.info(f"{label}: {count}")
            
        return features, target_labels

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 2,
    criterion: str = "entropy"
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.

    Args:
        X: Training features
        y: Training labels
        max_depth: Maximum tree depth
        criterion: Splitting criterion

    Returns:
        Trained DecisionTreeClassifier

    Raises:
        ValueError: If input data is invalid
    """
    try:
        logger.info(f"\nTraining Decision Tree (max_depth={max_depth}, criterion={criterion})...")
        
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y must match")
        
        clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=42
        )
        clf.fit(X, y)
        
        # Log feature importance
        logger.info("\nFeature Importance:")
        for feature, importance in zip(X.columns, clf.feature_importances_):
            logger.info(f"{feature}: {importance:.3f}")
            
        return clf

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(
    clf: DecisionTreeClassifier,
    X: pd.DataFrame,
    y: pd.Series
) -> Dict:
    """
    Evaluate model performance using various metrics.

    Args:
        clf: Trained classifier
        X: Features
        y: True labels

    Returns:
        Dictionary containing evaluation metrics

    Raises:
        ValueError: If input data is invalid
    """
    try:
        # Get predictions and probabilities
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)
        
        # Convert probabilities to DataFrame
        prob_df = pd.DataFrame(
            probabilities,
            columns=clf.classes_,
            index=X.index
        )
        
        # Initialize evaluation metrics
        metrics = ClassificationMetrics(predictions, y, prob_df)
        
        # Get summary of all metrics
        summary = metrics.get_summary()
        
        # Log results
        logger.info("\nPer-class Metrics:")
        for target, metrics_dict in summary["per_class"].items():
            logger.info(f"\n{target}:")
            for metric, value in metrics_dict.items():
                if value is not None:  # Skip None values (e.g., AUC when not available)
                    logger.info(f"{metric}: {value:.3f}")
        
        logger.info("\nAverage Metrics:")
        for avg_type, metrics_dict in summary["averages"].items():
            logger.info(f"\n{avg_type}:")
            for metric, value in metrics_dict.items():
                logger.info(f"{metric}: {value:.3f}")
        
        return summary

    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def main():
    """
    Main function to run the evaluation pipeline.
    """
    try:
        # Set up paths
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "data"
        
        # Load data
        X, y = load_data(data_dir / "Iris_train.csv")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        # Train model
        clf = train_model(X_train, y_train)
        
        # Evaluate model on test set
        logger.info("\nEvaluating on test set...")
        results = evaluate_model(clf, X_test, y_test)
        
        # Log final summary
        logger.info("\nEvaluation Summary:")
        best_class = max(
            results["per_class"].items(),
            key=lambda x: x[1]["f1"]
        )[0]
        logger.info(f"Best performing class: {best_class}")
        logger.info(f"Best F1 score: {results['per_class'][best_class]['f1']:.3f}")
        
        # Log overall accuracy
        logger.info(f"\nOverall accuracy: {results['accuracy']:.3f}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()




