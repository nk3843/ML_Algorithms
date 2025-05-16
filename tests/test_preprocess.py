"""
Test script for preprocessing module.
Tests PCA, normalization, and stratified sampling on the Iris dataset.

Author: [Nikhil Kumar]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple, List, Dict, Optional
import algorithms.PCA_implementation.preprocess as preprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path: Path, is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Load and prepare the Iris dataset.

    Args:
        data_path: Path to the data file
        is_training: Whether loading training or test data

    Returns:
        Tuple containing features DataFrame and target labels Series (if training)
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
        
        # Validate required columns exist
        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        features = data[feature_columns]
        
        if is_training:
            if "Species" not in data.columns:
                raise ValueError("Training data missing Species column")
            target_labels = data["Species"]
            # Validate no NaN values in training data
            if target_labels.isna().any():
                raise ValueError("Training data contains NaN values in target labels")
            return features, target_labels
        return features, None

    except Exception as e:
        logger.error(f"Error loading {dataset_type} data: {str(e)}")
        raise

def verify_data(X: pd.DataFrame, y: Optional[pd.Series] = None, dataset_type: str = "data"):
    """
    Verify data integrity and basic statistics.

    Args:
        X: Features DataFrame
        y: Optional target labels
        dataset_type: Type of dataset for logging
    """
    logger.info(f"\n{dataset_type.capitalize()} Verification:")
    logger.info(f"Shape: {X.shape}")
    logger.info("\nFeature Statistics:")
    logger.info(X.describe())
    
    if y is not None:
        logger.info("\nClass Distribution:")
        for label, count in Counter(y).items():
            logger.info(f"{label}: {count}")

def preprocess_data(
    X: pd.DataFrame,
    principal_components: Optional[np.ndarray] = None,
    is_training: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess the data using normalization and PCA.

    Args:
        X: Input features
        principal_components: PCA components (required for test data)
        is_training: Whether processing training or test data

    Returns:
        Tuple containing:
        - Transformed features
        - Principal components (only for training data)
    """
    try:
        # Normalize data
        logger.info("Normalizing data...")
        X_norm = preprocess.normalize(X, norm="Standard_Score")
        
        if is_training:
            # Perform PCA on training data
            logger.info("Performing PCA...")
            principal_components, X_pca = preprocess.pca(X_norm, n_components=3)
            
            # Log explained variance
            logger.info(f"PCA components shape: {principal_components.shape}")
            logger.info("PCA component statistics:")
            logger.info(pd.DataFrame(X_pca).describe())
            
            return X_pca, principal_components
        else:
            # Transform test data
            if principal_components is None:
                raise ValueError("Principal components required for test data")
            X_pca = X_norm @ principal_components
            return X_pca, None

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Train and evaluate the model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        # Train model
        logger.info("\nTraining Decision Tree classifier...")
        clf = DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # Make predictions
        train_predictions = clf.predict(X_train)
        test_predictions = clf.predict(X_test)
        
        # Calculate accuracies
        train_accuracy = np.mean(train_predictions == y_train)
        test_accuracy = np.mean(test_predictions == y_test)
        
        # Generate classification report
        report = classification_report(y_test, test_predictions)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, test_predictions)
        
        logger.info("\nModel Performance:")
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(report)
        logger.info("\nConfusion Matrix:")
        logger.info(cm)
        
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "predictions": test_predictions,
            "confusion_matrix": cm,
            "classification_report": report
        }

    except Exception as e:
        logger.error(f"Error in model training/evaluation: {str(e)}")
        raise

def main():
    """
    Main function to run the preprocessing and classification pipeline.
    """
    try:
        # Set up paths
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "data"
        
        # Load training data
        X_train, y_train = load_data(data_dir / "Iris_train.csv", is_training=True)
        verify_data(X_train, y_train, "training")
        
        # Preprocess training data
        X_train_pca, principal_components = preprocess_data(X_train, is_training=True)
        
        # Perform stratified sampling
        logger.info("\nPerforming stratified sampling...")
        sample_indices = preprocess.stratified_sampling(
            y_train,
            ratio=0.5,
            replace=False,
            random_state=42
        )
        
        # Get sampled data
        X_sample = X_train_pca[sample_indices]
        y_sample = y_train.iloc[sample_indices].to_numpy()
        
        # Verify sampling results
        logger.info("\nSampling Results:")
        logger.info(f"Original data size: {len(y_train)}")
        logger.info(f"Sampled data size: {len(y_sample)}")
        logger.info("\nClass distribution in sampled data:")
        for label, count in Counter(y_sample).items():
            logger.info(f"{label}: {count}")
        
        # Load test data
        X_test = load_data(data_dir / "Iris_test.csv", is_training=False)[0]  # Only get features
        verify_data(X_test, None, "test")
        
        # Preprocess test data
        X_test_pca, _ = preprocess_data(
            X_test,
            principal_components=principal_components,
            is_training=False
        )
        
        # Train model with better parameters
        logger.info("\nTraining Decision Tree classifier...")
        clf = DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        clf.fit(X_sample, y_sample)
        
        # Make predictions
        test_predictions = clf.predict(X_test_pca)
        
        # Log predictions with confidence scores
        logger.info("\nTest Data Predictions with Confidence:")
        for i, pred in enumerate(test_predictions):
            # Get probability estimates
            probs = clf.predict_proba(X_test_pca[i:i+1])[0]
            max_prob = np.max(probs)
            logger.info(f"Sample {i}: {pred} (confidence: {max_prob:.3f})")
        
        # Log prediction distribution
        logger.info("\nPrediction Distribution:")
        for label, count in Counter(test_predictions).items():
            logger.info(f"{label}: {count}")
            
        # Log feature importance
        if hasattr(clf, 'feature_importances_'):
            logger.info("\nFeature Importance:")
            for i, importance in enumerate(clf.feature_importances_):
                logger.info(f"Component {i}: {importance:.3f}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()



