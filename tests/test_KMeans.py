"""
Test script for KMeans clustering implementation.
Tests the clustering algorithm on the Iris dataset.

Author: [Nikhil Kumar]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List
from algorithms.KMeans_implementation.KMeans import KMeans

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_path: Path, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare the Iris dataset.

    Args:
        data_path: Path to the data file
        is_training: Whether loading training or test data

    Returns:
        Tuple containing features DataFrame and target labels Series
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
            return features, target_labels
        return features, None

    except Exception as e:
        logger.error(f"Error loading {dataset_type} data: {str(e)}")
        raise

def evaluate_clustering(clf: KMeans, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray) -> None:
    """
    Evaluate and log clustering results.

    Args:
        clf: Trained KMeans classifier
        X: Feature matrix
        y: True labels
        y_pred: Predicted cluster assignments
    """
    logger.info("\nClustering Results:")
    logger.info("-" * 50)
    
    # Log cluster assignments
    assignments = [(y.iloc[i], y_pred[i]) for i in range(len(y))]
    logger.info("Cluster Assignments (True Label, Predicted Cluster):")
    for true_label, pred_cluster in assignments:
        logger.info(f"{true_label:<15} -> Cluster {pred_cluster}")
    
    # Log centroids
    logger.info("\nCluster Centroids:")
    for i, centroid in enumerate(clf.cluster_centers_):
        logger.info(f"Cluster {i}: {centroid}")
    
    # Log inertia
    logger.info(f"\nInertia (Within-cluster sum of squares): {clf.inertia_:.4f}")
    
    # Calculate silhouette score if scikit-learn is available
    try:
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X, y_pred)
        logger.info(f"Silhouette Score: {silhouette_avg:.4f}")
    except ImportError:
        logger.info("Silhouette score calculation skipped (scikit-learn not available)")

def main():
    """
    Main function to run KMeans clustering on Iris data.
    """
    try:
        # Set up paths
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / "data"
        
        # Load training data
        X_train, y_train = load_data(data_dir / "Iris_train.csv", is_training=True)
        
        # Initialize and train KMeans
        n_clusters = 3
        logger.info(f"Training KMeans with {n_clusters} clusters...")
        clf = KMeans(n_clusters=n_clusters)
        y_pred = clf.fit_predict(X_train)
        
        # Evaluate clustering
        evaluate_clustering(clf, X_train, y_train, y_pred)
        
        # Load and transform test data
        X_test, _ = load_data(data_dir / "Iris_test.csv", is_training=False)
        
        # Get cluster distances and predictions for test data
        logger.info("\nTest Data Results:")
        logger.info("-" * 50)
        
        # Calculate distances to centroids
        distances = clf.transform(X_test)
        logger.info("Distances to cluster centroids:")
        for i, dist in enumerate(distances):
            logger.info(f"Sample {i}: {dist}")
        
        # Get cluster assignments
        test_predictions = clf.predict(X_test)
        logger.info("\nTest data cluster assignments:")
        for i, cluster in enumerate(test_predictions):
            logger.info(f"Sample {i} -> Cluster {cluster}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
