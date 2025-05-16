"""
Preprocessing module for machine learning algorithms.
Includes PCA, normalization, and stratified sampling implementations.

Author: [Nikhil Kumar]
"""

import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from collections import Counter
from typing import Union, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass

def pca(
    X: np.ndarray,
    n_components: int = 5,
    center: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Args:
        X: Input matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        center: Whether to center the data before PCA

    Returns:
        Tuple containing:
        - principal_components: Top n_components principal components
        - X_pca: Transformed data

    Raises:
        PreprocessingError: If input validation fails
    """
    try:
        # Input validation
        if not isinstance(X, np.ndarray):
            raise PreprocessingError("Input X must be a numpy array")
        if n_components <= 0 or n_components > X.shape[1]:
            raise PreprocessingError(
                f"n_components must be between 1 and {X.shape[1]}"
            )

        # Center the data if requested
        if center:
            X = X - np.mean(X, axis=0)

        # Perform SVD
        U, s, Vh = svd(X, full_matrices=False)
        
        # Get principal components
        principal_components = Vh[:n_components].T
        
        # Transform data
        X_pca = X @ principal_components

        logger.info(
            f"PCA completed: {n_components} components extracted from {X.shape[1]} features"
        )
        
        return principal_components, X_pca

    except Exception as e:
        logger.error(f"Error in PCA: {str(e)}")
        raise PreprocessingError(f"PCA failed: {str(e)}")

def normalize_vector(
    x: np.ndarray,
    norm: str = "Min-Max"
) -> np.ndarray:
    """
    Normalize a vector using the specified normalization method.

    Args:
        x: Input vector
        norm: Normalization method
            - "Min-Max": Scale to [0,1]
            - "L1": L1 normalization
            - "L2": L2 normalization
            - "Standard_Score": Z-score normalization

    Returns:
        Normalized vector

    Raises:
        PreprocessingError: If invalid normalization method
    """
    try:
        x = np.asarray(x, dtype=np.float64)
        
        if norm == "Min-Max":
            x_min, x_max = np.min(x), np.max(x)
            if x_max == x_min:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min)
            
        elif norm == "L1":
            x_sum = np.sum(np.abs(x))
            return x / x_sum if x_sum != 0 else x
            
        elif norm == "L2":
            x_norm = np.linalg.norm(x)
            return x / x_norm if x_norm != 0 else x
            
        elif norm == "Standard_Score":
            x_mean = np.mean(x)
            x_std = np.std(x)
            return (x - x_mean) / x_std if x_std != 0 else x
            
        else:
            raise PreprocessingError(
                f"Unknown normalization method: {norm}. "
                "Choose from: Min-Max, L1, L2, Standard_Score"
            )

    except Exception as e:
        logger.error(f"Error in vector normalization: {str(e)}")
        raise PreprocessingError(f"Vector normalization failed: {str(e)}")

def normalize(
    X: Union[np.ndarray, List],
    norm: str = "Min-Max",
    axis: int = 1
) -> np.ndarray:
    """
    Normalize a matrix along the specified axis.

    Args:
        X: Input matrix
        norm: Normalization method
        axis: Axis along which to normalize (0 for rows, 1 for columns)

    Returns:
        Normalized matrix

    Raises:
        PreprocessingError: If invalid parameters
    """
    try:
        # Convert input to numpy array
        X = np.asarray(X, dtype=np.float64)
        
        # Input validation
        if axis not in [0, 1]:
            raise PreprocessingError("axis must be 0 or 1")
            
        # Create copy to avoid modifying original data
        X_norm = deepcopy(X)
        
        if axis == 1:
            # Normalize columns
            for col in range(X.shape[1]):
                X_norm[:, col] = normalize_vector(X[:, col], norm)
        else:
            # Normalize rows
            for row in range(X.shape[0]):
                X_norm[row] = normalize_vector(X[row], norm)
                
        logger.info(f"Matrix normalized using {norm} normalization along axis {axis}")
        return X_norm

    except Exception as e:
        logger.error(f"Error in matrix normalization: {str(e)}")
        raise PreprocessingError(f"Matrix normalization failed: {str(e)}")

def stratified_sampling(
    y: Union[np.ndarray, List],
    ratio: float,
    replace: bool = True,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Perform stratified sampling on the data.

    Args:
        y: Array of class labels
        ratio: Sampling ratio (0 < ratio < 1)
        replace: Whether to sample with replacement
        random_state: Random state for reproducibility

    Returns:
        Indices of sampled points

    Raises:
        PreprocessingError: If invalid parameters
    """
    try:
        # Input validation
        if not 0 < ratio < 1:
            raise PreprocessingError("ratio must be between 0 and 1")
            
        # Convert input to numpy array
        y = np.asarray(y)
        
        # Set random state if provided
        if random_state is not None:
            np.random.seed(random_state)
            
        # Get unique labels and their counts
        unique_labels = np.unique(y)
        samples = []
        
        # Sample from each class
        for label in unique_labels:
            # Get indices for current class
            class_indices = np.where(y == label)[0]
            
            # Calculate number of samples for this class
            n_samples = int(np.ceil(ratio * len(class_indices)))
            
            # Sample indices
            class_samples = np.random.choice(
                class_indices,
                size=n_samples,
                replace=replace
            )
            samples.extend(class_samples)
            
        logger.info(
            f"Stratified sampling completed: {len(samples)} samples selected "
            f"from {len(y)} total samples"
        )
        
        return np.array(samples)

    except Exception as e:
        logger.error(f"Error in stratified sampling: {str(e)}")
        raise PreprocessingError(f"Stratified sampling failed: {str(e)}")
