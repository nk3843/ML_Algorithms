"""
KMeans clustering implementation.
A production-ready implementation of the KMeans clustering algorithm.

Author: [Nikhil Kumar]
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class KMeans:
    """
    KMeans clustering algorithm implementation.
    
    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization of centroids.
    n_init : int, default=10
        Number of times the algorithm will run with different centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    random_state : Optional[int], default=None
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None
    ):
        """Initialize KMeans clustering algorithm."""
        # Validate input parameters
        if n_clusters < 1:
            raise ValueError("n_clusters must be a positive integer")
        if init not in ["k-means++", "random"]:
            raise ValueError("init must be either 'k-means++' or 'random'")
        if n_init < 1:
            raise ValueError("n_init must be a positive integer")
        if max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if tol < 0:
            raise ValueError("tol must be non-negative")

        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        # Initialize random state if provided
        if random_state is not None:
            np.random.seed(random_state)

        # Initialize attributes
        self.classes_ = range(n_clusters)
        self.cluster_centers_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute Euclidean distance between two points.

        Args:
            a: First point
            b: Second point

        Returns:
            float: Euclidean distance between points
        """
        return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))

    def _initialize_centroids(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Initialize cluster centers using specified method.

        Args:
            X: Input data array

        Returns:
            List of initial cluster centers
        """
        n_samples = len(X)

        if self.init == "random":
            random_indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
            return [X[i] for i in random_indices]

        elif self.init == "k-means++":
            # Start with a random point
            idx = np.random.choice(n_samples, 1)
            centers = [X[idx]]
            
            # Select remaining centers
            for _ in range(self.n_clusters - 1):
                # Calculate distances to nearest center for each point
                distances = np.array([
                    min(self._compute_distance(x, c) for c in centers)
                    for x in X
                ])
                
                # Convert distances to probabilities
                probs = distances / np.sum(distances)
                
                # Select new center
                new_center_idx = np.random.choice(len(X), 1, p=probs)
                centers.append(X[new_center_idx][0])
            
            return centers

    def _fit_single_run(self, X: np.ndarray) -> Tuple[List[np.ndarray], float, int]:
        """
        Perform a single run of KMeans clustering.

        Args:
            X: Input data array

        Returns:
            Tuple of (cluster centers, inertia, number of iterations)
        """
        # Initialize centers
        centers = self._initialize_centroids(X)
        last_inertia = float('inf')
        n_iter = 0

        for iteration in range(self.max_iter):
            # Assign points to clusters
            clusters = [[] for _ in range(self.n_clusters)]
            inertia = 0.0

            for x in X:
                # Calculate distances to all centers
                distances = [self._compute_distance(x, center) for center in centers]
                min_dist = min(distances)
                
                # Update inertia
                inertia += min_dist ** 2
                
                # Assign to nearest cluster
                cluster_id = np.argmin(distances)
                clusters[cluster_id].append(x)

            # Check convergence
            if abs(last_inertia - inertia) < self.tol:
                break

            # Update centers
            centers = [
                np.mean(np.array(cluster), axis=0) if cluster else centers[i]
                for i, cluster in enumerate(clusters)
            ]
            
            last_inertia = inertia
            n_iter = iteration + 1

        return centers, inertia, n_iter

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'KMeans':
        """
        Fit the KMeans clustering algorithm to the data.

        Args:
            X: Training data

        Returns:
            self: The fitted KMeans instance
        """
        # Convert input to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        # Validate input data
        if len(X) < self.n_clusters:
            raise ValueError(
                f"n_samples={len(X)} should be >= n_clusters={self.n_clusters}"
            )

        # Run multiple times and keep best result
        best_inertia = float('inf')
        best_centers = None
        best_n_iter = 0

        for run in range(self.n_init):
            logger.info(f"KMeans run {run + 1}/{self.n_init}")
            centers, inertia, n_iter = self._fit_single_run(X)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform data to cluster-distance space.

        Args:
            X: Data to transform

        Returns:
            Array of distances to each cluster center
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        return np.array([
            [self._compute_distance(x, center) for center in self.cluster_centers_]
            for x in X
        ])

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.

        Args:
            X: Data to predict

        Returns:
            Array of cluster assignments
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")

        return np.argmin(self.transform(X), axis=1)

    def fit_predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute cluster centers and predict cluster index for each sample.

        Args:
            X: Training data

        Returns:
            Array of cluster assignments
        """
        return self.fit(X).predict(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute clustering and transform X to cluster-distance space.

        Args:
            X: Training data

        Returns:
            Array of distances to each cluster center
        """
        return self.fit(X).transform(X)
