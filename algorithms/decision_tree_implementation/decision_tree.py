"""
Decision Tree Classifier Implementation

A custom implementation of a Decision Tree classifier that supports both
gini and entropy criteria for splitting nodes. This implementation follows
a similar API to scikit-learn's DecisionTreeClassifier.

Features:
- Supports gini and entropy impurity measures
- Configurable max depth and minimum samples for splitting
- Probability predictions
- Binary tree structure using dictionary representation

Author: [Nikhil Kumar]
"""
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Union, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DecisionTree:

    def __init__(
            self,
            criterion="gini",
            max_depth=8,
            min_impurity_decrease=0,
            min_samples_split=2
        ):
        """
        Initialize the Decision Tree classifier.

        Args:
            criterion: {"gini", "entropy"}, default="gini"
                The function to measure the quality of a split
            max_depth: int, default=8
                Maximum depth of the tree
            min_impurity_decrease: float, default=0
                Minimum required decrease in impurity for splitting
            min_samples_split: int, default=2
                Minimum samples required to split a node
        """
        if criterion not in ["gini", "entropy"]:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = int(max_depth)
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_split = int(min_samples_split)
        self.tree: Dict = {}
        self.classes_: List = []
        logger.info(f"Initialized DecisionTree with {criterion} criterion")

    def impurity(self, labels):
        """
        Calculate impurity (unweighted) for a set of labels.

        Args:
            labels: Array-like of labels

        Returns:
            float: Impurity score
        """
        stats = Counter(labels)
        N = float(len(labels))
        
        if N == 0:
            return 0
        
        if self.criterion == "gini":
            return 1.0 - sum((stats[label]/N)**2 for label in stats)
        else:
            return -sum((stats[label]/N) * (np.log2(stats[label]/N)) for label in stats)

    def find_best_split(self, pop: np.ndarray, X: pd.DataFrame, labels: np.ndarray) -> Tuple:
        """
        Find the best split for a node.
        """
        best_feature = None
        best_impurity = float('inf')
        best_split = 0.0
        best_indices = None
        best_impurities = None

        try:
            N = len(pop)
            current_impurity = self.impurity(labels[pop]) * N

            for feature in X.keys():
                feature_values = X[feature].iloc[pop].values
                unique_values = np.unique(feature_values)
                
                # Try splitting on each value
                for split_val in unique_values:
                    left_mask = feature_values < split_val
                    right_mask = ~left_mask
                    
                    left_indices = pop[left_mask]
                    right_indices = pop[right_mask]
                    
                    # Skip if split creates empty node
                    if len(left_indices) == 0 or len(right_indices) == 0:
                        continue

                    # Calculate impurity for split
                    left_impurity = self.impurity(labels[left_indices]) * len(left_indices)
                    right_impurity = self.impurity(labels[right_indices]) * len(right_indices)
                    split_impurity = left_impurity + right_impurity

                    # Update best split if this one is better
                    if split_impurity < best_impurity:
                        best_feature = feature
                        best_impurity = split_impurity
                        best_split = split_val
                        best_indices = [left_indices, right_indices]
                        best_impurities = [left_impurity, right_impurity]

            # If no improvement in impurity, make this a leaf node
            if best_impurity >= current_impurity:
                return (None, current_impurity, None, [pop], [current_impurity])

            return (best_feature, best_impurity, best_split, best_indices, best_impurities)

        except Exception as e:
            logger.error(f"Error in finding best split: {str(e)}")
            raise
                    
    def fit(self, X: pd.DataFrame, y: Union[List, np.ndarray, pd.Series]) -> 'DecisionTree':
        """
        Build the decision tree from training data.
        """
        try:
            self.classes_ = list(set(y))
            labels = np.array(y)
            N = len(y)

            # Initialize tree structure
            self.tree = {}
            population = {0: np.array(range(N))}
            impurity = {0: self.impurity(labels[population[0]]) * N}

            level = 0
            nodes = [0]

            while level < self.max_depth and nodes:
                next_nodes = []
                for node in nodes:
                    current_pop = population[node]
                    
                    # Don't split if too few samples
                    if len(current_pop) < self.min_samples_split:
                        self.tree[node] = Counter(labels[current_pop])
                        continue

                    # Find the best split
                    split_info = self.find_best_split(current_pop, X, labels)
                    
                    # If no valid split found or not enough improvement, make leaf
                    if split_info[0] is None or (impurity[node] - split_info[1]) < self.min_impurity_decrease * N:
                        self.tree[node] = Counter(labels[current_pop])
                        continue

                    # Create split
                    self.tree[node] = (split_info[0], split_info[2])
                    left_node, right_node = node * 2 + 1, node * 2 + 2
                    
                    # Add child nodes to next level
                    next_nodes.extend([left_node, right_node])
                    population[left_node] = split_info[3][0]
                    population[right_node] = split_info[3][1]
                    impurity[left_node] = split_info[4][0]
                    impurity[right_node] = split_info[4][1]

                nodes = next_nodes
                level += 1

            logger.info(f"Successfully built tree with depth {level}")
            return self

        except Exception as e:
            logger.error(f"Error during fitting: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> List:
        """
        Predict class labels for samples in X.
        """
        if not self.tree:
            raise ValueError("Tree not fitted. Call fit before predicting")

        try:
            predictions = []
            for i in range(len(X)):
                node = 0
                while node in self.tree:  # Check if node exists in tree
                    if isinstance(self.tree[node], Counter):
                        # Get most common class for this leaf node
                        label = max(self.tree[node].items(), key=lambda x: x[1])[0]
                        predictions.append(label)
                        break
                    else:
                        feature, split_value = self.tree[node]
                        next_node = node * 2 + 1 if X[feature].iloc[i] < split_value else node * 2 + 2
                        if next_node not in self.tree:
                            # If child doesn't exist, treat current node as leaf
                            label = max(Counter(self.classes_).items(), key=lambda x: x[1])[0]
                            predictions.append(label)
                            break
                        node = next_node

            return predictions

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities for samples in X.
        """
        if not self.tree:
            raise ValueError("Tree not fitted. Call fit before predicting probabilities")

        try:
            predictions = []
            for i in range(len(X)):
                node = 0
                while node in self.tree:  # Check if node exists in tree
                    if isinstance(self.tree[node], Counter):
                        total = float(sum(self.tree[node].values()))
                        probs = {key: self.tree[node].get(key, 0) / total for key in self.classes_}
                        predictions.append(probs)
                        break
                    else:
                        feature, split_value = self.tree[node]
                        next_node = node * 2 + 1 if X[feature].iloc[i] < split_value else node * 2 + 2
                        if next_node not in self.tree:
                            # If child doesn't exist, use current node's distribution
                            counts = Counter(self.classes_)
                            total = float(sum(counts.values()))
                            probs = {key: counts.get(key, 0) / total for key in self.classes_}
                            predictions.append(probs)
                            break
                        node = next_node

            return pd.DataFrame(predictions, columns=self.classes_)

        except Exception as e:
            logger.error(f"Error during probability prediction: {str(e)}")
            raise
