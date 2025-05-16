"""
Evaluation metrics implementation for classification tasks.
Supports binary and multi-class classification evaluation.

Author: [Nikhil Kumar]
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class ClassificationMetrics:
    """
    Classification metrics implementation for evaluating model performance.
    
    Supports:
    - Binary and multi-class classification
    - Precision, Recall, F1-score
    - AUC-ROC
    - Confusion matrix
    - Various averaging methods (macro, micro, weighted)
    """

    def __init__(
        self,
        predictions: Union[List, np.ndarray],
        actuals: Union[List, np.ndarray],
        pred_proba: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the evaluation metrics calculator.

        Args:
            predictions: List of predicted classes
            actuals: List of ground truth labels
            pred_proba: DataFrame of prediction probabilities for each class

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Convert inputs to numpy arrays
            self.predictions = np.array(predictions)
            self.actuals = np.array(actuals)
            
            # Validate inputs
            if len(self.predictions) != len(self.actuals):
                raise ValueError("Predictions and actuals must have the same length")
            
            # Set prediction probabilities
            self.pred_proba = pred_proba
            
            # Get unique classes
            if pred_proba is not None:
                self.classes_ = list(pred_proba.columns)
            else:
                self.classes_ = list(set(self.predictions) | set(self.actuals))
            
            # Initialize metrics
            self.confusion_matrix = None
            self.accuracy = None
            
            logger.info(f"Initialized with {len(self.classes_)} classes: {self.classes_}")
            
        except Exception as e:
            logger.error(f"Error initializing ClassificationMetrics: {str(e)}")
            raise

    def _compute_confusion_matrix(self) -> None:
        """
        Compute confusion matrix for each class.
        
        Updates:
        - self.confusion_matrix: Dict of confusion matrices per class
        - self.accuracy: Overall accuracy
        """
        try:
            # Calculate overall accuracy
            correct = self.predictions == self.actuals
            self.accuracy = float(Counter(correct)[True]) / len(correct)
            
            # Initialize confusion matrix dictionary
            self.confusion_matrix = {}
            
            # Calculate metrics for each class
            for label in self.classes_:
                tp = Counter(correct & (self.predictions == label))[True]
                fp = Counter((self.actuals != label) & (self.predictions == label))[True]
                tn = Counter(correct & (self.predictions != label))[True]
                fn = Counter((self.actuals == label) & (self.predictions != label))[True]
                
                self.confusion_matrix[label] = {
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn
                }
                
            logger.debug("Confusion matrix computed successfully")
            
        except Exception as e:
            logger.error(f"Error computing confusion matrix: {str(e)}")
            raise

    def get_accuracy(self) -> float:
        """
        Get overall accuracy.

        Returns:
            float: Accuracy score
        """
        if self.confusion_matrix is None:
            self._compute_confusion_matrix()
        return self.accuracy

    def get_precision(
        self,
        target: Optional[str] = None,
        average: str = "macro"
    ) -> float:
        """
        Compute precision score.

        Args:
            target: Target class. If None, return average precision
            average: Averaging method ("macro", "micro", "weighted")

        Returns:
            float: Precision score

        Raises:
            ValueError: If invalid target class or averaging method
        """
        try:
            if self.confusion_matrix is None:
                self._compute_confusion_matrix()
                
            if target in self.classes_:
                # Calculate precision for specific class
                tp = self.confusion_matrix[target]["TP"]
                fp = self.confusion_matrix[target]["FP"]
                return float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
                
            elif target is None:
                if average == "micro":
                    return self.get_accuracy()
                    
                # Calculate average precision
                total_precision = 0.0
                n_samples = len(self.actuals)
                
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fp = self.confusion_matrix[label]["FP"]
                    prec_label = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
                    
                    # Calculate weight based on averaging method
                    if average == "macro":
                        weight = 1.0 / len(self.classes_)
                    elif average == "weighted":
                        weight = Counter(self.actuals)[label] / float(n_samples)
                    else:
                        raise ValueError(f"Invalid averaging method: {average}")
                        
                    total_precision += prec_label * weight
                    
                return total_precision
                
            else:
                raise ValueError(f"Invalid target class: {target}")
                
        except Exception as e:
            logger.error(f"Error computing precision: {str(e)}")
            raise

    def get_recall(
        self,
        target: Optional[str] = None,
        average: str = "macro"
    ) -> float:
        """
        Compute recall score.

        Args:
            target: Target class. If None, return average recall
            average: Averaging method ("macro", "micro", "weighted")

        Returns:
            float: Recall score

        Raises:
            ValueError: If invalid target class or averaging method
        """
        try:
            if self.confusion_matrix is None:
                self._compute_confusion_matrix()
                
            if target in self.classes_:
                # Calculate recall for specific class
                tp = self.confusion_matrix[target]["TP"]
                fn = self.confusion_matrix[target]["FN"]
                return float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
                
            elif target is None:
                if average == "micro":
                    return self.get_accuracy()
                    
                # Calculate average recall
                total_recall = 0.0
                n_samples = len(self.actuals)
                
                for label in self.classes_:
                    tp = self.confusion_matrix[label]["TP"]
                    fn = self.confusion_matrix[label]["FN"]
                    rec_label = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
                    
                    # Calculate weight based on averaging method
                    if average == "macro":
                        weight = 1.0 / len(self.classes_)
                    elif average == "weighted":
                        weight = Counter(self.actuals)[label] / float(n_samples)
                    else:
                        raise ValueError(f"Invalid averaging method: {average}")
                        
                    total_recall += rec_label * weight
                    
                return total_recall
                
            else:
                raise ValueError(f"Invalid target class: {target}")
                
        except Exception as e:
            logger.error(f"Error computing recall: {str(e)}")
            raise

    def get_f1(
        self,
        target: Optional[str] = None,
        average: str = "macro"
    ) -> float:
        """
        Compute F1 score.

        Args:
            target: Target class. If None, return average F1
            average: Averaging method ("macro", "micro", "weighted")

        Returns:
            float: F1 score
        """
        try:
            precision = self.get_precision(target, average)
            recall = self.get_recall(target, average)
            
            if precision + recall == 0:
                return 0.0
                
            return (2 * precision * recall) / (precision + recall)
            
        except Exception as e:
            logger.error(f"Error computing F1 score: {str(e)}")
            raise

    def get_auc(self, target: str) -> float:
        """
        Compute AUC-ROC score for a specific class.

        Args:
            target: Target class

        Returns:
            float: AUC-ROC score

        Raises:
            ValueError: If prediction probabilities not provided or invalid target
        """
        try:
            if self.pred_proba is None:
                raise ValueError("Prediction probabilities required for AUC calculation")
                
            if target not in self.classes_:
                raise ValueError(f"Invalid target class: {target}")
                
            # Sort by prediction probability
            order = np.argsort(self.pred_proba[target])[::-1]
            
            # Initialize counters
            tp = fp = 0
            fn = Counter(self.actuals)[target]
            tn = len(self.actuals) - fn
            
            # Calculate AUC
            auc_score = 0.0
            prev_fpr = 0.0
            
            for idx in order:
                if self.actuals[idx] == target:
                    tp += 1
                    fn -= 1
                    tpr = float(tp) / (tp + fn)
                else:
                    fp += 1
                    tn -= 1
                    fpr = float(fp) / (fp + tn)
                    auc_score += tpr * (fpr - prev_fpr)
                    prev_fpr = fpr
                    
            return auc_score
            
        except Exception as e:
            logger.error(f"Error computing AUC: {str(e)}")
            raise

    def get_summary(self) -> Dict:
        """
        Get summary of all metrics.

        Returns:
            Dict containing all evaluation metrics
        """
        try:
            summary = {
                "accuracy": self.get_accuracy(),
                "per_class": {},
                "averages": {
                    "macro": {},
                    "micro": {},
                    "weighted": {}
                }
            }
            
            # Calculate per-class metrics
            for label in self.classes_:
                summary["per_class"][label] = {
                    "precision": self.get_precision(label),
                    "recall": self.get_recall(label),
                    "f1": self.get_f1(label),
                    "auc": self.get_auc(label) if self.pred_proba is not None else None
                }
            
            # Calculate average metrics
            for avg in ["macro", "micro", "weighted"]:
                summary["averages"][avg] = {
                    "precision": self.get_precision(average=avg),
                    "recall": self.get_recall(average=avg),
                    "f1": self.get_f1(average=avg)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise


