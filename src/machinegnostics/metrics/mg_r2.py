'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar


'''

from machinegnostics.magcal.util.logging import get_logger
import logging
import numpy as np

class EvaluationMetrics:
    """
    Class to calculate evaluation metrics for robust regression models.

    This class provides methods to calculate various (gnostic) evaluation metrics for robust regression models, including:
    - RobR²: Weighted R-squared, a robust version of the coefficient of determination.
    - GMMFE: Geometric Mean of Multiplicative Fitting Errors, a measure of the geometric mean of fitting errors.
    - DivI: Divergence of Information, a metric to quantify the divergence between true and predicted values.
    - EvalMet: An overall evaluation metric combining RobR², GMMFE, and DivI.

    Attributes:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        weights (np.ndarray): Weights for each observation. Defaults to an array of ones if not provided.
        N (int): Number of observations.
        logger (logging.Logger): Logger instance for logging messages.

    Methods:
        calculate_rob_r2():
            Calculate the Weighted R-squared (RobR²).
        calculate_gmmfe():
            Calculate the Geometric Mean of Multiplicative Fitting Errors (GMMFE).
        calculate_divi():
            Calculate the Divergence of Information (DivI).
        calculate_evalmet():
            Calculate the overall evaluation metric (EvalMet).
        generate_report():
            Generate a complete evaluation report containing all metrics.
    
    Example:
        
        >>> from machinegnostics.metrics import EvaluationMetrics
        >>> y_true = np.array([3, -0.5, 2, 7])
        >>> y_pred = np.array([2.5, 0.0, 2, 8])
        >>> evaluator = EvaluationMetrics(y_true, y_pred, verbose=True)
        >>> report = evaluator.generate_report()
        >>> print(report)
    """
    
    def __init__(self, 
                 y_true: np.ndarray, 
                 y_pred: np.ndarray, 
                 weights=None,
                 verbose: bool = False):
        """
        Initialize the evaluation metrics calculator.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
            weights (np.ndarray, optional): Weights for each observation. Defaults to None.
            verbose (bool, optional): If True, enables detailed logging. Defaults to False.
        """
        self.logger = get_logger('EvaluationMetrics', level=logging.WARNING if not verbose else logging.INFO)
        self.logger.info("Initializing EvaluationMetrics...")
        self.y_true = np.asarray(y_true).ravel()
        self.y_pred = np.asarray(y_pred).ravel()
        self.weights = np.ones_like(y_true) if weights is None else np.asarray(weights)
        self.N = len(y_true)
        
    def calculate_rob_r2(self):
        """Calculate the Weighted R-square (RobR²).

        This metric measures the proportion of variance in the dependent variable that is predictable from the independent variables, using weighted observations.

        Returns:
            float: The RobR² value, where 1 indicates perfect prediction and values closer to 0 indicate poor prediction.
        """
        self.logger.info("Calculating RobR²...")
        errors = self.y_true - self.y_pred
        weighted_errors_squared = np.sum(self.weights * (errors ** 2))
        weighted_total_variance = np.sum(self.weights * (self.y_true - np.mean(self.y_true)) ** 2)
        
        rob_r2 = 1 - (weighted_errors_squared / weighted_total_variance)
        self.logger.info(f"RobR² calculated: {rob_r2}")
        return rob_r2
    
    def calculate_gmmfe(self):
        """Calculate the Geometric Mean of Multiplicative Fitting Errors (GMMFE).

        This metric calculates the geometric mean of the multiplicative fitting errors between true and predicted values.

        Returns:
            float: The GMMFE value, where values closer to 1 indicate better fitting.
        """
        self.logger.info("Calculating GMMFE...")
        ratio = self.y_true / (self.y_pred + 1e-10)
        # avoid invalid value for log
        ratio = np.where(ratio <= 0, 1e-10, ratio)
        log_sum = np.sum(np.abs(np.log(ratio))) / self.N
        gmmfe = np.exp(log_sum)
        self.logger.info(f"GMMFE calculated: {gmmfe}")
        return gmmfe
    
    def calculate_divi(self):
        """Calculate the Divergence of Information (DivI).

        This metric quantifies the divergence between the information content of true and predicted values.

        Returns:
            float: The DivI value, where lower values indicate less divergence and better predictions.
        """
        self.logger.info("Calculating DivI...")
        I_true = self._calculate_information(self.y_true)
        I_pred = self._calculate_information(self.y_pred)
        divi = np.sum(I_true / I_pred) / self.N
        self.logger.info(f"DivI calculated: {divi}")
        return divi
    
    def _calculate_information(self, y):
        """Helper method to calculate information content.""" # NOTE place holder
        self.logger.info("Calculating information content...")
        # This is a simplified version - you might want to implement
        return np.abs(y) + 1e-10  # Adding small constant to avoid division by zero
    
    def calculate_evalmet(self):
        """Calculate the overall evaluation metric (EvalMet).

        This metric combines RobR², GMMFE, and DivI to provide an overall evaluation of the model's performance.

        Returns:
            float: The EvalMet value, where higher values indicate better overall performance.
        """
        self.logger.info("Calculating EvalMet...")
        rob_r2 = self.calculate_rob_r2()
        gmmfe = self.calculate_gmmfe()
        divi = self.calculate_divi()
        
        evalmet = rob_r2 / (gmmfe * divi)
        self.logger.info(f"EvalMet calculated: {evalmet}")
        return evalmet
    
    def generate_report(self) -> dict:
        """Generate a complete evaluation report.

        This method calculates all evaluation metrics (RobR², GMMFE, DivI, EvalMet) and returns them in a dictionary format.

        Returns:
            dict: A dictionary containing all evaluation metrics with their respective values.
        """
        self.logger.info("Generating evaluation report...")
        rob_r2 = self.calculate_rob_r2()
        gmmfe = self.calculate_gmmfe()
        divi = self.calculate_divi()
        evalmet = self.calculate_evalmet()
        self.logger.info("Evaluation report generated.")
        return {
            'RobR²': rob_r2,
            'GMMFE': gmmfe,
            'DivI': divi,
            'EvalMet': evalmet
        }
