'''
ManGo - Machine Gnostics Library
Copyright (C) 2025  ManGo Team

This work is licensed under the terms of the GNU General Public License version 3.0.

Author: Nirmal Parmar
Date: 2025-10-01
Description:

'''

import numpy as np
from src.magcal import GnosticsCharacteristics, DataConversion, ScaleParam, GnosticsWeights 

// ...existing code...

class EvaluationMetrics:
    """
    Class to calculate evaluation metrics for robust regression models.
    Implements RobR², GMMFE, DivI, and EvalMet calculations.
    """
    
    def __init__(self, y_true, y_pred, weights=None):
        """
        Initialize the evaluation metrics calculator.
        
        Args:
            y_true (np.ndarray): True target values
            y_pred (np.ndarray): Predicted target values
            weights (np.ndarray, optional): Weights for each observation. Defaults to None.
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.weights = np.ones_like(y_true) if weights is None else np.asarray(weights)
        self.N = len(y_true)
        
    def calculate_rob_r2(self):
        """Calculate the Weighted R-square (RobR²)."""
        errors = self.y_true - self.y_pred
        weighted_errors_squared = np.sum(self.weights * (errors ** 2))
        weighted_total_variance = np.sum(self.weights * (self.y_true - np.mean(self.y_true)) ** 2)
        
        rob_r2 = 1 - (weighted_errors_squared / weighted_total_variance)
        return rob_r2
    
    def calculate_gmmfe(self):
        """Calculate the Geometric Mean of Multiplicative Fitting Errors (GMMFE)."""
        ratio = self.y_true / self.y_pred
        log_sum = np.sum(np.abs(np.log(ratio))) / self.N
        gmmfe = np.exp(log_sum)
        return gmmfe
    
    def calculate_divi(self):
        """Calculate the Divergence of Information (DivI)."""
        I_true = self._calculate_information(self.y_true)
        I_pred = self._calculate_information(self.y_pred)
        divi = np.sum(I_true / I_pred) / self.N
        return divi
    
    def _calculate_information(self, y):
        """Helper method to calculate information content."""
        # This is a simplified version - you might want to implement
        # your specific information calculation method
        return np.abs(y) + 1e-10  # Adding small constant to avoid division by zero
    
    def calculate_evalmet(self):
        """Calculate the overall evaluation metric (EvalMet)."""
        rob_r2 = self.calculate_rob_r2()
        gmmfe = self.calculate_gmmfe()
        divi = self.calculate_divi()
        
        evalmet = rob_r2 / (gmmfe * divi)
        return evalmet
    
    def generate_report(self):
        """Generate a complete evaluation report."""
        rob_r2 = self.calculate_rob_r2()
        gmmfe = self.calculate_gmmfe()
        divi = self.calculate_divi()
        evalmet = self.calculate_evalmet()
        
        return {
            'RobR²': rob_r2,
            'GMMFE': gmmfe,
            'DivI': divi,
            'EvalMet': evalmet
        }